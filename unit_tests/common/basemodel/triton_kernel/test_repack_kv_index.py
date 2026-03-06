import torch
import pytest
from lightllm.utils.log_utils import init_logger
from lightllm.common.basemodel.triton_kernel.repack_kv_index import repack_kv_index, paged_repack_kv_index
from lightllm.utils.envs_utils import get_page_size

logger = init_logger(__name__)

seed = 42
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@pytest.mark.parametrize(
    "batch, max_seq_len",
    [(a, b) for a in [1, 16, 32, 128, 512] for b in [16, 32, 512, 2048]],
)
def test_repack_kv_index(batch, max_seq_len):
    def repack_kv_ref(req_to_token_indexs, b_req_idx, b_seq_len, b_start_loc, output):
        for b, sl, start in zip(b_req_idx, b_seq_len, b_start_loc):
            output[start : start + sl] = req_to_token_indexs[b][:sl]

    BATCH, MAX_SEQ_LEN = batch, max_seq_len
    rand_idx = torch.randperm(2 * MAX_SEQ_LEN * BATCH).cuda().int()
    b_req_idx = torch.randperm(BATCH).cuda().int()
    b_seq_len = torch.randint(1, MAX_SEQ_LEN, (BATCH,)).cuda().int()
    req_to_token_indexs = torch.zeros((2 * BATCH, 2 * MAX_SEQ_LEN)).cuda().int()
    b_start_loc = (
        torch.cat([torch.zeros([1], device=b_seq_len.device, dtype=b_seq_len.dtype), b_seq_len[0:-1].cumsum(0)])
        .cuda()
        .int()
    )

    output = torch.zeros((b_seq_len.sum(),)).cuda().int()
    ref = torch.zeros((b_seq_len.sum(),)).cuda().int()
    for b, sl, start in zip(b_req_idx, b_seq_len, b_start_loc):
        req_to_token_indexs[b][:sl] = rand_idx[start : start + sl]

    repack_kv_ref(req_to_token_indexs, b_req_idx, b_seq_len, b_start_loc, ref)
    repack_kv_index(req_to_token_indexs, b_req_idx, b_seq_len, b_start_loc, MAX_SEQ_LEN, output)
    assert torch.allclose(output.float(), ref.float())


@pytest.mark.parametrize(
    "batch, max_seq_len, page_size",
    [
        (1, 16, 4),
        (8, 32, 4),
        (16, 128, 8),
    ],
)
def test_paged_repack_kv_index(batch, max_seq_len, page_size, monkeypatch):
    def repack_page_kv_ref(req_to_token_indexs, b_req_idx, b_page_len, b_start_loc, output, page_size):
        for b, sl, start in zip(b_req_idx, b_page_len, b_start_loc):
            output[start : start + sl] = req_to_token_indexs[b][: sl * page_size : page_size] // page_size

    BATCH, MAX_SEQ_LEN = batch, max_seq_len
    max_page_len = (MAX_SEQ_LEN + page_size - 1) // page_size
    total_token_len = 2 * MAX_SEQ_LEN
    total_page_len = (total_token_len + page_size - 1) // page_size

    req_to_token_indexs = torch.empty((2 * BATCH, total_token_len), dtype=torch.int32, device="cuda")
    page_offsets = torch.arange(page_size, dtype=torch.int32, device="cuda")
    for row in range(2 * BATCH):
        page_ids = torch.arange(row * total_page_len, (row + 1) * total_page_len, dtype=torch.int32, device="cuda")
        req_to_token_indexs[row] = (page_ids[:, None] * page_size + page_offsets[None, :]).reshape(-1)[:total_token_len]

    b_req_idx = torch.randperm(BATCH, device="cuda", dtype=torch.int32)
    b_seq_len = torch.randint(1, MAX_SEQ_LEN + 1, (BATCH,), device="cuda", dtype=torch.int32)
    b_page_len = (b_seq_len + page_size - 1) // page_size
    b_start_loc = torch.cat(
        [torch.zeros((1,), dtype=torch.int32, device="cuda"), b_page_len[:-1].cumsum(dim=0, dtype=torch.int32)]
    )

    output = torch.zeros((b_page_len.sum(),), dtype=torch.int32, device="cuda")
    ref = torch.zeros((b_page_len.sum(),), dtype=torch.int32, device="cuda")

    monkeypatch.setenv("PAGE_SIZE", str(page_size))
    get_page_size.cache_clear()
    try:
        repack_page_kv_ref(req_to_token_indexs, b_req_idx, b_page_len, b_start_loc, ref, page_size)
        paged_repack_kv_index(req_to_token_indexs, b_req_idx, b_page_len, b_start_loc, max_page_len, output)
    finally:
        monkeypatch.delenv("PAGE_SIZE", raising=False)
        get_page_size.cache_clear()

    assert torch.equal(output, ref)
