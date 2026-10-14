import torch
import pytest
from lightllm.utils.log_utils import init_logger
from lightllm.common.basemodel.triton_kernel.repack_kv_index import repack_kv_index

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
    repack_kv_index(
        req_to_token_indexs=req_to_token_indexs,
        b_req_idx=b_req_idx,
        b_token_len=b_seq_len,
        b_page_start_loc=b_start_loc,
        max_token_len=MAX_SEQ_LEN,
        out_page_indices=output,
    )
    assert torch.allclose(output.float(), ref.float())


@pytest.mark.parametrize("page_size, page_count", [(1, 3), (3, 3), (4, 3), (16, 3), (4, 65), (16, 65)])
def test_repack_kv_index_with_pages(page_size, page_count):
    req_to_token_indexs = torch.tensor(
        [
            list(range(10 * page_size, (10 + page_count) * page_size)),
            list(range(20 * page_size, (20 + page_count) * page_size)),
            list(range(30 * page_size, (30 + page_count) * page_size)),
        ],
        dtype=torch.int32,
        device="cuda",
    )
    req_indexes = torch.tensor([2, 0, 1], dtype=torch.int32, device="cuda")
    # 使用不满的末页，并覆盖超过单个 block（64 页）的请求。
    max_seq_len = page_count * page_size - (page_size > 1)
    seq_lens = torch.tensor([2 * page_size, 1, max_seq_len], dtype=torch.int32, device="cuda")
    starts = torch.tensor([0, 2, 3], dtype=torch.int32, device="cuda")
    output = torch.empty((3 + page_count,), dtype=torch.int32, device="cuda")

    repack_kv_index(
        req_to_token_indexs=req_to_token_indexs,
        b_req_idx=req_indexes,
        b_token_len=seq_lens,
        b_page_start_loc=starts,
        max_token_len=max_seq_len,
        out_page_indices=output,
        page_size=page_size,
    )

    assert output.cpu().tolist() == [30, 31, 10] + list(range(20, 20 + page_count))
