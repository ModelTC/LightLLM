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
    repack_kv_index(req_to_token_indexs, b_req_idx, b_seq_len, b_start_loc, MAX_SEQ_LEN, output)
    assert torch.allclose(output.float(), ref.float())
