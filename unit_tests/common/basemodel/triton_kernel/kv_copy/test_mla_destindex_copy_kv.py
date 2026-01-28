import torch
import pytest
from lightllm.common.basemodel.triton_kernel.kv_copy.mla_copy_kv import destindex_copy_kv
from lightllm.utils.log_utils import init_logger
import torch.nn.functional as F

logger = init_logger(__name__)

seed = 42
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@pytest.mark.parametrize(
    "batch, seqlen, heads, nope_head, rope_head, copy_len",
    [
        (a, b, c, d, e, f)
        for a in [1, 16, 32, 128, 512]
        for b in [1024, 2048]
        for c in [1]
        for d in [512]
        for e in [64]
        for f in [10, 20, 100, 1024]
    ],
)
def test_destindex_copy_kv(batch, seqlen, heads, nope_head, rope_head, copy_len):
    B, N_CTX, H, NOPE_HEAD, ROPE_HEAD, COPY_LEN = batch, seqlen, heads, nope_head, rope_head, copy_len
    dtype = torch.bfloat16
    dest_loc = torch.randperm(COPY_LEN).cuda()
    kv = torch.randn((len(dest_loc), H, NOPE_HEAD + ROPE_HEAD), dtype=dtype).cuda()
    O_nope = torch.zeros((B * N_CTX, H, NOPE_HEAD), dtype=dtype).cuda()
    O_rope = torch.zeros((B * N_CTX, H, ROPE_HEAD), dtype=dtype).cuda()

    kv_nope = kv[:, :, :NOPE_HEAD]
    kv_rope = kv[:, :, NOPE_HEAD:]
    destindex_copy_kv(kv_nope, kv_rope, dest_loc, O_nope, O_rope)

    assert torch.allclose(O_nope[dest_loc], kv_nope, atol=1e-2, rtol=0)
    assert torch.allclose(O_rope[dest_loc], kv_rope, atol=1e-2, rtol=0)
