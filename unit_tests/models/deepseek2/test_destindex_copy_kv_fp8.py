import torch
import pytest
from lightllm.models.deepseek2.triton_kernel.destindex_copy_kv_fp8 import destindex_copy_kv_fp8
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
def test_destindex_copy_kv_fp8(batch, seqlen, heads, nope_head, rope_head, copy_len):
    B, N_CTX, H, NOPE_HEAD, ROPE_HEAD, COPY_LEN = batch, seqlen, heads, nope_head, rope_head, copy_len
    dtype = torch.bfloat16
    NUM = COPY_LEN
    dest_loc = torch.arange(NUM).cuda()
    kv = torch.randn((len(dest_loc), H, NOPE_HEAD + ROPE_HEAD), dtype=dtype).cuda()
    out = torch.zeros((B * N_CTX, H, NOPE_HEAD + ROPE_HEAD + 2), dtype=torch.uint8).cuda()

    fp8_type = torch.float8_e4m3fn
    kv_nope = kv[:, :, :NOPE_HEAD]
    kv_rope = kv[:, :, NOPE_HEAD:]
    O_nope = out[:, :, :NOPE_HEAD].view(fp8_type)
    O_rope = out[:, :, NOPE_HEAD:-2].view(fp8_type)
    O_scale = out[:, :, -2:].view(dtype)
    destindex_copy_kv_fp8(kv_nope, kv_rope, dest_loc, O_nope, O_rope, O_scale)

    cos1 = F.cosine_similarity(O_nope[:NUM].to(dtype) * O_scale[:NUM], kv_nope).mean()
    cos2 = F.cosine_similarity(O_rope[:NUM].to(dtype) * O_scale[:NUM], kv_rope).mean()
    assert cos1 > 0.98
    assert cos2 > 0.98
