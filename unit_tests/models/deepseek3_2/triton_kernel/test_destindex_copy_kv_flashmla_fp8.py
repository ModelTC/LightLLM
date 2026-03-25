import sys
from pathlib import Path

import pytest
import torch

CUR_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str((CUR_DIR / "../../../../lightllm/models/deepseek3_2/triton_kernel").resolve()))

from destindex_copy_kv_flashmla_fp8 import (
    dequantize_kv_reference,
    destindex_copy_kv_flashmla_fp8,
    pack_kv_reference,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_destindex_copy_kv_flashmla_fp8_matches_reference():
    token_num = 257
    kv = torch.randn((token_num, 1, 576), dtype=torch.bfloat16, device="cuda")
    dest_loc = torch.randperm(token_num, device="cuda", dtype=torch.int64)

    out = torch.empty((token_num, 1, 656), dtype=torch.uint8, device="cuda")
    out_nope = out[:, :, :512].view(torch.float8_e4m3fn)
    out_scale = out[:, :, 512:528].view(torch.float32)
    out_rope = out[:, :, 528:].view(torch.bfloat16)

    destindex_copy_kv_flashmla_fp8(
        kv[:, :, :512],
        kv[:, :, 512:],
        dest_loc,
        out_nope,
        out_scale,
        out_rope,
    )
    torch.cuda.synchronize()

    ref = pack_kv_reference(kv)
    assert torch.equal(out[dest_loc], ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_destindex_copy_kv_flashmla_fp8_roundtrip():
    token_num = 257
    kv = torch.randn((token_num, 1, 576), dtype=torch.bfloat16, device="cuda")
    packed = pack_kv_reference(kv)
    dequant = dequantize_kv_reference(packed)

    rope_err = (dequant[:, :, 512:] - kv[:, :, 512:]).abs().max().item()
    nope_err = (dequant[:, :, :512] - kv[:, :, :512]).abs().max().item()

    assert rope_err == 0.0
    assert nope_err < 4e-1
