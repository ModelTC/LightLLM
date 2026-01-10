import pytest
import torch
from lightllm.common.basemodel.triton_kernel.att.decode_att.int8kv.ppl_int8kv_flash_decoding_diverse_stage3 import (
    flash_diverse_decode_stage3,
)


@pytest.mark.parametrize(
    "batch, head_num, seq_len, shared_seq_len, block_seq, head_dim",
    [
        (2, 4, 256, 256, 256, 128),
        (1, 8, 256 * 2, 256, 256, 128),
        (3, 2, 256 * 4, 256 * 2, 256, 128),
    ],
)
def test_flash_diverse_decode_stage3(batch, head_num, seq_len, shared_seq_len, block_seq, head_dim):
    # Initialize inputs
    mid_out = torch.randn(batch, head_num, seq_len // block_seq + 2, head_dim, dtype=torch.bfloat16, device="cuda")
    mid_out_logexpsum = torch.randn(batch, head_num, seq_len // block_seq + 2, dtype=torch.bfloat16, device="cuda")
    B_Seqlen = torch.tensor([seq_len] * batch, dtype=torch.int32, device="cuda")
    b_shared_seq_len = torch.tensor([shared_seq_len] * batch, dtype=torch.int32, device="cuda")
    out = torch.zeros(batch, head_num, head_dim, dtype=torch.float32, device="cuda")

    # Call the function
    flash_diverse_decode_stage3(mid_out, mid_out_logexpsum, B_Seqlen, b_shared_seq_len, out, block_seq)

    true_out = torch.zeros_like(out)

    from lightllm.common.basemodel.triton_kernel.att.decode_att.mha.flash_decoding.flash_decoding_stage2 import (
        flash_decode_stage2,
    )

    flash_decode_stage2(mid_out, mid_out_logexpsum, B_Seqlen, true_out, block_seq)

    assert torch.allclose(out, true_out, atol=1e-2)
