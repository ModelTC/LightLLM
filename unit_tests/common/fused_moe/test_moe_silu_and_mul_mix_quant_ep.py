import torch
import pytest


def is_fp8_native_supported():
    """检查是否为 H100/B200 等原生支持 FP8 的硬件 (SM90+)"""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


if not is_fp8_native_supported():
    pytest.skip(reason="not support fp8 test in this gpu card", allow_module_level=True)

from lightllm.common.basemodel.triton_kernel.fused_moe.moe_silu_and_mul_mix_quant_ep import (
    silu_and_mul_psum_post_quant_fwd,
)
from lightllm.common.basemodel.triton_kernel.fused_moe.moe_silu_and_mul import silu_and_mul_fwd
from lightllm.common.basemodel.triton_kernel.quantization.fp8act_quant_kernel import per_token_group_quant_fp8
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def _build_psum_layout(token_counts, expert_alignment):
    expert_ends = []
    valid_rows = []
    previous_end = 0
    for token_count in token_counts:
        expert_start = (previous_end + expert_alignment - 1) // expert_alignment * expert_alignment
        expert_end = expert_start + token_count
        expert_ends.append(expert_end)
        valid_rows.extend(range(expert_start, expert_end))
        previous_end = expert_end

    num_rows = (expert_ends[-1] + expert_alignment - 1) // expert_alignment * expert_alignment
    return (
        torch.tensor(expert_ends, dtype=torch.int32, device="cuda"),
        torch.tensor(valid_rows, dtype=torch.int64, device="cuda"),
        num_rows,
    )


@pytest.mark.parametrize(
    "expert_num, token_num, hidden_dim",
    [
        (
            expert_num,
            token_num,
            hidden_dim,
        )
        for expert_num in range(3, 6)
        for hidden_dim in [256, 128 * 4, 2048]
        for token_num in range(1, 7, 2)
    ],
)
def test_silu_and_mul_psum(expert_num, token_num, hidden_dim):
    quant_group_size = 128
    expert_alignment = 128
    expert_token_psum, valid_rows, num_rows = _build_psum_layout(
        [token_num] * expert_num,
        expert_alignment,
    )
    in_tensor = torch.randn((num_rows, hidden_dim), dtype=torch.bfloat16, device="cuda")
    out_tensor = torch.full(
        (num_rows, hidden_dim // 2),
        1.0,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    out_scale_tensor = torch.full(
        (num_rows, hidden_dim // 2 // quant_group_size),
        7.0,
        dtype=torch.float32,
        device="cuda",
    )

    true_out_tensor_mid = torch.empty((valid_rows.numel(), hidden_dim // 2), dtype=in_tensor.dtype, device="cuda")
    silu_and_mul_fwd(in_tensor[valid_rows], true_out_tensor_mid)
    true_out_tensor, true_out_scale_tensor = per_token_group_quant_fp8(
        true_out_tensor_mid,
        quant_group_size,
        alloc_func=torch.empty,
    )

    silu_and_mul_psum_post_quant_fwd(
        in_tensor,
        out_tensor,
        out_scale_tensor,
        expert_token_psum,
        expert_alignment,
        quant_group_size,
    )

    assert torch.allclose(true_out_scale_tensor, out_scale_tensor[valid_rows], atol=1e-3, rtol=1e-2)
    true_dequant = true_out_tensor.to(torch.float32) * true_out_scale_tensor.repeat_interleave(quant_group_size, dim=-1)
    out_dequant = out_tensor[valid_rows].to(torch.float32) * out_scale_tensor[valid_rows].repeat_interleave(
        quant_group_size, dim=-1
    )
    assert torch.allclose(true_dequant, out_dequant, atol=1e-1, rtol=1e-1)

    padding_rows = torch.ones(num_rows, dtype=torch.bool, device="cuda")
    padding_rows[valid_rows] = False
    assert torch.equal(out_tensor[padding_rows], torch.ones_like(out_tensor[padding_rows]))
    assert torch.equal(out_scale_tensor[padding_rows], torch.full_like(out_scale_tensor[padding_rows], 7.0))
    return


def test_silu_and_mul_psum_skips_padding_and_empty_expert():
    token_num = 4
    hidden_dim = 256
    quant_group_size = 128
    expert_alignment = 128
    expert_token_psum, valid_rows, num_rows = _build_psum_layout(
        [0, 2, token_num],
        expert_alignment,
    )

    in_tensor = torch.randn((num_rows, hidden_dim), dtype=torch.bfloat16, device="cuda")
    out_tensor = torch.empty((num_rows, hidden_dim // 2), dtype=torch.float8_e4m3fn, device="cuda")
    out_scale_tensor = torch.empty((num_rows, hidden_dim // 2 // quant_group_size), dtype=torch.float32, device="cuda")
    out_tensor.fill_(1.0)
    out_scale_tensor.fill_(7.0)

    silu_and_mul_psum_post_quant_fwd(
        in_tensor,
        out_tensor,
        out_scale_tensor,
        expert_token_psum,
        expert_alignment,
        quant_group_size,
    )
    torch.cuda.synchronize()

    padding_rows = torch.ones(num_rows, dtype=torch.bool, device="cuda")
    padding_rows[valid_rows] = False
    assert torch.equal(
        out_tensor[padding_rows],
        torch.ones_like(out_tensor[padding_rows]),
    )
    assert torch.equal(
        out_scale_tensor[padding_rows],
        torch.full_like(out_scale_tensor[padding_rows], 7.0),
    )


if __name__ == "__main__":
    pytest.main()
