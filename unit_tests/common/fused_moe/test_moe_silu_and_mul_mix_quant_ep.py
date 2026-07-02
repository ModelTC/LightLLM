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
    silu_and_mul_masked_post_quant_fwd,
)
from lightllm.common.basemodel.triton_kernel.fused_moe.moe_silu_and_mul import silu_and_mul_fwd
from lightllm.common.basemodel.triton_kernel.quantization.fp8act_quant_kernel import per_token_group_quant_fp8
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


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
def test_silu_and_mul_masked(expert_num, token_num, hidden_dim):
    quant_group_size = 128
    in_tensor = torch.randn((expert_num, token_num, hidden_dim), dtype=torch.bfloat16, device="cuda")
    out_tensor = torch.empty((expert_num, token_num, hidden_dim // 2), dtype=torch.float8_e4m3fn, device="cuda")
    out_scale_tensor = torch.randn(
        (expert_num, token_num, hidden_dim // 2 // quant_group_size), dtype=torch.float32, device="cuda"
    )

    true_out_tensor_mid = torch.empty((expert_num, token_num, hidden_dim // 2), dtype=in_tensor.dtype, device="cuda")

    masked_m = torch.full((expert_num,), token_num, dtype=torch.int32, device="cuda")

    silu_and_mul_fwd(in_tensor.view(-1, hidden_dim), true_out_tensor_mid.view(-1, hidden_dim // 2))
    true_out_tensor, true_out_scale_tensor = per_token_group_quant_fp8(
        true_out_tensor_mid.view(-1, hidden_dim // 2),
        quant_group_size,
        alloc_func=torch.empty,
    )

    silu_and_mul_masked_post_quant_fwd(in_tensor, out_tensor, out_scale_tensor, quant_group_size, masked_m)

    true_out_tensor = true_out_tensor.view(out_tensor.shape)
    true_out_scale_tensor = true_out_scale_tensor.view(out_scale_tensor.shape)

    hidden_dim_scale_count = hidden_dim // 2 // quant_group_size
    for expert_id, expert_token_num in enumerate(masked_m.cpu().numpy()):
        true_scale = true_out_scale_tensor[expert_id, :expert_token_num, :hidden_dim_scale_count]
        out_scale = out_scale_tensor[expert_id, :expert_token_num, :hidden_dim_scale_count]
        assert torch.allclose(
            true_scale,
            out_scale,
            atol=1e-3,
            rtol=1e-2,
        )

        true_out = true_out_tensor[expert_id, :expert_token_num, :].to(torch.float32)
        out = out_tensor[expert_id, :expert_token_num, :].to(torch.float32)
        true_dequant = true_out * true_scale.repeat_interleave(quant_group_size, dim=-1)
        out_dequant = out * out_scale.repeat_interleave(quant_group_size, dim=-1)
        assert torch.allclose(true_dequant, out_dequant, atol=1e-1, rtol=1e-1)
    return


def test_silu_and_mul_masked_skips_padded_tokens():
    expert_num = 3
    token_num = 4
    hidden_dim = 256
    quant_group_size = 128
    masked_m = torch.tensor([0, 2, token_num], dtype=torch.int32, device="cuda")

    in_tensor = torch.randn((expert_num, token_num, hidden_dim), dtype=torch.bfloat16, device="cuda")
    out_tensor = torch.empty((expert_num, token_num, hidden_dim // 2), dtype=torch.float8_e4m3fn, device="cuda")
    out_scale_tensor = torch.empty(
        (expert_num, token_num, hidden_dim // 2 // quant_group_size), dtype=torch.float32, device="cuda"
    )
    out_tensor.fill_(1.0)
    out_scale_tensor.fill_(7.0)

    silu_and_mul_masked_post_quant_fwd(in_tensor, out_tensor, out_scale_tensor, quant_group_size, masked_m)
    torch.cuda.synchronize()

    for expert_id, expert_token_num in enumerate(masked_m.cpu().tolist()):
        assert torch.equal(
            out_tensor[expert_id, expert_token_num:, :],
            torch.ones_like(out_tensor[expert_id, expert_token_num:, :]),
        )
        assert torch.equal(
            out_scale_tensor[expert_id, expert_token_num:, :],
            torch.full_like(out_scale_tensor[expert_id, expert_token_num:, :], 7.0),
        )


if __name__ == "__main__":
    pytest.main()
