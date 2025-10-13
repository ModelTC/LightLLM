import torch
import time
import pytest
import random
from lightllm.common.fused_moe.moe_silu_and_mul_mix_quant_ep import silu_and_mul_masked_post_quant_fwd
from lightllm.common.fused_moe.moe_silu_and_mul import silu_and_mul_fwd
from lightllm.common.quantization.triton_quant.fp8.fp8act_quant_kernel import per_token_group_quant_fp8
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
    in_tensor = torch.randn((expert_num, token_num, hidden_dim), dtype=torch.float16, device="cuda")
    out_tensor = torch.empty((expert_num, token_num, hidden_dim // 2), dtype=torch.float8_e4m3fn, device="cuda")
    out_scale_tensor = torch.randn(
        (expert_num, token_num, hidden_dim // 2 // quant_group_size), dtype=torch.float32, device="cuda"
    )

    true_out_tensor_mid = torch.randn((expert_num, token_num, hidden_dim // 2), dtype=torch.float16, device="cuda")

    masked_m = [random.randint(0, token_num) for _ in range(expert_num)]
    masked_m = torch.tensor(masked_m, dtype=torch.int32, device="cuda")

    silu_and_mul_fwd(in_tensor.view(-1, hidden_dim), true_out_tensor_mid.view(-1, hidden_dim // 2))
    true_out_tensor, true_out_scale_tensor = per_token_group_quant_fp8(
        true_out_tensor_mid.view(-1, hidden_dim // 2),
        quant_group_size,
        alloc_func=torch.empty,
    )

    silu_and_mul_masked_post_quant_fwd(in_tensor, out_tensor, out_scale_tensor, quant_group_size, masked_m)

    true_out_tensor = true_out_tensor.view(out_tensor.shape)
    true_out_scale_tensor = true_out_scale_tensor.view(out_scale_tensor.shape)

    for expert_id, expert_token_num in enumerate(masked_m.cpu().numpy()):
        assert torch.allclose(
            true_out_tensor[expert_id, :expert_token_num, :].to(torch.float32),
            out_tensor[expert_id, :expert_token_num, :].to(torch.float32),
            atol=1e-3,
            rtol=1e-2,
        )
        hidden_dim_scale_count = hidden_dim // 2 // quant_group_size
        assert torch.allclose(
            true_out_scale_tensor[expert_id, :expert_token_num, :hidden_dim_scale_count],
            out_scale_tensor[expert_id, :expert_token_num, :hidden_dim_scale_count],
            atol=1e-3,
            rtol=1e-2,
        )
    return


if __name__ == "__main__":
    pytest.main()
