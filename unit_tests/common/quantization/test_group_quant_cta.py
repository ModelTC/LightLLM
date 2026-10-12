import pytest
import torch

from lightllm.common.basemodel.triton_kernel.quantization import fp8act_quant_kernel as quant


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "rows,width", [(1, 4096), (63, 4096), (64, 4096), (129, 4096), (256, 4096), (8192, 4096), (171, 1536), (173, 3840)]
)
def test_grouped_cta_preserves_single_group_quantization(rows, width):
    torch.manual_seed(1012)
    x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
    x[0, :128] = 0
    x[0, 128:256] = 1e-7
    q = torch.full((rows + 1, width), 1, device="cuda", dtype=torch.float8_e4m3fn)
    scale_storage = torch.full((width // 128, rows + 4), -1.0, device="cuda")
    scales = scale_storage[:, :rows].T
    quant.lightllm_per_token_group_quant_fp8(x, 128, q[:rows], scales, use_ue8m0_scales=True)
    reference_q = torch.empty_like(q[:rows])
    reference_scales = torch.empty_like(scales)
    groups = x.numel() // 128
    quant._per_token_group_quant_fp8[(groups,)](
        x,
        reference_q,
        reference_scales,
        128,
        groups,
        128,
        1e-10,
        fp8_min=-448.0,
        fp8_max=448.0,
        xs_n=width // 128,
        xs_stride_m=reference_scales.stride(0),
        xs_stride_n=reference_scales.stride(1),
        topk_ids_ptr=x,
        topk_weights_ptr=x,
        topk_ids_out_ptr=x,
        topk_weights_out_ptr=x,
        num_topk=0,
        topk_row_stride=0,
        topk_out_row_stride=0,
        BLOCK=128,
        TOPK_BLOCK=1,
        NEED_MASK=False,
        USE_UE8M0_SCALE=True,
        COPY_TOPK=False,
        GROUPS_PER_CTA=1,
        num_warps=1,
        num_stages=1,
    )
    torch.testing.assert_close(q[:rows].view(torch.uint8), reference_q.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(scales, reference_scales, rtol=0, atol=0)
    assert torch.all(q[-1].float() == 1)
    assert torch.all(scale_storage[:, rows:] == -1)
