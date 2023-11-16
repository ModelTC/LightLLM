import torch

def quantize_int4_ppl(weight, group_size=128, tp_rank=8):
    """
    weight: [K, N]
    return:
        qweight: [K, N//8] int32 (packed int4*8) new pack_order
        q_scale: [K//group_size, N] int32
    """
    weight = weight.to(dtype=torch.float16).transpose(0, 1).contiguous().cuda(tp_rank)
    from lightllm_ppl_int4_kernel import WeightPreProcess_i4
    qweight_new, q_scale = WeightPreProcess_i4(weight, group_size)
    return qweight_new, q_scale

def matmul_dequantize_int4_ppl(
        x: torch.FloatTensor,
        qweight: torch.IntTensor,
        scale_weight: torch.IntTensor,
        group_size=128,
        workspace = None
) -> torch.FloatTensor:
    """
    x is activation:             (M, K) float16
    qweight is quant weight:     (N//8, K) int32 (int4*8 packed with pack_order)
    return tensor:               (M, N) float16
    """
    if workspace is None:
        workspace = torch.empty(size=[33554432], dtype=torch.int8, device='cuda') # 32MB workspace
        matmul_dequantize_int4_ppl.__defaults__ = (128, workspace)
    from lightllm_ppl_int4_kernel import matmul_i4_fp16
    return matmul_i4_fp16(x, qweight, scale_weight, workspace, group_size)
