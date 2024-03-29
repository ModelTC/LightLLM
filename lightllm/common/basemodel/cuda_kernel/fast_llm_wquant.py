import torch

@torch.no_grad()
def fp6_quant(weight: torch.Tensor, tp_rank=0):
    weight = weight.to(dtype=torch.float32).transpose(0, 1).contiguous().cpu()
    M = weight.shape[0]
    N = weight.shape[1]
    fp6_weight = torch.zeros((M, (N * 6) // 32), dtype=torch.int32, device="cpu")

    weight_max = torch.abs(weight).amax(-1, keepdim=True)
    scale = weight_max / 28
    quant_half = (weight / (scale.view(M, 1) * (2 ** 12))).half().contiguous()
    from flash_llm_fp6_llm import weight_quant_to_fp6
    fp6_weight = weight_quant_to_fp6(quant_half, fp6_weight, True)

    return fp6_weight.cuda(tp_rank), scale.half().contiguous().cuda(tp_rank)


def matmul_dequantize_int6_fast_llm(x: torch.FloatTensor, qweight: torch.IntTensor, scale_weight: torch.IntTensor):
    from flash_llm_fp6_llm import linear_forward_cuda
    return linear_forward_cuda(x, qweight, scale_weight, 1)