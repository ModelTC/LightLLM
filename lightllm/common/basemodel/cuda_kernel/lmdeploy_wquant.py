import torch

@torch.no_grad()
def quantize_int4_lmdeploy(weight, group_size=128, tp_rank=0, pack_order=[0, 2, 4, 6, 1, 3, 5, 7]):
    """
    weight: [K, N]
    return:
        qweight: [K, N//8] int32 (packed int4*8) new pack_order
        scale_zeros: [K//group_size, N] int32
        # qzeros: [K//group_size, N//8] int32 (packed int4*8) new pack_order
    """
    K, N = weight.shape
    weight = weight.transpose(1, 0)
    print("tp_rank: {} quantize_int4_lmdeploy for K={} N={} ...".format(tp_rank, K, N))
    assert K % 8 == 0 and N % 8 == 0, "K={} N={}".format(K, N)
    assert K % group_size == 0, "K={} N={}".format(K, N)

    weight = weight.contiguous().view(-1, group_size).cuda(tp_rank)
    weight_max = weight.amax(-1, keepdim=True)
    weight_max = torch.where(weight_max < 0, 0, weight_max)
    weight_min = weight.amin(-1, keepdim=True)
    weight_min = torch.where(weight_min > 0, 0, weight_min)
    weight_range = weight_max - weight_min 
    
    scale = (weight_range / (2 ** 4 - 1))
    zero_point = (-weight_min / scale).round().clamp(0, 15).to(torch.int32)
    # (N, K)
    weight = (weight / scale + zero_point).round().clamp(0, 15).to(torch.int32).view(N, K)
    # (N, K//group_size)
    scale = scale.view(N, -1)
    # (N, K//group_size)
    zero_point = zero_point.view(N, -1)

    # pack 8 int4 in an int32 number at axis-N
    qweight = torch.zeros((N // 8, K), dtype=torch.int32, device=weight.device)
    qzeros  = torch.zeros((N // 8, K // group_size), dtype=torch.int32, device=weight.device)

    for pack in range(0, N, 8):
        for i in range(8):
            qweight[pack // 8, :] += weight[pack + pack_order[i], :] << (i * 4)
            qzeros[pack // 8, :] += zero_point[pack + pack_order[i], :] << (i * 4)

    weight = None
    qweight = qweight.transpose(1, 0).contiguous()
    scale = scale.transpose(1, 0).contiguous()
    qzeros = qzeros.transpose(1, 0).contiguous()

    # convert to layout defined inside lmdeploy
    qweight_new = torch.zeros_like(qweight)
    scale_zeros = torch.zeros_like(scale, dtype=torch.int32)  # half2
    temp = torch.zeros_like(scale)
    from lightllm_lmdeploy_kernel import convert_s4_k_m8
    convert_s4_k_m8(
        qweight_new,
        scale_zeros,
        temp,
        qweight,
        scale,
        qzeros,
        N,
        K,
        group_size
    )
    temp = None
    scale = None
    return qweight_new, scale_zeros, group_size


def matmul_dequantize_int4_lmdeploy(
        x: torch.FloatTensor,
        qweight: torch.IntTensor,
        scale_zeros: torch.IntTensor,
        group_size,
        output = None,
        has_silu = False,
) -> torch.FloatTensor:
    """
    x is activation:             (M, K) float16
    qweight is quant weight:     (K, N//8) int32 (int4*8 packed with pack_order)
    scale_zeros is quant merged(scales, qzeros):      (K//group_size, N) int32
    return tensor:               (M, N) float16
    """
    assert x.shape[1] == qweight.shape[0], "A must be a multiple of 8 in the last dimension"
    M, K = x.shape
    N = qweight.shape[1] * 8
    if output is None:
        output = torch.empty((M, N), dtype=x.dtype, device="cuda")
    from lightllm_lmdeploy_kernel import int4fp16_matmul
    int4fp16_matmul(output, qweight, x, scale_zeros, N, M, K, group_size, has_silu)
    return output