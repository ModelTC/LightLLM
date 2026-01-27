import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from lightllm.common.kernel_config import KernelConfigs
from frozendict import frozendict
from functools import lru_cache
from typing import Dict


class BmmScaledFp8KernelConfig(KernelConfigs):
    kernel_name: str = "bmm_scaled_fp8"

    def closest_power_2(n: int) -> int:
        return 1 << (n - 1).bit_length() if n & (n - 1) else n

    @classmethod
    @lru_cache(maxsize=200)
    def try_to_get_best_config(
        cls,
        B,
        M,
        N,
        K,
        batch_size,
        head_dim,
    ) -> dict:
        key_params = {
            "B": B,
            "M": M,
            "N": N,
            "K": K,
            "out_dtype": str(torch.bfloat16),
        }
        finded_config = cls.get_the_config(frozendict(key_params))

        search_keys = [batch_size, head_dim]
        if finded_config:
            config = finded_config
            for key in search_keys:
                config = config[min(config.keys(), key=lambda x: abs(int(x) - key))]
        else:
            config = {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
                "num_stages": 4,
                "num_warps": 8,
            }
        return config

    @classmethod
    def save_config(cls, B, M, N, K, config_json: Dict[int, Dict[int, Dict]]):
        key_params = {
            "B": B,
            "M": M,
            "N": N,
            "K": K,
            "out_dtype": str(torch.bfloat16),
        }
        key_params = frozendict(key_params)
        return cls.store_config(key_params, config_json)


@triton.jit
def bmm_scaled_fp8_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    M,
    N,
    K,
    stride_ah,
    stride_am,
    stride_ak,
    stride_bh,
    stride_bk,
    stride_bn,
    stride_ch,
    stride_cm,
    stride_cn,
    stride_scale_ah,
    stride_scale_am,
    stride_scale_bh,
    stride_scale_bn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    stride_ah = stride_ah.to(tl.int64)
    stride_bh = stride_bh.to(tl.int64)
    stride_ch = stride_ch.to(tl.int64)
    pid = tl.program_id(axis=0)
    head_id = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_am = tl.where(offs_am < M, offs_am, 0)
    offs_bn = tl.where(offs_bn < N, offs_bn, 0)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + head_id * stride_ah + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + head_id * stride_bh + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    a_scale_ptrs = a_scale_ptr + head_id * stride_scale_ah + offs_am[:, None] * stride_scale_am
    b_scale_ptrs = b_scale_ptr + head_id * stride_scale_bh + offs_bn[None, :] * stride_scale_bn
    a_scale = tl.load(a_scale_ptrs)
    b_scale = tl.load(b_scale_ptrs)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator * a_scale * b_scale

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + head_id * stride_ch + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def bmm_scaled_fp8(a, a_scale, b, b_scale, c, **run_config):
    assert a.shape[0] == b.shape[0], "Incompatible dimensions"
    assert c.shape[0] == b.shape[0], "Incompatible dimensions"
    assert a.shape[2] == b.shape[1], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    M, K = a.shape[1], a.shape[2]
    K, N = b.shape[1], b.shape[2]
    HEAD = a.shape[0]

    if not run_config:
        M2 = BmmScaledFp8KernelConfig.closest_power_2(M)
        run_config = BmmScaledFp8KernelConfig.try_to_get_best_config(
            B=HEAD,
            M=M2,
            N=N,
            K=K,
            batch_size=M2,
            head_dim=N,
        )

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        HEAD,
    )
    assert b.stride(1) == 1
    bmm_scaled_fp8_kernel[grid](
        a,
        b,
        c,
        a_scale,
        b_scale,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        c.stride(0),
        c.stride(1),
        c.stride(2),
        a_scale.stride(0),
        a_scale.stride(1),
        b_scale.stride(0),
        b_scale.stride(1),
        BLOCK_SIZE_M=run_config["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=run_config["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=run_config["BLOCK_SIZE_K"],
        GROUP_SIZE_M=run_config["GROUP_SIZE_M"],
        num_stages=run_config["num_stages"],
        num_warps=run_config["num_warps"],
    )
    return c


if __name__ == "__main__":
    B, M, N, K = 16, 100, 512, 128
    dtype = torch.bfloat16
    a_scale = torch.randn([B, M, 1], device="cuda", dtype=dtype)
    b_scale = torch.randn([B, N, 1], device="cuda", dtype=dtype)
    a = torch.randn([B, M, K], device="cuda", dtype=dtype)
    b = torch.randn([B, K, N], device="cuda", dtype=dtype)
    c = torch.zeros([B, M, N], device="cuda", dtype=dtype)
    a = a.to(torch.float8_e4m3fn)
    b = b.to(torch.float8_e4m3fn).transpose(1, 2).contiguous().transpose(1, 2)
    aa = a.to(dtype) * a_scale
    bb = b.to(dtype) * b_scale.transpose(1, 2)
    o = torch.bmm(aa, bb)
    bmm_scaled_fp8(a, a_scale, b, b_scale, c)
    cos = F.cosine_similarity(c, o).mean()
    assert cos == 1.0

    fn1 = lambda: torch.bmm(aa, bb)
    fn2 = lambda: bmm_scaled_fp8(a, a_scale, b, b_scale, c)
    ms1 = triton.testing.do_bench_cudagraph(fn1)
    ms2 = triton.testing.do_bench_cudagraph(fn2)
    print(ms1, ms2)
