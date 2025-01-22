import triton
import torch
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

from lightllm.utils.tuning_utils import mp_tuning, set_seed, tuning_configs
import sys
import os

from lightllm.common.basemodel.triton_kernel.bmm_scaled_fp8 import bmm_scaled_fp8, BmmScaledFp8KernelConfig


@torch.no_grad()
def test_func(
    B,
    M,
    N,
    K,
    dtype,
    test_count: int = 20,
    **run_config,
):
    set_seed()

    a_scale = torch.randn([B, M, 1], device="cuda", dtype=dtype)
    b_scale = torch.randn([B, N, 1], device="cuda", dtype=dtype)
    a = torch.randn([B, M, K], device="cuda", dtype=dtype)
    b = torch.randn([B, K, N], device="cuda", dtype=dtype)
    c = torch.zeros([B, M, N], device="cuda", dtype=dtype)
    a = a.to(torch.float8_e4m3fn)
    b = b.to(torch.float8_e4m3fn).transpose(1, 2).contiguous().transpose(1, 2)
    fn = lambda: bmm_scaled_fp8(a, a_scale, b, b_scale, c, **run_config)
    cost_time = triton.testing.do_bench_cudagraph(fn, rep=test_count)

    logger.info(f"bf16 {B, M, N, K} cost time: {cost_time} ms")
    return cost_time


def get_test_configs(split_id, split_count, **kwargs):
    index = 0
    for block_size_m in [64, 128, 256]:
        for block_size_n in [64, 128, 256]:
            for block_size_k in [64, 128]:
                for group_size_m in [4, 8, 16]:
                    for num_warps in [4, 8]:
                        for num_stages in [2, 3, 4]:
                            t_config = {
                                "BLOCK_SIZE_M": block_size_m,
                                "BLOCK_SIZE_N": block_size_n,
                                "BLOCK_SIZE_K": block_size_k,
                                "GROUP_SIZE_M": group_size_m,
                                "num_stages": num_stages,
                                "num_warps": num_warps,
                            }
                            if index % split_count == split_id:
                                yield t_config
                            index += 1


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    import collections

    store_json_ans = collections.defaultdict(dict)
    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        for head_dim in [128, 512]:
            k = 128 if head_dim == 512 else 512
            test_func_args = {
                "B": 16,
                "M": batch_size,
                "N": head_dim,
                "K": k,
                "dtype": torch.bfloat16,
                "test_count": 20,
            }
            ans = mp_tuning(
                tuning_configs,
                {
                    "test_func": test_func,
                    "test_func_args": test_func_args,
                    "get_test_configs_func": get_test_configs,
                },
            )
            store_json_ans[batch_size][head_dim] = ans
            BmmScaledFp8KernelConfig.save_config(
                B=16,
                M=batch_size,
                N=head_dim,
                K=k,
                config_json=store_json_ans,
            )

    pass
