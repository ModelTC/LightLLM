import os
import torch
import time
import torch.multiprocessing as mp
from lightllm.common.fused_moe.grouped_fused_moe import fused_experts_impl, moe_align, moe_align1, grouped_matmul
from typing import List
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def set_seed():
    import torch
    import random
    import numpy as np

    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return


def quantize_moe(weight):
    try:
        HAS_VLLM = True
        from lightllm.common.vllm_kernel import _custom_ops as ops
    except:
        HAS_VLLM = False

    assert HAS_VLLM

    num_experts = weight.shape[0]
    qweights = []
    weight_scales = []
    qweights = torch.empty_like(weight, dtype=torch.float8_e4m3fn).cuda()
    for i in range(num_experts):
        qweight, weight_scale = ops.scaled_fp8_quant(
            weight[i].contiguous().cuda(), scale=None, use_per_token_if_dynamic=False
        )
        qweights[i] = qweight
        weight_scales.append(weight_scale)
    weight_scale = torch.cat(weight_scales, dim=0).reshape(-1)
    return qweights, weight_scale


@torch.no_grad()
def test_kernel(
    expert_num: int,
    m: int,
    n: int,
    k: int,
    topk: int,
    dtype: torch.dtype,
    test_count: int,
    use_fp8_w8a8: bool,
    is_up: bool,
    **config,
):
    set_seed()
    input_tuples = []

    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((expert_num, 2 * n, k), device="cuda", dtype=dtype) / 10
    w2 = torch.randn((expert_num, k, n), device="cuda", dtype=dtype) / 10
    rnd_logics = torch.randn(m, expert_num, device="cuda")
    topk_values, topk_ids = torch.topk(rnd_logics, topk, dim=1)
    topk_weights = torch.randn((m, topk), device="cuda", dtype=dtype) / 10

    expert_to_tokens = torch.empty((expert_num, topk * m), dtype=torch.int32, device="cuda")
    expert_to_weights = torch.empty((expert_num, topk * m), dtype=torch.float32, device="cuda")
    moe_align(topk_ids=topk_ids, out=expert_to_tokens)
    expert_to_token_num = torch.empty((expert_num,), dtype=torch.int32, device="cuda")
    moe_align1(expert_to_tokens, topk_weights, expert_to_weights, expert_to_token_num, topk=topk)
    if use_fp8_w8a8:
        w1, w1_scale = quantize_moe(w1)
        w2, w2_scale = quantize_moe(w2)
    else:
        w1_scale = torch.empty((0,))
        w2_scale = torch.empty((0,))

    out1 = torch.zeros((m * topk, 2 * n), dtype=torch.bfloat16, device="cuda")
    down_in = torch.zeros((m * topk, n), dtype=torch.bfloat16, device="cuda")
    out2 = torch.zeros((m * topk, k), dtype=torch.bfloat16, device="cuda")

    for _ in range(test_count):
        input_tuples.append(
            (
                a.clone(),
                w1.clone(),
                w2.clone(),
                w1_scale.clone(),
                w2_scale.clone(),
                topk_ids.clone(),
                topk_weights.clone(),
                out1.clone(),
                out2.clone(),
                down_in.clone(),
            )
        )

    if is_up:
        grouped_matmul(
            a,
            None,
            expert_to_token_num,
            expert_to_tokens,
            expert_to_weights=expert_to_weights,
            expert_weights=w1,
            expert_to_weights_scale=w1_scale,
            topk_num=topk,
            out=out1,
            expert_token_limit=2 ** 31 - 1,
            mul_routed_weight=False,
            use_fp8_w8a8=use_fp8_w8a8,
            **config,
        )
    else:
        grouped_matmul(
            down_in,
            None,
            expert_to_token_num,
            expert_to_tokens,
            expert_to_weights=expert_to_weights,
            expert_weights=w2,
            expert_to_weights_scale=w2_scale,
            topk_num=1,
            out=out2,
            expert_token_limit=2 ** 31 - 1,
            mul_routed_weight=True,
            use_fp8_w8a8=use_fp8_w8a8,
            **config,
        )

    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph):
        for index in range(test_count):
            a, w1, w2, w1_scale, w2_scale, topk_ids, topk_weights, out1, out2, down_in = input_tuples[index]
            if is_up:
                grouped_matmul(
                    a,
                    None,
                    expert_to_token_num,
                    expert_to_tokens,
                    expert_to_weights=expert_to_weights,
                    expert_weights=w1,
                    expert_to_weights_scale=w1_scale,
                    topk_num=topk,
                    out=out1,
                    expert_token_limit=2 ** 31 - 1,
                    mul_routed_weight=False,
                    use_fp8_w8a8=use_fp8_w8a8,
                    **config,
                )
            else:
                grouped_matmul(
                    down_in,
                    None,
                    expert_to_token_num,
                    expert_to_tokens,
                    expert_to_weights=expert_to_weights,
                    expert_weights=w2,
                    expert_to_weights_scale=w2_scale,
                    topk_num=1,
                    out=out2,
                    expert_token_limit=2 ** 31 - 1,
                    mul_routed_weight=True,
                    use_fp8_w8a8=use_fp8_w8a8,
                    **config,
                )

    graph.replay()

    torch.cuda.synchronize()
    start = time.time()
    graph.replay()
    torch.cuda.synchronize()

    cost_time = (time.time() - start) * 1000

    logger.info(str(config))
    logger.info(f"bf16 {m} cost time: {cost_time} ms")
    return cost_time


def worker(
    expert_num: int,
    m: int,
    n: int,
    k: int,
    topk: int,
    dtype: torch.dtype,
    test_count: int,
    use_fp8_w8a8: bool,
    is_up: bool,
    test_configs,
    queue,
):
    try:
        for index in range(len(test_configs)):
            cost_time = test_kernel(
                expert_num=expert_num,
                m=m,
                n=n,
                k=k,
                topk=topk,
                dtype=dtype,
                test_count=test_count,
                use_fp8_w8a8=use_fp8_w8a8,
                is_up=is_up,
                **test_configs[index],
            )
            queue.put(cost_time)  # Put result in queue

    except Exception as ex:
        logger.error(str(ex))
        logger.exception(str(ex))
        import sys

        sys.exit(-1)
        pass


def get_test_configs(split_id, split_count):
    index = 0
    for num_stages in range(1, 6):
        for GROUP_SIZE_M in [
            1,
            2,
            4,
            8,
        ]:
            for num_warps in [
                2,
                4,
                8,
                16,
            ]:
                for BLOCK_SIZE_M in [
                    16,
                    32,
                    64,
                    128,
                ]:
                    for BLOCK_SIZE_N in [16, 32, 64, 128]:
                        for BLOCK_SIZE_K in [16, 32, 64, 128]:
                            t_config = {
                                "BLOCK_SIZE_M": BLOCK_SIZE_M,
                                "BLOCK_SIZE_N": BLOCK_SIZE_N,
                                "BLOCK_SIZE_K": BLOCK_SIZE_K,
                                "GROUP_SIZE_M": GROUP_SIZE_M,
                                "num_warps": num_warps,
                                "num_stages": num_stages,
                            }
                            if index % split_count == split_id:
                                yield t_config
                                index += 1
                            else:
                                index += 1


def tuning_configs(
    device_id: int,  # use for mult mp tunning
    device_count: int,
    expert_num: int,
    m: int,
    n: int,
    k: int,
    topk: int,
    dtype: torch.dtype,
    test_count: int,
    use_fp8_w8a8: bool,
    is_up: bool,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    best_config, best_cost_time = None, 10000000
    queue = mp.Queue()
    test_configs = []
    for t_config in get_test_configs(device_id, device_count):
        test_configs.append(t_config)
        if len(test_configs) < 256:
            continue

        p = mp.Process(
            target=worker,
            args=(
                expert_num,
                m,
                n,
                k,
                topk,
                dtype,
                test_count,
                use_fp8_w8a8,
                is_up,
                test_configs,
                queue,
            ),
        )
        p.start()
        p.join()
        while len(test_configs) != 0:
            try:
                cost_time = queue.get_nowait()
                logger.info(f"get {test_configs[0]} cost_time: {cost_time}")
                if cost_time < best_cost_time:
                    best_config = test_configs[0]
                    best_cost_time = cost_time
                    logger.info(f"cur best : {best_config} {best_cost_time}")
                del test_configs[0:1]
            except:
                del test_configs[0:16]
                logger.info(f"cur best : {best_config} {best_cost_time}")
                break

    while len(test_configs) != 0:
        p = mp.Process(
            target=worker,
            args=(
                expert_num,
                m,
                n,
                k,
                topk,
                dtype,
                test_count,
                use_fp8_w8a8,
                is_up,
                test_configs,
                queue,
            ),
        )
        p.start()
        p.join()

        while len(test_configs) != 0:
            try:
                cost_time = queue.get_nowait()
                logger.info(f"get {test_configs[0]} cost_time: {cost_time}")
                if cost_time < best_cost_time:
                    best_config = test_configs[0]
                    best_cost_time = cost_time
                    logger.info(f"cur best : {best_config} {best_cost_time}")
                del test_configs[0:1]
            except:
                del test_configs[0:16]
                logger.info(f"cur best : {best_config} {best_cost_time}")
                break

    logger.info(f"{best_config} best cost: {best_cost_time}")
    return best_config, best_cost_time


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    from lightllm.utils.tuning_utils import mp_tuning
    from lightllm.common.fused_moe.moe_kernel_configs import MoeGroupedGemmKernelConfig

    # tuning to get deepseekv2 lite configs and store
    up_dict = {}
    for m in [1, 8, 64, 128, 256, 512, 1024, 4096, 8192]:
        ans = mp_tuning(
            tuning_configs,
            {
                "expert_num": 64,
                "m": m,
                "n": 1408 // 2,
                "k": 2048,
                "topk": 6,
                "dtype": torch.bfloat16,
                "test_count": 20,
                "use_fp8_w8a8": False,
                "is_up": True,
            },
        )
        up_dict[m] = ans
        MoeGroupedGemmKernelConfig.save_config(
            N=1408,
            K=2048,
            topk_num=6,
            expert_num=64,
            mul_routed_weight=False,
            use_fp8_w8a8=False,
            out_dtype=str(torch.bfloat16),
            config_json=up_dict,
        )

    down_dict = {}
    for m in [1, 8, 64, 128, 256, 512, 1024, 4096, 8192]:
        ans = mp_tuning(
            tuning_configs,
            {
                "expert_num": 64,
                "m": m,
                "n": 1408 // 2,
                "k": 2048,
                "topk": 6,
                "dtype": torch.bfloat16,
                "test_count": 20,
                "use_fp8_w8a8": False,
                "is_up": False,
            },
        )
        down_dict[m] = ans
        MoeGroupedGemmKernelConfig.save_config(
            N=2048,
            K=1408 // 2,
            topk_num=1,
            expert_num=64,
            mul_routed_weight=True,
            use_fp8_w8a8=False,
            out_dtype=str(torch.bfloat16),
            config_json=down_dict,
        )
