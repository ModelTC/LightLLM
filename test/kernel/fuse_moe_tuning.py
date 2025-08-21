import os
import argparse
import torch
import time
import torch.multiprocessing as mp
from lightllm.common.fused_moe.grouped_fused_moe import fused_experts_impl, moe_align, moe_align1, grouped_matmul
from typing import List
from lightllm.utils.log_utils import init_logger
from transformers import AutoConfig
import torch.nn.functional as F

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
    block_shape,
    num_fused_shared_experts: int,
    **config,
):
    set_seed()
    input_tuples = []

    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1_scale = w2_scale = None
    if num_fused_shared_experts > 0:
        expert_num += num_fused_shared_experts

    if use_fp8_w8a8:
        init_dtype = dtype
        w1 = torch.randn(expert_num, 2 * n, k, dtype=init_dtype).cuda()
        w2 = torch.randn(expert_num, k, 2 * n // 2, dtype=init_dtype).cuda()
        w1 = w1.to(torch.float8_e4m3fn)
        w2 = w2.to(torch.float8_e4m3fn)

        if block_shape is None:
            w1_scale = torch.randn(expert_num, dtype=torch.float32).cuda()
            w2_scale = torch.randn(expert_num, dtype=torch.float32).cuda()
        else:
            block_n, block_k = block_shape[0], block_shape[1]
            n_tiles_w1 = (2 * n + block_n - 1) // block_n
            n_tiles_w2 = (k + block_n - 1) // block_n
            k_tiles_w1 = (k + block_k - 1) // block_k
            k_tiles_w2 = (2 * n // 2 + block_k - 1) // block_k
            w1_scale = torch.rand((expert_num, n_tiles_w1, k_tiles_w1), dtype=torch.float32).cuda()
            w2_scale = torch.rand((expert_num, n_tiles_w2, k_tiles_w2), dtype=torch.float32).cuda()
    else:
        w1 = torch.randn(expert_num, 2 * n, k, dtype=dtype).cuda()
        w2 = torch.randn(expert_num, k, 2 * n // 2, dtype=dtype).cuda()

    rnd_logics = torch.randn(m, expert_num - num_fused_shared_experts, device="cuda")
    topk_values, topk_ids = torch.topk(rnd_logics, topk, dim=1)
    if num_fused_shared_experts > 0:
        # 存在融合共享专家的时候，需要pad 共享专家对应的id 到topk_ids 中
        pad_topk_ids = (
            torch.arange(
                start=expert_num - num_fused_shared_experts, end=expert_num, step=1, dtype=topk_ids.dtype, device="cuda"
            )
            .view(1, num_fused_shared_experts)
            .repeat(topk_ids.shape[0], 1)
        )
        topk_ids = torch.cat([topk_ids, pad_topk_ids], dim=1)
    topk_weights = torch.randn((m, topk + num_fused_shared_experts), device="cuda", dtype=dtype) / 10

    expert_to_tokens = torch.empty(
        (expert_num, (topk + num_fused_shared_experts) * m), dtype=torch.int32, device="cuda"
    )
    expert_to_weights = torch.empty(
        (expert_num, (topk + num_fused_shared_experts) * m), dtype=torch.float32, device="cuda"
    )
    moe_align(topk_ids=topk_ids, out=expert_to_tokens)
    expert_to_token_num = torch.empty((expert_num,), dtype=torch.int32, device="cuda")
    moe_align1(
        expert_to_tokens, topk_weights, expert_to_weights, expert_to_token_num, topk=topk + num_fused_shared_experts
    )

    out1 = torch.zeros((m * (topk + num_fused_shared_experts), 2 * n), dtype=torch.bfloat16, device="cuda")
    down_in = torch.zeros((m * (topk + num_fused_shared_experts), n), dtype=torch.bfloat16, device="cuda")
    out2 = torch.zeros((m * (topk + num_fused_shared_experts), k), dtype=torch.bfloat16, device="cuda")

    for _ in range(test_count):
        input_tuples.append(
            (
                a.clone(),
                w1.clone(),
                w2.clone(),
                w1_scale.clone() if w1_scale is not None else None,
                w2_scale.clone() if w2_scale is not None else None,
                topk_ids.clone(),
                topk_weights.clone(),
                out1.clone(),
                out2.clone(),
                down_in.clone(),
            )
        )

    if is_up:
        grouped_matmul(
            topk_ids.numel(),
            a,
            None,
            expert_to_token_num,
            expert_to_tokens,
            expert_to_weights=expert_to_weights,
            expert_weights=w1,
            expert_to_weights_scale=w1_scale,
            topk_num=topk,
            out=out1,
            mul_routed_weight=False,
            use_fp8_w8a8=use_fp8_w8a8,
            run_config=config,
        )
    else:
        grouped_matmul(
            topk_ids.numel(),
            down_in,
            None,
            expert_to_token_num,
            expert_to_tokens,
            expert_to_weights=expert_to_weights,
            expert_weights=w2,
            expert_to_weights_scale=w2_scale,
            topk_num=1,
            out=out2,
            mul_routed_weight=True,
            use_fp8_w8a8=use_fp8_w8a8,
            run_config=config,
        )

    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph):
        for index in range(test_count):
            a, w1, w2, w1_scale, w2_scale, topk_ids, topk_weights, out1, out2, down_in = input_tuples[index]
            if is_up:
                grouped_matmul(
                    topk_ids.numel(),
                    a,
                    None,
                    expert_to_token_num,
                    expert_to_tokens,
                    expert_to_weights=expert_to_weights,
                    expert_weights=w1,
                    expert_to_weights_scale=w1_scale,
                    topk_num=topk,
                    out=out1,
                    mul_routed_weight=False,
                    use_fp8_w8a8=use_fp8_w8a8,
                    run_config=config,
                )
            else:
                grouped_matmul(
                    topk_ids.numel(),
                    down_in,
                    None,
                    expert_to_token_num,
                    expert_to_tokens,
                    expert_to_weights=expert_to_weights,
                    expert_weights=w2,
                    expert_to_weights_scale=w2_scale,
                    topk_num=1,
                    out=out2,
                    mul_routed_weight=True,
                    use_fp8_w8a8=use_fp8_w8a8,
                    run_config=config,
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
    block_shape,
    num_fused_shared_experts: int,
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
                block_shape=block_shape,
                num_fused_shared_experts=num_fused_shared_experts,
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
    for num_stages in [
        1,
        2,
        3,
        4,
        5,
    ]:
        for GROUP_SIZE_M in [
            1,
            2,
            4,
        ]:
            for num_warps in [
                2,
                4,
                8,
            ]:
                for BLOCK_SIZE_M in [16, 32, 64, 128]:
                    for BLOCK_SIZE_N in [32, 64, 128]:
                        for BLOCK_SIZE_K in [32, 64, 128]:
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
    block_shape,
    num_fused_shared_experts: int,
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
                block_shape,
                num_fused_shared_experts,
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
                block_shape,
                num_fused_shared_experts,
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


def main(args):
    torch.multiprocessing.set_start_method("spawn")
    from lightllm.utils.tuning_utils import mp_tuning
    from lightllm.common.fused_moe.moe_kernel_configs import MoeGroupedGemmKernelConfig

    config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    if config.architectures[0] == "Qwen3MoeForCausalLM":
        expert_num = config.num_experts
        topk_num = config.num_experts_per_tok
        n = config.moe_intermediate_size // args.tp
    elif config.architectures[0] in ["DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM"]:
        expert_num = config.n_routed_experts
        topk_num = config.num_experts_per_tok
        n = config.moe_intermediate_size // args.tp
    else:
        pass

    hidden_dim = getattr(config, "hidden_size", None) or config.text_config.hidden_size
    print(n, hidden_dim)
    use_fp8_w8a8 = args.use_fp8_w8a8
    block_shape = None
    if hasattr(config, "quantization_config") and "weight_block_size" in config.quantization_config:
        block_shape = config.quantization_config["weight_block_size"]
        assert len(block_shape) == 2
        use_fp8_w8a8 = True

    up_dict = {}
    for m in [1, 8, 64, 128, 256, 512, 1024, 4096, 8192]:
        ans = mp_tuning(
            tuning_configs,
            {
                "expert_num": expert_num,
                "m": m,
                "n": n,
                "k": hidden_dim,
                "topk": topk_num,
                "dtype": torch.bfloat16,
                "test_count": 20,
                "use_fp8_w8a8": use_fp8_w8a8,
                "is_up": True,
                "block_shape": block_shape,
                "num_fused_shared_experts": args.num_fused_shared_experts,
            },
        )
        up_dict[m] = ans
        MoeGroupedGemmKernelConfig.save_config(
            N=n * 2,
            K=hidden_dim,
            topk_num=topk_num,
            expert_num=expert_num + args.num_fused_shared_experts,
            mul_routed_weight=False,
            use_fp8_w8a8=use_fp8_w8a8,
            out_dtype=str(torch.bfloat16),
            config_json=up_dict,
        )

    down_dict = {}
    for m in [1, 8, 64, 128, 256, 512, 1024, 4096, 8192, 16384, 32768]:
        ans = mp_tuning(
            tuning_configs,
            {
                "expert_num": expert_num,
                "m": m,
                "n": n,
                "k": hidden_dim,
                "topk": topk_num,
                "dtype": torch.bfloat16,
                "test_count": 20,
                "use_fp8_w8a8": use_fp8_w8a8,
                "is_up": False,
                "block_shape": block_shape,
                "num_fused_shared_experts": args.num_fused_shared_experts,
            },
        )
        down_dict[m] = ans

        MoeGroupedGemmKernelConfig.save_config(
            N=hidden_dim,
            K=n,
            topk_num=1,
            expert_num=expert_num + args.num_fused_shared_experts,
            mul_routed_weight=True,
            use_fp8_w8a8=use_fp8_w8a8,
            out_dtype=str(torch.bfloat16),
            config_json=down_dict,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="deepseek-ai/DeepSeek-R1")
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--use_fp8_w8a8", action="store_true")
    parser.add_argument("--num_fused_shared_experts", type=int, default=0)
    args = parser.parse_args()
    main(args)
