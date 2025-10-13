import torch
import time
import os
import torch.multiprocessing as mp
from typing import List
from lightllm.utils.log_utils import init_logger
from lightllm.common.quantization.triton_quant.fp8.fp8w8a8_block_gemm_kernel import w8a8_block_fp8_matmul
from lightllm.utils.watchdog_utils import Watchdog

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


@torch.no_grad()
def test_fp8_block_gemm(
    M: int,
    N: int,
    K: int,
    block_size: int,
    dtype: torch.dtype,
    test_count: int = 20,
    **run_config,
):
    set_seed()

    input_tuples = []
    for _ in range(test_count):
        A = torch.randn((M, K), dtype=torch.float32).cuda().to(torch.float8_e4m3fn)  # Activation
        B = torch.randn((K, N), dtype=torch.float32).cuda().to(torch.float8_e4m3fn)  # Weight
        Ascale = torch.ones((M, (K + block_size - 1) // block_size)).cuda()
        Bscale = torch.ones(((K + block_size - 1) // block_size, (N + block_size - 1) // block_size)).cuda()
        C = torch.randn((M, N), dtype=dtype).cuda()  # weight
        input_tuples.append((A, B, Ascale, Bscale, C))
    w8a8_block_fp8_matmul(A, B, Ascale, Bscale, C, (block_size, block_size), dtype, run_config=run_config)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for index in range(test_count):
            A, B, Ascale, Bscale, C = input_tuples[index]
            w8a8_block_fp8_matmul(
                A,
                B,
                Ascale,
                Bscale,
                C,
                (block_size, block_size),
                run_config=run_config,
            )

    graph.replay()
    torch.cuda.synchronize()
    start = time.time()
    graph.replay()
    torch.cuda.synchronize()
    cost_time = (time.time() - start) * 1000
    logger.info(f"fp8 mm {M} {N} {K} block {block_size} cost time: {cost_time} ms")
    return cost_time


def worker(
    M: int,
    N: int,
    K: int,
    block_size: int,
    dtype: torch.dtype,
    test_count: int,
    test_configs,
    queue,
):
    dog = Watchdog(timeout=10)
    dog.start()

    try:
        for index in range(len(test_configs)):
            tuning_config = test_configs[index]
            cost_time = test_fp8_block_gemm(
                M=M,
                N=N,
                K=K,
                block_size=block_size,
                dtype=dtype,
                test_count=test_count,
                **tuning_config,
            )
            dog.heartbeat()
            queue.put(cost_time)  # Put result in queue
    except Exception as ex:
        logger.exception(str(ex) + f"config {tuning_config}")
        import sys

        sys.exit(-1)
        pass


def get_test_configs(split_id, split_count):
    fp8_gemm_configs = [
        {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8, "num_stages": 3, "num_warps": 8},
        {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8, "num_stages": 4, "num_warps": 4},
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8, "num_stages": 4, "num_warps": 4},
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8, "num_stages": 4, "num_warps": 4},
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8, "num_stages": 4, "num_warps": 4},
        {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8, "num_stages": 4, "num_warps": 4},
        {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8, "num_stages": 5, "num_warps": 2},
        {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8, "num_stages": 5, "num_warps": 2},
        {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8, "num_stages": 3, "num_warps": 8},
        {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8, "num_stages": 3, "num_warps": 8},
        {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_M": 8, "num_stages": 4, "num_warps": 4},
        {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8, "num_stages": 4, "num_warps": 4},
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8, "num_stages": 4, "num_warps": 4},
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8, "num_stages": 4, "num_warps": 4},
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8, "num_stages": 4, "num_warps": 4},
        {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64, "GROUP_M": 8, "num_stages": 4, "num_warps": 4},
    ]
    index = 0
    for cfg in fp8_gemm_configs:
        if index % split_count == split_id:
            yield cfg
            index += 1
        else:
            index += 1


def tuning_configs(
    device_id: int,  # use for mult mp tunning
    device_count: int,
    M: int,
    N: int,
    K: int,
    block_size: int,
    dtype: torch.dtype,
    test_count: int,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    best_config, best_cost_time = None, 10000000
    queue = mp.Queue()
    test_configs = []
    for t_config in get_test_configs(device_id, device_count):
        test_configs.append(t_config)
        if len(test_configs) < 64:
            continue

        p = mp.Process(
            target=worker,
            args=(
                M,
                N,
                K,
                block_size,
                dtype,
                test_count,
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
                    logger.info(f"cur best {best_config}, {best_cost_time}")
                del test_configs[0:1]
            except:
                logger.info(f"cur best {best_config}, {best_cost_time}")
                del test_configs[0:1]
                break

    while len(test_configs) != 0:
        p = mp.Process(
            target=worker,
            args=(
                M,
                N,
                K,
                block_size,
                dtype,
                test_count,
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
                    logger.info(f"cur best {best_config}, {best_cost_time}")
                del test_configs[0:1]
            except:
                logger.info(f"cur best {best_config}, {best_cost_time}")
                del test_configs[0:1]
                break

    logger.info(f"{best_config} best cost: {best_cost_time}")
    return best_config, best_cost_time


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    from lightllm.utils.tuning_utils import mp_tuning
    from lightllm.common.quantization.triton_quant.fp8.fp8w8a8_block_gemm_kernel import Fp8BlockMMKernelConfig
    import collections

    block_size = 128
    store_json_ans = collections.defaultdict(dict)
    for N, K in [
        (256, 7168),
        (512, 7168),
        (576, 7168),
        (1536, 1536),
        (1536, 7168),
        (2048, 512),
        (2304, 7168),
        (8072, 7168),
        (4096, 512),
        (7168, 256),
        (7168, 1024),
        (7168, 1152),
        (7168, 2048),
        (7168, 2304),
        (7168, 16384),
        (7168, 18432),
        (24576, 7168),
        (32768, 512),
        (36864, 7168),
    ]:
        for M in [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 1536, 2048, 3072, 4096]:
            ans = mp_tuning(
                tuning_configs,
                {
                    "M": M,
                    "N": N,
                    "K": K,
                    "block_size": block_size,
                    "dtype": torch.bfloat16,
                    "test_count": 4,
                },
            )
            store_json_ans[M] = ans

            Fp8BlockMMKernelConfig.save_config(
                N=N,
                K=K,
                block_size=[block_size, block_size],
                out_dtype=torch.bfloat16,
                config_json=store_json_ans,
            )

    pass
