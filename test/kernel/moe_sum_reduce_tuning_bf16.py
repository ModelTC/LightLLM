import os
import torch
import time
import torch.multiprocessing as mp
import itertools
from lightllm.common.fused_moe.moe_sum_reduce import MoeSumReduceKernelConfig, moe_sum_reduce
from lightllm.utils.watchdog_utils import Watchdog
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


@torch.no_grad()
def test_kernel(
    m: int,
    topk_num: int,
    hidden_dim: int,
    dtype: torch.dtype,
    test_count: int,
    **config,
):
    set_seed()
    input_tuples = []

    input = torch.randn((m, topk_num, hidden_dim), device="cuda", dtype=dtype) / 10
    output = torch.randn((m, hidden_dim), device="cuda", dtype=dtype)

    for _ in range(test_count):
        input_tuples.append((input.clone(), output.clone()))

    # warm_up
    moe_sum_reduce(input, output, run_config=config)

    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph):
        for index in range(test_count):
            input, output = input_tuples[index]
            moe_sum_reduce(input, output, run_config=config)

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
    m: int,
    topk_num: int,
    hidden_dim: int,
    dtype: torch.dtype,
    test_count: int,
    test_configs,
    queue,
):
    dog = Watchdog(timeout=10)
    dog.start()
    try:
        for index in range(len(test_configs)):
            cost_time = test_kernel(
                m=m,
                topk_num=topk_num,
                hidden_dim=hidden_dim,
                dtype=dtype,
                test_count=test_count,
                **test_configs[index],
            )
            dog.heartbeat()
            queue.put(cost_time)  # Put result in queue

    except Exception as ex:
        logger.error(str(ex))
        logger.exception(str(ex))
        import sys

        sys.exit(-1)
        pass


def get_test_configs(split_id, split_count):
    index = 0
    result = itertools.product(
        [1, 2, 4, 8, 16, 32],
        [64, 128, 256, 512, 1024],
        [1, 2, 4, 8, 16],
        [1, 2, 3, 4, 5, 6, 7, 8],
    )
    for BLOCK_M, BLOCK_DIM, num_warps, NUM_STAGE in result:

        t_config = {
            "BLOCK_M": BLOCK_M,
            "BLOCK_DIM": BLOCK_DIM,
            "num_warps": num_warps,
            "NUM_STAGE": NUM_STAGE,
        }
        if index % split_count == split_id:
            yield t_config
            index += 1
        else:
            index += 1


def tuning_configs(
    device_id: int,  # use for mult mp tunning
    device_count: int,
    m: int,
    topk_num: int,
    hidden_dim: int,
    dtype: torch.dtype,
    test_count: int,
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
                m,
                topk_num,
                hidden_dim,
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
                m,
                topk_num,
                hidden_dim,
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

    for hidden_dim in [1024, 2048, 4096, 5120, 8192]:
        for topk_num in [1, 6]:
            json_dict = {}
            for m in [1, 8, 64, 128, 200, 256, 512, 1024, 2048, 4096, 8192]:
                ans = mp_tuning(
                    tuning_configs,
                    {
                        "m": m,
                        "topk_num": topk_num,
                        "hidden_dim": hidden_dim,
                        "dtype": torch.bfloat16,
                        "test_count": 20,
                    },
                )
                json_dict[m] = ans
                MoeSumReduceKernelConfig.save_config(
                    topk_num=topk_num,
                    hidden_dim=hidden_dim,
                    out_dtype=str(torch.bfloat16),
                    config_json=json_dict,
                )
