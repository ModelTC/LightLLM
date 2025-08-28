import os
import torch
import time
import torch.multiprocessing as mp
import itertools
from lightllm.models.deepseek2.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.deepseek2.triton_kernel.rotary_emb_config import DeepseekV3RotaryKernelConfig
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
    M: int,
    Q_HEAD_NUM: int,
    K_HEAD_NUM: int,
    HEAD_DIM: int,
    dtype: torch.dtype,
    test_count: int,
    **config,
):
    set_seed()
    input_tuples = []

    q = torch.randn((M, Q_HEAD_NUM, HEAD_DIM), device="cuda", dtype=dtype) / 10
    k = torch.randn((M, K_HEAD_NUM, HEAD_DIM), device="cuda", dtype=dtype) / 10
    cos = torch.randn((M, HEAD_DIM // 2), device="cuda", dtype=dtype)
    sin = torch.randn((M, HEAD_DIM // 2), device="cuda", dtype=dtype)

    for _ in range(test_count):
        input_tuples.append((q.clone(), k.clone(), cos.clone(), sin.clone()))

    # warm_up
    rotary_emb_fwd(q=q, k=k, cos=cos, sin=sin, run_config=config)

    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph):
        for index in range(test_count):
            q, k, cos, sin = input_tuples[index]
            rotary_emb_fwd(q=q, k=k, cos=cos, sin=sin, run_config=config)

    graph.replay()

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    graph.replay()
    end_event.record()
    end_event.synchronize()

    cost_time = start_event.elapsed_time(end_event)

    logger.info(str(config))
    logger.info(f"bf16 {M} cost time: {cost_time} ms")
    return cost_time


def worker(
    M: int,
    Q_HEAD_NUM: int,
    K_HEAD_NUM: int,
    HEAD_DIM: int,
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
                M=M,
                Q_HEAD_NUM=Q_HEAD_NUM,
                K_HEAD_NUM=K_HEAD_NUM,
                HEAD_DIM=HEAD_DIM,
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
    result = itertools.product([1, 2, 4, 8, 16, 32], [1, 2, 4, 8], [1, 2, 3, 4, 5], [1, 2, 4, 8, 16])
    for BLOCK_SEQ, num_warps, num_stages, HEAD_PARALLEL_NUM in result:
        t_config = {
            "BLOCK_SEQ": BLOCK_SEQ,
            "HEAD_PARALLEL_NUM": HEAD_PARALLEL_NUM,
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
    M: int,
    Q_HEAD_NUM: int,
    K_HEAD_NUM: int,
    HEAD_DIM: int,
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
                M,
                Q_HEAD_NUM,
                K_HEAD_NUM,
                HEAD_DIM,
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
                M,
                Q_HEAD_NUM,
                K_HEAD_NUM,
                HEAD_DIM,
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

    logger.info(f"M {M} {best_config} best cost: {best_cost_time}")
    return best_config, best_cost_time


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    from lightllm.utils.tuning_utils import mp_tuning

    # for deepseekv3 600B

    for q_head_num in [128, 64, 32, 16, 8]:
        k_head_num = 1
        head_dim = 64
        dtype = torch.bfloat16
        for m in [1, 128, 256, 1024, 2048, 4096, 8192]:
            json_dict = {}
            ans = mp_tuning(
                tuning_configs,
                {
                    "M": m,
                    "Q_HEAD_NUM": q_head_num,
                    "K_HEAD_NUM": k_head_num,
                    "HEAD_DIM": head_dim,
                    "dtype": dtype,
                    "test_count": 20,
                },
            )
            json_dict[m] = ans
            DeepseekV3RotaryKernelConfig.save_config(
                Q_HEAD_NUM=q_head_num,
                K_HEAD_NUM=k_head_num,
                HEAD_DIM=head_dim,
                dtype=str(dtype),
                config_json=json_dict,
            )
