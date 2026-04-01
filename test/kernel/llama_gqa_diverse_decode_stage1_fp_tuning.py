import collections
import os
import torch
import torch.multiprocessing as mp
from typing import List
from pathlib import Path

from lightllm.utils.log_utils import init_logger
from lightllm.utils.watchdog_utils import Watchdog
from lightllm.common.basemodel.triton_kernel.att.decode_att.gqa.flash_decoding.gqa_flash_decoding_diverse_stage1 import (
    flash_decode_stage1,
    GQADiverseDecodeStage1KernelConfig,
)

logger = init_logger(__name__)


def set_seed():
    import random
    import numpy as np

    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def test_decode_attentions(
    block_seq: int,
    q_shape: List[int],
    kv_shape: List[int],
    max_batch_group_size: int,
    test_seq_len: int,
    shared_seq_len: int,
    max_len_in_batch: int,
    dtype: torch.dtype,
    test_count: int = 20,
    **run_config,
):
    set_seed()
    args = []
    batch_size = q_shape[0]
    q_head_num = q_shape[1]
    q_head_dim = q_shape[2]
    kv_head_num = kv_shape[1]

    for _ in range(test_count):
        q = torch.randn(q_shape, device="cuda", dtype=dtype) / 10
        k = torch.randn(kv_shape, device="cuda", dtype=dtype)
        v = torch.randn(kv_shape, device="cuda", dtype=dtype)
        req_to_tokens = torch.arange(0, batch_size * test_seq_len, step=1, dtype=torch.int32, device="cuda").view(
            batch_size, test_seq_len
        )
        b_req_idx = torch.arange(0, batch_size, step=1, dtype=torch.int32, device="cuda")
        b_shared_seq_len = torch.full((batch_size,), fill_value=shared_seq_len, dtype=torch.int32, device="cuda")
        b_mark_shared_group = torch.zeros((batch_size,), dtype=torch.int32, device="cuda")
        group_start = 0
        while group_start < batch_size:
            group_size = min(max_batch_group_size, batch_size - group_start)
            b_mark_shared_group[group_start + group_size - 1] = group_size
            if group_size == 1:
                b_shared_seq_len[group_start] = 0
            group_start += group_size
        mid_out = torch.zeros(
            size=(batch_size, q_head_num, (max_len_in_batch // block_seq) + 2, q_head_dim), dtype=q.dtype, device="cuda"
        )
        mid_out_logsumexp = torch.zeros(
            size=(batch_size, q_head_num, (max_len_in_batch // block_seq) + 2), dtype=torch.float32, device="cuda"
        )
        arg_list, kwargs = (
            q,
            k,
            v,
            req_to_tokens,
            b_req_idx,
            b_shared_seq_len,
            b_mark_shared_group,
            max_len_in_batch,
            mid_out,
            mid_out_logsumexp,
            block_seq,
            max_batch_group_size,
        ), dict(run_config=run_config)
        args.append((arg_list, kwargs))

    graph = torch.cuda.CUDAGraph()
    arg_list, kwargs = args[0]
    flash_decode_stage1(*arg_list, **kwargs)
    with torch.cuda.graph(graph):
        for index in range(test_count):
            arg_list, kwargs = args[index]
            flash_decode_stage1(*arg_list, **kwargs)

    graph.replay()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    graph.replay()
    end_event.record()
    end_event.synchronize()
    cost_time = start_event.elapsed_time(end_event=end_event)
    logger.info(f"{dtype} {test_seq_len} cost time: {cost_time} ms")
    return cost_time


def worker(
    block_seq,
    q_shape,
    kv_shape,
    max_batch_group_size,
    test_seq_len,
    shared_seq_len,
    max_len_in_batch,
    dtype,
    test_count,
    test_configs,
    queue,
):
    dog = Watchdog(timeout=10)
    dog.start()
    try:
        for tuning_config in test_configs:
            cost_time = test_decode_attentions(
                block_seq=block_seq,
                q_shape=q_shape,
                kv_shape=kv_shape,
                max_batch_group_size=max_batch_group_size,
                test_seq_len=test_seq_len,
                shared_seq_len=shared_seq_len,
                max_len_in_batch=max_len_in_batch,
                dtype=dtype,
                test_count=test_count,
                **tuning_config,
            )
            dog.heartbeat()
            queue.put(cost_time)
    except Exception as ex:
        logger.error(
            f"{ex} config {tuning_config} q_shape {q_shape} kv_shape {kv_shape} "
            f"test_seq_len {test_seq_len} shared_seq_len {shared_seq_len} max_len_in_batch {max_len_in_batch} dtype {dtype}"
        )
        import sys
        import traceback

        traceback.print_exc()
        sys.exit(-1)


def get_test_configs(split_id, split_count):
    index = 0
    for block_n in [16, 32, 64]:
        for num_warps in [2, 4, 8, 16]:
            for num_stages in [1, 2, 3, 4, 5, 7, 9, 10, 11]:
                config = {"BLOCK_N": block_n, "num_warps": num_warps, "num_stages": num_stages}
                if index % split_count == split_id:
                    yield config
                index += 1


def tuning_configs(
    device_id,
    device_count,
    block_seq,
    q_shape,
    kv_shape,
    max_batch_group_size,
    test_seq_len,
    shared_seq_len,
    max_len_in_batch,
    dtype,
    test_count,
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
                block_seq,
                q_shape,
                kv_shape,
                max_batch_group_size,
                test_seq_len,
                shared_seq_len,
                max_len_in_batch,
                dtype,
                test_count,
                test_configs,
                queue,
            ),
        )
        p.start()
        p.join()
        while test_configs:
            try:
                cost_time = queue.get_nowait()
                logger.info(f"get {test_configs[0]} cost_time: {cost_time}")
                if cost_time < best_cost_time:
                    best_config = test_configs[0]
                    best_cost_time = cost_time
                    logger.info(f"cur best {best_config}, {best_cost_time}")
            except Exception:
                logger.info(f"cur best {best_config}, {best_cost_time}")
            finally:
                del test_configs[0:1]

    while test_configs:
        p = mp.Process(
            target=worker,
            args=(
                block_seq,
                q_shape,
                kv_shape,
                max_batch_group_size,
                test_seq_len,
                shared_seq_len,
                max_len_in_batch,
                dtype,
                test_count,
                test_configs,
                queue,
            ),
        )
        p.start()
        p.join()
        while test_configs:
            try:
                cost_time = queue.get_nowait()
                logger.info(f"get {test_configs[0]} cost_time: {cost_time}")
                if cost_time < best_cost_time:
                    best_config = test_configs[0]
                    best_cost_time = cost_time
                    logger.info(f"cur best {best_config}, {best_cost_time}")
            except Exception:
                logger.info(f"cur best {best_config}, {best_cost_time}")
            finally:
                del test_configs[0:1]

    logger.info(f"{best_config} best cost: {best_cost_time}")
    return best_config, best_cost_time


def tune_one_shape(
    block_seq: int,
    batch_size: int,
    seq_len: int,
    shared_seq_len: int,
    max_len_in_batch: int,
    gqa_group_size: int,
    q_head_dim: int,
    max_batch_group_size: int,
    dtype: torch.dtype,
    test_count: int = 1,
):
    from lightllm.utils.tuning_utils import mp_tuning
    os.chdir(Path(__file__).resolve().parents[2])
    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    return mp_tuning(
        tuning_configs,
        {
            "block_seq": block_seq,
            "q_shape": [batch_size, gqa_group_size, q_head_dim],
            "kv_shape": [batch_size * seq_len, 1, q_head_dim],
            "max_batch_group_size": max_batch_group_size,
            "test_seq_len": seq_len,
            "shared_seq_len": shared_seq_len,
            "max_len_in_batch": max_len_in_batch,
            "dtype": dtype,
            "test_count": test_count,
        },
    )


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    from lightllm.utils.tuning_utils import mp_tuning

    block_seq = 256

    def config_iter():
        for batch_size in [1, 4, 10, 16, 24, 32]:
            yield batch_size, 3072

    for q_head_dim in [128]:
        for max_batch_group_size in [8]:
            for gqa_group_size in [4]:
                store_json_ans = collections.defaultdict(dict)
                for batch_size, seq_len in config_iter():
                    ans = tune_one_shape(
                        block_seq=block_seq,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        shared_seq_len=2900,
                        max_len_in_batch=8192,
                        gqa_group_size=gqa_group_size,
                        q_head_dim=q_head_dim,
                        max_batch_group_size=max_batch_group_size,
                        dtype=torch.bfloat16,
                        test_count=1,
                    )
                    store_json_ans[seq_len][batch_size] = ans
                    GQADiverseDecodeStage1KernelConfig.save_config(
                        gqa_group_size=gqa_group_size,
                        max_batch_group_size=max_batch_group_size,
                        q_head_dim=q_head_dim,
                        block_seq=block_seq,
                        out_dtype=str(torch.bfloat16),
                        config_json=store_json_ans,
                    )
