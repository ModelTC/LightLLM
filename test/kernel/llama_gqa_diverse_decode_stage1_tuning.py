import torch
import time
import os
import torch.multiprocessing as mp
from typing import List
from lightllm.utils.log_utils import init_logger
from lightllm.models.llama.triton_kernel.ppl_int8kv_flash_decoding_diverse_stage1 import (
    flash_decode_stage1,
    GQADiverseDecodeStage1KernelConfig,
)
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
def test_decode_attentions(
    block_seq: int,
    q_shape: List[int],
    kv_shape: List[int],
    test_seq_len: int,
    dtype: torch.dtype,
    test_count: int = 20,
    **run_config,
):
    set_seed()
    tmp_class = type("TestObj", (object,), {})
    state = tmp_class()
    state.batch_size = q_shape[0]
    state.max_len_in_batch = test_seq_len
    state.req_manager = tmp_class()
    state.req_manager.req_to_token_indexs = torch.zeros(
        (state.batch_size, state.max_len_in_batch), dtype=torch.int32, device="cuda"
    )
    state.req_manager.req_to_token_indexs.view(-1)[:] = torch.arange(
        0, state.batch_size * state.max_len_in_batch, step=1, dtype=torch.int32
    ).cuda()
    state.b_req_idx = torch.arange(0, state.batch_size, step=1, dtype=torch.int32).cuda()
    state.b_seq_len = torch.full((state.batch_size,), fill_value=test_seq_len, dtype=torch.int32).cuda()

    args = []
    batch_size = q_shape[0]
    q_head_dim = q_shape[2]
    q_head_num = q_shape[1]
    # kv_head_dim = kv_shape[2]
    kv_head_num = kv_shape[1]
    state.q_head_num = q_head_num
    state.q_head_dim = q_head_dim
    state.kv_head_num = kv_head_num
    state.softmax_scale = 1 / (q_head_dim ** 0.5)
    state.total_token_num = state.batch_size * test_seq_len

    infer_state = state
    for _ in range(test_count):
        q = torch.randn(q_shape, device="cuda", dtype=dtype) / 10
        k = torch.randint(low=-100, high=100, size=kv_shape, device="cuda", dtype=torch.int8)
        k_scale = torch.ones(
            size=(
                kv_shape[0],
                kv_shape[1],
                kv_shape[2] // 8,
            ),
            device="cuda",
            dtype=dtype,
        )
        v = torch.randint(low=-100, high=100, size=kv_shape, device="cuda", dtype=torch.int8)
        v_scale = torch.ones(
            size=(
                kv_shape[0],
                kv_shape[1],
                kv_shape[2] // 8,
            ),
            device="cuda",
            dtype=dtype,
        )
        mid_out = torch.zeros(
            size=(batch_size, q_head_num, (test_seq_len // block_seq) + 2, q_head_dim), dtype=q.dtype, device="cuda"
        )
        mid_out_logsumexp = torch.zeros(
            size=(batch_size, q_head_num, (test_seq_len // block_seq) + 2), dtype=q.dtype, device="cuda"
        )
        arg_list, kwargs = (
            q,
            k,
            k_scale,
            v,
            v_scale,
            infer_state.req_manager.req_to_token_indexs,
            infer_state.b_req_idx,
            infer_state.b_seq_len,
            torch.ones(size=(batch_size,), device="cuda", dtype=torch.int32),
            infer_state.max_len_in_batch,
            mid_out,
            mid_out_logsumexp,
            block_seq,
            4,
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

    logger.info(f"fp16 {test_seq_len} cost time: {cost_time} ms")
    return cost_time


def worker(
    block_seq: int,
    q_shape: List[int],
    kv_shape: List[int],
    test_seq_len: int,
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
            cost_time = test_decode_attentions(
                block_seq=block_seq,
                q_shape=q_shape,
                kv_shape=kv_shape,
                test_seq_len=test_seq_len,
                dtype=dtype,
                test_count=test_count,
                **tuning_config,
            )
            dog.heartbeat()
            queue.put(cost_time)  # Put result in queue
    except Exception as ex:
        logger.error(
            str(ex)
            + f" config {tuning_config} q_shape {q_shape} kv_shape {kv_shape} test_seq_len {test_seq_len} dtype {dtype}"
        )
        import sys
        import traceback

        traceback.print_exc()
        sys.exit(-1)
        pass


def get_test_configs(split_id, split_count):
    index = 0
    for block_n in [16, 32, 64]:
        for num_warps in [
            2,
            4,
            8,
            16,
        ]:
            # for stage1_num_warps in [2, 4, 8, 16]:
            for num_stages in [
                1,
                2,
                3,
                4,
                5,
                7,
                9,
                10,
                11,
            ]:
                t_config = {
                    "BLOCK_N": block_n,
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
    block_seq: int,
    q_shape: List[int],
    kv_shape: List[int],
    test_seq_len: int,
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
                block_seq,
                q_shape,
                kv_shape,
                test_seq_len,
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
                block_seq,
                q_shape,
                kv_shape,
                test_seq_len,
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
    import collections

    block_seq = 256

    store_json_ans = collections.defaultdict(dict)

    def config_iter():
        for batch_size in [8, 32, 128, 256]:
            for seq_len in [4096, 8192]:
                yield batch_size, seq_len

    for q_head_dim in [128]:
        for gqa_group_size in [2, 4, 5, 8, 16]:
            store_json_ans = collections.defaultdict(dict)
            for batch_size, seq_len in config_iter():
                ans = mp_tuning(
                    tuning_configs,
                    {
                        "block_seq": block_seq,
                        "q_shape": [batch_size, gqa_group_size, q_head_dim],
                        "kv_shape": [batch_size * seq_len, 1, q_head_dim],
                        "test_seq_len": seq_len,
                        "dtype": torch.half,
                        "test_count": 1,
                    },
                )
                store_json_ans[seq_len][batch_size] = ans

                GQADiverseDecodeStage1KernelConfig.save_config(
                    gqa_group_size=gqa_group_size,
                    q_head_dim=q_head_dim,
                    block_seq=block_seq,
                    out_dtype=str(torch.float16),
                    config_json=store_json_ans,
                )

                GQADiverseDecodeStage1KernelConfig.save_config(
                    gqa_group_size=gqa_group_size,
                    q_head_dim=q_head_dim,
                    block_seq=block_seq,
                    out_dtype=str(torch.bfloat16),
                    config_json=store_json_ans,
                )
