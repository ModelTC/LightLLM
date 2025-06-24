import torch
import time
import os
import torch.multiprocessing as mp
from typing import List
from lightllm.utils.log_utils import init_logger
from lightllm.models.llama.triton_kernel.gqa_flash_decoding_vsm import (
    gqa_token_decode_attention_flash_decoding_vsm,
    GQAVSMDecodeAttentionKernelConfig,
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
        k = torch.randn(kv_shape, device="cuda", dtype=dtype) / 10
        v = torch.randn(kv_shape, device="cuda", dtype=dtype) / 10
        o_tensor = torch.empty_like(q)
        arg_list, kwargs = (q, k, v, infer_state), dict(out=o_tensor, **run_config)
        args.append((arg_list, kwargs))

    tensor_dict = {}

    def inner_alloc_func(shape, dtype=torch.float32, device="cuda"):
        shape = tuple(shape)
        if shape not in tensor_dict:
            ans = torch.empty(shape, dtype=dtype, device=device)
            tensor_dict[shape] = ans
            return ans
        else:
            return tensor_dict[shape]

    graph = torch.cuda.CUDAGraph()
    arg_list, kwargs = args[0]
    gqa_token_decode_attention_flash_decoding_vsm(*arg_list, **kwargs)
    with torch.cuda.graph(graph):
        for index in range(test_count):
            arg_list, kwargs = args[index]
            gqa_token_decode_attention_flash_decoding_vsm(*arg_list, **kwargs)

    graph.replay()

    torch.cuda.synchronize()
    start = time.time()
    # graph.replay()
    torch.cuda.synchronize()

    cost_time = (time.time() - start) * 1000

    logger.info(f"fp16 {test_seq_len} cost time: {cost_time} ms")
    return cost_time


def worker(
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
    for block_n in [16]:
        # for block_n in [16, 32, 64, 128]:
        for block_q_head in [
            16,
        ]:
            for stage1_num_warps in [
                2,
            ]:
                # for stage1_num_warps in [2, 4, 8, 16]:
                for stage1_num_stages in [
                    1,
                    # 2,
                    # 3,
                ]:
                    for stage2_num_warps in [
                        2,
                    ]:
                        # for stage2_num_warps in [1, 2, 4]:
                        for stage2_num_stages in [
                            1,
                            # 2,
                            # 3,
                        ]:
                            t_config = {
                                "BLOCK_N": block_n,
                                "BLOCK_Q_HEAD": block_q_head,
                                "stage1_num_warps": stage1_num_warps,
                                "stage1_num_stages": stage1_num_stages,
                                "stage2_num_warps": stage2_num_warps,
                                "stage2_num_stages": stage2_num_stages,
                            }
                            if index % split_count == split_id:
                                yield t_config
                                index += 1
                            else:
                                index += 1


def tuning_configs(
    device_id: int,  # use for mult mp tunning
    device_count: int,
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
    from lightllm.models.deepseek2.triton_kernel.gqa_flash_decoding_config import MlaDecodeAttentionKernelConfig

    import collections

    store_json_ans = collections.defaultdict(dict)

    def config_iter():
        for q_head_num in [32]:
            for q_head_dim in [64, 128]:
                for group_size in [8, 16, 32]:
                    for batch_size in [1, 8, 16, 32, 64, 128, 256]:
                        for seq_len in [256, 512, 1024, 2048, 4096, 8192]:
                            if batch_size * seq_len > 128 * 1024 * 4:
                                continue
                            yield q_head_num, q_head_dim, group_size, batch_size, seq_len

    for q_head_num, q_head_dim, group_size, batch_size, seq_len in config_iter():

        kv_head_num = q_head_num // group_size
        ans = mp_tuning(
            tuning_configs,
            {
                "q_shape": [batch_size, q_head_num, q_head_dim],
                "kv_shape": [batch_size * seq_len, kv_head_num, q_head_dim],
                "test_seq_len": seq_len,
                "dtype": torch.half,
                "test_count": 1,
            },
        )
        store_json_ans[seq_len][batch_size] = ans

        GQAVSMDecodeAttentionKernelConfig.save_config(
            q_head_num=q_head_num,
            q_head_dim=q_head_dim,
            kv_head_num=kv_head_num,
            out_dtype=str(torch.half),
            config_json=store_json_ans,
        )
