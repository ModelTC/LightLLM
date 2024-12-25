import torch
import time
import os
import torch.multiprocessing as mp
from typing import List
from lightllm.utils.log_utils import init_logger
from lightllm.models.deepseek2.triton_kernel.gqa_flash_decoding import gqa_token_decode_attention_flash_decoding

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
    q_nope_shape: List[int],
    q_rope_shape: List[int],
    kv_nope_shape: List[int],
    kv_rope_shape: List[int],
    test_seq_len: int,
    dtype: torch.dtype,
    test_count: int = 20,
    **run_config,
):
    set_seed()
    tmp_class = type("TestObj", (object,), {})
    infer_state = tmp_class()
    infer_state.batch_size = q_nope_shape[0]
    infer_state.max_len_in_batch = test_seq_len
    infer_state.req_manager = tmp_class()
    infer_state.req_manager.req_to_token_indexs = torch.zeros(
        (infer_state.batch_size, infer_state.max_len_in_batch), dtype=torch.int32, device="cuda"
    )
    infer_state.req_manager.req_to_token_indexs.view(-1)[:] = torch.arange(
        0, infer_state.batch_size * infer_state.max_len_in_batch, step=1, dtype=torch.int32
    ).cuda()
    infer_state.b_req_idx = torch.arange(0, infer_state.batch_size, step=1, dtype=torch.int32).cuda()
    infer_state.b_seq_len = torch.full((infer_state.batch_size,), fill_value=test_seq_len, dtype=torch.int32).cuda()

    input_tuples = []
    for _ in range(test_count):
        q_nope = torch.randn(q_nope_shape, device="cuda", dtype=dtype) / 10
        q_rope = torch.randn(q_rope_shape, device="cuda", dtype=dtype) / 10
        kv_buffer_shape = [
            (test_seq_len + 10) * infer_state.batch_size,
            kv_nope_shape[1],
            kv_nope_shape[2] + kv_rope_shape[2],
        ]
        kv_buffer = torch.randn(kv_buffer_shape, device="cuda", dtype=dtype) / 10

        kv_nope = kv_buffer[:, :, 0 : kv_nope_shape[2]]
        kv_rope = kv_buffer[:, :, kv_nope_shape[2] :]
        o_tensor = torch.empty_like(q_nope)
        input_tuples.append((q_nope, q_rope, kv_buffer, kv_nope, kv_rope, o_tensor))

    tensor_dict = {}

    def inner_alloc_func(shape, dtype=torch.float32, device="cuda"):
        shape = tuple(shape)
        if shape not in tensor_dict:
            ans = torch.empty(shape, dtype=dtype, device=device)
            tensor_dict[shape] = ans
            return ans
        else:
            return tensor_dict[shape]

    gqa_token_decode_attention_flash_decoding(
        q_nope,
        q_rope,
        kv_nope,
        kv_rope,
        infer_state,
        q_nope_shape[1],
        q_nope_shape[2],
        q_rope_shape[2],
        None,
        0.01,
        out=o_tensor,
        alloc_tensor_func=inner_alloc_func,
        **run_config,
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for index in range(test_count):
            q_nope, q_rope, kv_buffer, kv_nope, kv_rope, o_tensor = input_tuples[index]
            gqa_token_decode_attention_flash_decoding(
                q_nope,
                q_rope,
                kv_nope,
                kv_rope,
                infer_state,
                q_nope_shape[1],
                q_nope_shape[2],
                q_rope_shape[2],
                None,
                0.01,
                out=o_tensor,
                alloc_tensor_func=inner_alloc_func,
                **run_config,
            )

    graph.replay()

    torch.cuda.synchronize()
    start = time.time()
    graph.replay()
    torch.cuda.synchronize()

    cost_time = (time.time() - start) * 1000

    logger.info(f"bf16 {test_seq_len} cost time: {cost_time} ms")
    return cost_time


def worker(
    q_nope_shape: List[int],
    q_rope_shape: List[int],
    kv_nope_shape: List[int],
    kv_rope_shape: List[int],
    test_seq_len: int,
    dtype: torch.dtype,
    test_count: int,
    test_configs,
    queue,
):
    try:
        for index in range(len(test_configs)):
            tuning_config = test_configs[index]
            cost_time = test_decode_attentions(
                q_nope_shape=q_nope_shape,
                q_rope_shape=q_rope_shape,
                kv_nope_shape=kv_nope_shape,
                kv_rope_shape=kv_rope_shape,
                test_seq_len=test_seq_len,
                dtype=dtype,
                test_count=test_count,
                **tuning_config,
            )
            queue.put(cost_time)  # Put result in queue
    except Exception as ex:
        logger.error(str(ex))
        import sys

        sys.exit(-1)
        pass


def get_test_configs(split_id, split_count):
    index = 0
    for block_seq in [32, 64, 128, 256]:
        for block_n in [16, 32, 64, 128, 256]:
            for block_q_head in [16, 32, 64]:
                for stage1_num_warps in [1, 2, 4, 8, 16]:
                    for stage1_num_stages in [1, 2, 3, 4, 5]:
                        for stage2_num_warps in [1, 2, 4, 8, 16]:
                            for stage2_num_stages in [1, 2, 3, 4, 5]:
                                if block_seq % block_n == 0:
                                    t_config = {
                                        "BLOCK_SEQ": block_seq,
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
    q_nope_shape: List[int],
    q_rope_shape: List[int],
    kv_nope_shape: List[int],
    kv_rope_shape: List[int],
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
                q_nope_shape,
                q_rope_shape,
                kv_nope_shape,
                kv_rope_shape,
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
                del test_configs[0:16]
                break

    while len(test_configs) != 0:
        p = mp.Process(
            target=worker,
            args=(
                q_nope_shape,
                q_rope_shape,
                kv_nope_shape,
                kv_rope_shape,
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
                del test_configs[0:16]
                break

    logger.info(f"{best_config} best cost: {best_cost_time}")
    return best_config, best_cost_time


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    from lightllm.utils.tuning_utils import mp_tuning
    from lightllm.models.deepseek2.triton_kernel.gqa_flash_decoding_config import MlaDecodeAttentionKernelConfig

    q_head_num = 16
    q_head_dim = 512
    q_rope_dim = 64
    import collections

    store_json_ans = collections.defaultdict(dict)
    for batch_size in [1, 8, 16, 32, 64, 128, 256]:
        for seq_len in [256, 512, 1024, 2048, 4096, 8192]:
            if batch_size * seq_len > 128 * 1024 * 4:
                continue

            ans = mp_tuning(
                tuning_configs,
                {
                    "q_nope_shape": [batch_size, q_head_num, q_head_dim],
                    "q_rope_shape": [batch_size, q_head_num, q_rope_dim],
                    "kv_nope_shape": [None, 1, q_head_dim],
                    "kv_rope_shape": [None, 1, q_rope_dim],
                    "test_seq_len": seq_len,
                    "dtype": torch.bfloat16,
                    "test_count": 20,
                },
            )
            store_json_ans[seq_len][batch_size] = ans

            MlaDecodeAttentionKernelConfig.save_config(
                q_head_num=q_head_num,
                q_head_dim=q_head_dim,
                q_rope_dim=q_rope_dim,
                out_dtype=str(torch.bfloat16),
                config_json=store_json_ans,
            )

    pass
