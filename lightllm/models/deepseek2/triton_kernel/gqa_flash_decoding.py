import os
import torch
import torch.multiprocessing as mp
from typing import List
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def gqa_token_decode_attention_flash_decoding(
    q_nope,
    q_rope,
    kv_nope,
    kv_rope,
    infer_state,
    q_head_num,
    kv_lora_rank,
    q_rope_dim,
    qk_nope_head_dim,
    softmax_scale,
    out=None,
    alloc_tensor_func=torch.empty,
):
    if hasattr(os, "tuning_config"):
        BLOCK_SEQ = os.tuning_config["BLOCK_SEQ"]
    else:
        BLOCK_SEQ = 64
    batch_size = infer_state.batch_size
    max_len_in_batch = infer_state.max_len_in_batch
    calcu_shape1 = (batch_size, q_head_num, kv_lora_rank)
    calcu_shape2 = (batch_size, q_head_num, q_rope_dim)

    from .gqa_flash_decoding_stage1 import flash_decode_stage1
    from .gqa_flash_decoding_stage2 import flash_decode_stage2

    o_tensor = alloc_tensor_func(q_nope.shape, q_nope.dtype, q_nope.device) if out is None else out

    mid_o = alloc_tensor_func(
        [batch_size, q_head_num, max_len_in_batch // BLOCK_SEQ + 1, kv_lora_rank], dtype=torch.float32, device="cuda"
    )
    mid_o_logexpsum = alloc_tensor_func(
        [batch_size, q_head_num, max_len_in_batch // BLOCK_SEQ + 1], dtype=torch.float32, device="cuda"
    )

    flash_decode_stage1(
        q_nope.view(calcu_shape1),
        q_rope.view(calcu_shape2),
        kv_nope,
        kv_rope,
        infer_state.req_manager.req_to_token_indexs,
        infer_state.b_req_idx,
        infer_state.b_seq_len,
        infer_state.max_len_in_batch,
        mid_o,
        mid_o_logexpsum,
        BLOCK_SEQ,
        softmax_scale,
    )
    flash_decode_stage2(mid_o, mid_o_logexpsum, infer_state.b_seq_len, o_tensor.view(calcu_shape1), BLOCK_SEQ)
    return o_tensor


@torch.no_grad()
def test_decode_attentions(
    q_nope_shape: List[int],
    q_rope_shape: List[int],
    kv_nope_shape: List[int],
    kv_rope_shape: List[int],
    test_seq_len: int,
    dtype: torch.dtype,
    test_count: int = 20,
):
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
    )

    import time

    torch.cuda.synchronize()
    start = time.time()
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
        )
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
            os.tuning_config = test_configs[index]
            cost_time = test_decode_attentions(
                q_nope_shape=q_nope_shape,
                q_rope_shape=q_rope_shape,
                kv_nope_shape=kv_nope_shape,
                kv_rope_shape=kv_rope_shape,
                test_seq_len=test_seq_len,
                dtype=dtype,
                test_count=test_count,
            )
            queue.put(cost_time)  # Put result in queue
    except Exception as ex:
        logger.error(str(ex))
        import sys

        sys.exit(-1)
        pass


def get_test_configs():
    for block_seq in [16, 32, 64, 128, 256]:
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
                                    yield t_config


def tuning_configs(
    q_nope_shape: List[int],
    q_rope_shape: List[int],
    kv_nope_shape: List[int],
    kv_rope_shape: List[int],
    test_seq_len: int,
    dtype: torch.dtype,
    test_count: int = 20,
):

    best_config, best_cost_time = None, 10000000
    queue = mp.Queue()
    test_configs = []
    for t_config in get_test_configs():
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
        get_count = 0
        while get_count < len(test_configs):
            try:
                cost_time = queue.get_nowait()
                logger.info(f"get {test_configs[get_count]} cost_time: {cost_time}")
                get_count += 1
                if cost_time < best_cost_time:
                    best_config = t_config
                    best_cost_time = cost_time
            except:
                break
        test_configs = test_configs[get_count + 1 :]

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
    get_count = 0
    while get_count < len(test_configs):
        try:
            cost_time = queue.get_nowait()
            logger.info(f"get {test_configs[get_count]} cost_time: {cost_time}")
            get_count += 1
            if cost_time < best_cost_time:
                best_config = t_config
                best_cost_time = cost_time
        except:
            break
    test_configs = test_configs[get_count + 1 :]

    logger.info(f"{best_config} best cost: {best_cost_time}")


if __name__ == "__main__":
    # q_node shape torch.Size([200, 16, 512]) q_rope shape torch.Size([200, 16, 64])
    # kv shape torch.Size([400000, 1, 512]) kv_rope torch.Size([400000, 1, 64])
    tuning_configs(
        q_nope_shape=[200, 16, 512],
        q_rope_shape=[200, 16, 64],
        kv_nope_shape=[None, 1, 512],
        kv_rope_shape=[None, 1, 64],
        test_seq_len=1035,
        dtype=torch.bfloat16,
        test_count=20,
    )
    pass
