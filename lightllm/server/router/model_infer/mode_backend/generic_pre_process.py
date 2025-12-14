import torch
import numpy as np
from typing import List, Tuple
from lightllm.server.router.model_infer.infer_batch import InferReq, g_infer_context
from lightllm.common.basemodel.infer_lock import g_infer_state_lock
from lightllm.common.basemodel.batch_objs import ModelInput
from lightllm.utils.envs_utils import (
    enable_diverse_mode_gqa_decode_fast_kernel,
    get_diverse_max_batch_shared_group_size,
)


def prepare_prefill_inputs(
    req_objs: List[InferReq], is_chuncked_mode: bool, is_multimodal: bool = False
) -> Tuple[ModelInput, List[InferReq]]:
    run_reqs = []
    total_token_num = 0
    prefix_total_token_num = 0
    input_ids = []
    b_req_idx = []
    b_seq_len = []
    b_q_seq_len = []
    batch_multimodal_params = []
    b_ready_cache_len = []
    b_mtp_index = []
    b_prefill_has_output = []

    for req in req_objs:
        run_reqs.append(req)
        batch_multimodal_params.append(req.multimodal_params)
        b_req_idx.append(req.req_idx)

        if is_chuncked_mode:
            input_token_ids = req.get_chuncked_input_token_ids()
        else:
            input_token_ids = req.get_input_token_ids()

        b_prefill_has_output.append(False if len(input_token_ids) < req.get_cur_total_len() else True)

        seq_len = len(input_token_ids)
        input_token_len = seq_len - req.cur_kv_len

        input_id = input_token_ids[req.cur_kv_len :]

        b_seq_len.append(seq_len)
        b_q_seq_len.append(input_token_len)
        input_ids.append(input_id)
        total_token_num += seq_len
        prefix_total_token_num += req.cur_kv_len
        b_ready_cache_len.append(req.cur_kv_len)
        b_mtp_index.append(0)

    max_kv_seq_len = max(b_seq_len)
    max_cache_len = max(b_ready_cache_len)
    max_len_in_batch = max(b_q_seq_len)
    max_q_seq_len = max(b_q_seq_len)

    input_ids = np.concatenate(input_ids, dtype=np.int64)
    input_ids = torch.tensor(input_ids, dtype=torch.int64, device="cpu")
    b_req_idx = torch.tensor(b_req_idx, dtype=torch.int32, device="cpu")
    b_seq_len = torch.tensor(b_seq_len, dtype=torch.int32, device="cpu")
    b_mtp_index = torch.tensor(b_mtp_index, dtype=torch.int32, device="cpu")
    b_ready_cache_len = torch.tensor(b_ready_cache_len, dtype=torch.int32, device="cpu")
    b_q_seq_len = torch.tensor(b_q_seq_len, dtype=torch.int32, device="cpu")
    b_prefill_start_loc = b_q_seq_len.cumsum(dim=0, dtype=torch.int32) - b_q_seq_len

    # dynamic prompt cache 准备 token
    g_infer_state_lock.acquire()
    if g_infer_context.radix_cache is not None:
        g_infer_context.radix_cache.free_radix_cache_to_get_enough_token(input_ids.shape[0])
    mem_indexes = g_infer_context.req_manager.mem_manager.alloc(input_ids.shape[0])
    g_infer_state_lock.release()

    model_input = ModelInput(
        batch_size=b_seq_len.shape[0],
        total_token_num=total_token_num,
        max_len_in_batch=max_len_in_batch,
        max_q_seq_len=max_q_seq_len,
        max_kv_seq_len=max_kv_seq_len,
        max_cache_len=max_cache_len,
        input_ids=input_ids,
        mem_indexes_cpu=mem_indexes,
        b_req_idx=b_req_idx,
        b_mtp_index=b_mtp_index,
        b_seq_len=b_seq_len,
        b_ready_cache_len=b_ready_cache_len,
        b_prefill_start_loc=b_prefill_start_loc,
        is_prefill=True,
        b_prefill_has_output_cpu=b_prefill_has_output,
        prefix_total_token_num=prefix_total_token_num,
    )
    if is_multimodal:
        model_input.multimodal_params = batch_multimodal_params

    return model_input, run_reqs


def prepare_decode_inputs(req_objs: List[InferReq]) -> Tuple[ModelInput, List[InferReq]]:
    run_reqs: List[InferReq] = []
    total_token_num = 0
    max_len_in_batch = 0
    b_req_idx = []
    b_mtp_index = []
    b_seq_len = []
    b_q_seq_len = []
    multimodal_params = []
    for req in req_objs:
        run_reqs.append(req)
        b_req_idx.append(req.req_idx)
        seq_len = req.get_cur_total_len()
        assert req.cur_kv_len == seq_len - 1, f"{req.cur_kv_len} {seq_len}"
        b_seq_len.append(seq_len)
        total_token_num += seq_len
        max_len_in_batch = max(max_len_in_batch, seq_len)
        b_mtp_index.append(0)
        multimodal_params.append(req.multimodal_params)
        # process the draft tokens.
        for step in range(req.mtp_step):
            run_reqs.append(req)
            b_req_idx.append(req.req_idx)
            seq_len += 1
            b_seq_len.append(seq_len)
            total_token_num += seq_len
            max_len_in_batch = max(max_len_in_batch, seq_len)
            b_mtp_index.append(step + 1)
            multimodal_params.append(req.multimodal_params)
        b_q_seq_len.append(req.mtp_step + 1)

    max_kv_seq_len = max(b_seq_len)
    max_q_seq_len = max(b_q_seq_len)

    b_req_idx = torch.tensor(b_req_idx, dtype=torch.int32, device="cpu")
    b_seq_len = torch.tensor(b_seq_len, dtype=torch.int32, device="cpu")
    b_mtp_index = torch.tensor(b_mtp_index, dtype=torch.int32, device="cpu")

    if enable_diverse_mode_gqa_decode_fast_kernel():
        b_shared_seq_len, b_mark_shared_group = build_diverse_shared_group_infos(run_reqs=run_reqs)
    else:
        b_shared_seq_len = None
        b_mark_shared_group = None

    # dynamic prompt cache 准备 token
    g_infer_state_lock.acquire()
    if g_infer_context.radix_cache is not None:
        g_infer_context.radix_cache.free_radix_cache_to_get_enough_token(b_seq_len.shape[0])
    mem_indexes = g_infer_context.req_manager.mem_manager.alloc(b_seq_len.shape[0])
    g_infer_state_lock.release()

    model_input = ModelInput(
        batch_size=b_seq_len.shape[0],
        total_token_num=total_token_num,
        max_len_in_batch=max_len_in_batch,
        max_q_seq_len=max_q_seq_len,
        max_kv_seq_len=max_kv_seq_len,
        input_ids=None,
        mem_indexes_cpu=mem_indexes,
        b_req_idx=b_req_idx,
        b_mtp_index=b_mtp_index,
        b_seq_len=b_seq_len,
        b_shared_seq_len=b_shared_seq_len,
        b_mark_shared_group=b_mark_shared_group,
        is_prefill=False,
    )
    model_input.multimodal_params = multimodal_params
    return model_input, run_reqs


def build_diverse_shared_group_infos(run_reqs: List[InferReq]) -> Tuple[torch.Tensor, torch.Tensor]:
    # b_shared_seq_len 和 b_mark_shared_group 只会在 diverse_mode 下的 decode 阶段真正被使用的参数,
    # 用于记录请求间的共享关系。
    # 举列说明:
    # b_shared_seq_len : [10, 10, 10, 11, 11, 11, 11]
    # b_mark_shared_group: [0, 0, 3, 0, 0, 0, 4]
    # b_mark_shared_group 中每一个不为0的位置都代表其与前面多少个请求形成一个共享前缀组。属于
    # 同一个共享前缀组的请求, 其在对应的 b_shared_seq_len 中的内容必然相同。某些模式可以利用这两个
    # 输入加速算子的运行。
    max_batch_shared_group_size = get_diverse_max_batch_shared_group_size()
    b_shared_seq_len = [req.get_radix_cache_shared_len() for req in run_reqs]
    b_mark_shared_group = []
    shared_nodes = [req.shared_kv_node for req in run_reqs]
    _current_group = []
    for node in shared_nodes:
        if not _current_group:
            _current_group.append(node)
        elif node == _current_group[-1]:
            _current_group.append(node)
        else:
            b_mark_shared_group.extend([0 for _ in range(len(_current_group))])
            b_mark_shared_group[-1] = len(_current_group)
            _current_group.clear()
            _current_group.append(node)

        if len(_current_group) == max_batch_shared_group_size:
            b_mark_shared_group.extend([0 for _ in range(len(_current_group))])
            b_mark_shared_group[-1] = len(_current_group)
            _current_group.clear()
    if _current_group:
        b_mark_shared_group.extend([0 for _ in range(len(_current_group))])
        b_mark_shared_group[-1] = len(_current_group)
        _current_group.clear()

    assert len(b_mark_shared_group) == len(run_reqs)
    # 如果一个 shared group 的长度为1， 则将其共享长度强制修改为0， 避免无效计算，提升
    # 算子执行效率。
    b_shared_seq_len = [
        0 if group_size == 1 else shared_len for shared_len, group_size in zip(b_shared_seq_len, b_mark_shared_group)
    ]
    b_shared_seq_len = torch.tensor(b_shared_seq_len, dtype=torch.int32, device="cpu")
    b_mark_shared_group = torch.tensor(b_mark_shared_group, dtype=torch.int32, device="cpu")
    return b_shared_seq_len, b_mark_shared_group
