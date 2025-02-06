import torch
import torch.distributed as dist
import numpy as np
from typing import List
from lightllm.server.router.model_infer.infer_batch import g_infer_context, InferReq
from lightllm.utils.infer_utils import calculate_time
from lightllm.common.mem_manager import MemoryManager
from lightllm.common.basemodel.infer_lock import g_infer_state_lock


# @calculate_time(show=True, min_cost_ms=1)
def prepare_prefill_inputs(req_ids: List[int], is_multimodal=False):
    run_reqs = []
    nopad_total_token_num = 0
    nopad_max_len_in_batch = 0
    start_loc = 0
    padding_token_num = 0
    input_ids = []
    nopad_b_req_idx = []
    nopad_b_start_loc = []
    nopad_b_seq_len = []
    batch_multimodal_params = []
    b_ready_cache_len = []
    for request_id in req_ids:
        req: InferReq = g_infer_context.requests_mapping[request_id]

        run_reqs.append(req)
        batch_multimodal_params.append(req.multimodal_params)
        nopad_b_req_idx.append(req.req_idx)
        nopad_b_start_loc.append(start_loc)

        input_token_ids = req.get_input_token_ids()
        seq_len = len(input_token_ids)
        input_token_len = seq_len - req.cur_kv_len

        input_id = input_token_ids[req.cur_kv_len :]

        nopad_b_seq_len.append(seq_len)
        input_ids.append(input_id)
        nopad_total_token_num += seq_len
        nopad_max_len_in_batch = max(nopad_max_len_in_batch, input_token_len)
        b_ready_cache_len.append(req.cur_kv_len)
        start_loc += input_token_len

    # padding one fake req for prefill
    if len(input_ids) == 0:
        input_ids = [[1]]
        nopad_b_req_idx = [g_infer_context.req_manager.HOLD_REQUEST_ID]
        nopad_b_start_loc = [0]
        nopad_b_seq_len = [1]
        b_ready_cache_len = [0]
        padding_token_num = 1
    input_ids = np.concatenate(input_ids, dtype=np.int64)
    input_ids = torch.tensor(input_ids, dtype=torch.int64, device="cuda")
    nopad_b_req_idx = torch.tensor(nopad_b_req_idx, dtype=torch.int32, device="cuda")
    nopad_b_start_loc = torch.tensor(nopad_b_start_loc, dtype=torch.int32, device="cuda")
    nopad_b_seq_len = torch.tensor(nopad_b_seq_len, dtype=torch.int32, device="cuda")
    b_ready_cache_len = torch.tensor(b_ready_cache_len, dtype=torch.int32, device="cuda")

    # dynamic prompt cache 准备 token
    g_infer_state_lock.acquire()
    if g_infer_context.radix_cache is not None:
        g_infer_context.radix_cache.free_radix_cache_to_get_enough_token(input_ids.shape[0] - padding_token_num)
    mem_indexes = g_infer_context.req_manager.mem_manager.alloc(input_ids.shape[0] - padding_token_num).cuda()
    if padding_token_num > 0:
        padding_indexs = torch.full(
            (padding_token_num,),
            fill_value=g_infer_context.req_manager.mem_manager.dp_use_token_index,
            dtype=torch.int32,
            device="cuda",
        )
        mem_indexes = torch.cat((mem_indexes, padding_indexs), dim=0)
    g_infer_state_lock.release()

    kwargs = {
        "batch_size": nopad_b_seq_len.shape[0],
        "total_token_num": nopad_total_token_num,
        "max_len_in_batch": nopad_max_len_in_batch,
        "input_ids": input_ids,
        "mem_indexes": mem_indexes,
        "b_req_idx": nopad_b_req_idx,
        "b_start_loc": nopad_b_start_loc,
        "b_seq_len": nopad_b_seq_len,
        "b_ready_cache_len": b_ready_cache_len,
        "is_prefill": True,
    }
    if is_multimodal:
        kwargs["multimodal_params"] = batch_multimodal_params

    return kwargs, run_reqs, padding_token_num


# @calculate_time(show=True, min_cost_ms=1)
def prepare_decode_inputs(req_ids: List[int]):
    run_reqs = []
    nopad_total_token_num = 0
    nopad_max_len_in_batch = 0
    start_loc = 0
    padding_token_num = 0
    batch_size = 0
    input_ids = []
    nopad_b_req_idx = []
    nopad_b_start_loc = []
    nopad_b_seq_len = []
    batch_size = len(req_ids)
    for request_id in req_ids:
        req: InferReq = g_infer_context.requests_mapping[request_id]
        run_reqs.append(req)
        nopad_b_req_idx.append(req.req_idx)
        nopad_b_start_loc.append(start_loc)
        input_token_ids = req.get_input_token_ids()
        input_id = input_token_ids[-1]
        seq_len = len(input_token_ids)
        assert req.cur_kv_len == seq_len - 1
        nopad_b_seq_len.append(seq_len)
        input_ids.append(input_id)
        nopad_total_token_num += seq_len
        nopad_max_len_in_batch = max(nopad_max_len_in_batch, seq_len)
        start_loc += seq_len
    max_batch_size = torch.tensor([batch_size], dtype=torch.int32).cuda()
    dist.all_reduce(max_batch_size, op=dist.ReduceOp.MAX)
    max_batch_size = max_batch_size.item()
    # padding one fake req for prefill
    if batch_size < max_batch_size:
        padding_batch_size = max_batch_size - batch_size
        for _ in range(padding_batch_size):
            input_ids.append(1)
            nopad_b_req_idx.append(g_infer_context.req_manager.HOLD_REQUEST_ID)
            nopad_b_start_loc.append(start_loc)
            nopad_b_seq_len.append(2)
            start_loc += 2
            padding_token_num += 1

    input_ids = torch.tensor(input_ids, dtype=torch.int64, device="cuda")
    nopad_b_req_idx = torch.tensor(nopad_b_req_idx, dtype=torch.int32, device="cuda")
    nopad_b_start_loc = torch.tensor(nopad_b_start_loc, dtype=torch.int32, device="cuda")
    nopad_b_seq_len = torch.tensor(nopad_b_seq_len, dtype=torch.int32, device="cuda")

    # dynamic prompt cache 准备 token
    g_infer_state_lock.acquire()
    if g_infer_context.radix_cache is not None:
        g_infer_context.radix_cache.free_radix_cache_to_get_enough_token(input_ids.shape[0] - padding_token_num)
    mem_indexes = g_infer_context.req_manager.mem_manager.alloc(input_ids.shape[0] - padding_token_num).cuda()
    if padding_token_num > 0:
        padding_indexs = torch.full(
            (padding_token_num,),
            fill_value=g_infer_context.req_manager.mem_manager.dp_use_token_index,
            dtype=torch.int32,
            device="cuda",
        )
        mem_indexes = torch.cat((mem_indexes, padding_indexs), dim=0)
    g_infer_state_lock.release()

    kwargs = {
        "batch_size": nopad_b_seq_len.shape[0],
        "total_token_num": nopad_total_token_num,
        "max_len_in_batch": nopad_max_len_in_batch,
        "input_ids": input_ids,
        "mem_indexes": mem_indexes,
        "b_req_idx": nopad_b_req_idx,
        "b_start_loc": nopad_b_start_loc,
        "b_seq_len": nopad_b_seq_len,
        "is_prefill": False,
    }
    return kwargs, run_reqs, padding_token_num
