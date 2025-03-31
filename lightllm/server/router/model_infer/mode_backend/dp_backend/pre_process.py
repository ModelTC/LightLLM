import torch
import torch.distributed as dist
import numpy as np
import triton
from typing import List
from lightllm.server.router.model_infer.infer_batch import g_infer_context, InferReq
from lightllm.utils.infer_utils import calculate_time
from lightllm.common.mem_manager import MemoryManager
from lightllm.common.basemodel.infer_lock import g_infer_state_lock
from lightllm.common.basemodel.microbatch_overlap_objs import DecodeMicroBatch, PrefillMicroBatch


def padded_prepare_prefill_inputs(req_objs: List[InferReq], max_prefill_num: int, is_multimodal=False):
    assert max_prefill_num != 0
    run_reqs = []
    nopad_total_token_num = 0
    nopad_max_len_in_batch = 0
    start_loc = 0
    # 当前 dp 没有请求的时候，需要进行 dp 操作。
    padded_req_num = 1 if len(req_objs) == 0 else 0
    input_ids = []
    nopad_b_req_idx = []
    nopad_b_start_loc = []
    nopad_b_seq_len = []
    batch_multimodal_params = []
    b_ready_cache_len = []
    for req in req_objs:

        run_reqs.append(req)
        batch_multimodal_params.append(req.multimodal_params)
        nopad_b_req_idx.append(req.req_idx)
        nopad_b_start_loc.append(start_loc)

        input_token_ids = req.get_chuncked_input_token_ids()
        seq_len = len(input_token_ids)
        input_token_len = seq_len - req.cur_kv_len
        input_id = input_token_ids[req.cur_kv_len :]

        nopad_b_seq_len.append(seq_len)
        input_ids.append(input_id)
        nopad_total_token_num += seq_len
        nopad_max_len_in_batch = max(nopad_max_len_in_batch, input_token_len)
        b_ready_cache_len.append(req.cur_kv_len)
        start_loc += input_token_len

    # padding fake req for prefill
    for _ in range(padded_req_num):
        input_ids.append([1])
        nopad_b_req_idx.append(g_infer_context.req_manager.HOLD_REQUEST_ID)
        nopad_b_start_loc.append(start_loc)
        start_loc += 1
        nopad_b_seq_len.append(1)
        b_ready_cache_len.append(0)
        nopad_total_token_num += 1
        nopad_max_len_in_batch = max(nopad_max_len_in_batch, 1)

    input_ids = np.concatenate(input_ids, dtype=np.int64)
    input_ids = torch.tensor(input_ids, dtype=torch.int64, device="cuda")
    nopad_b_req_idx = torch.tensor(nopad_b_req_idx, dtype=torch.int32, device="cuda")
    nopad_b_start_loc = torch.tensor(nopad_b_start_loc, dtype=torch.int32, device="cuda")
    nopad_b_seq_len = torch.tensor(nopad_b_seq_len, dtype=torch.int32, device="cuda")
    b_ready_cache_len = torch.tensor(b_ready_cache_len, dtype=torch.int32, device="cuda")

    # dynamic prompt cache 准备 token
    g_infer_state_lock.acquire()
    if g_infer_context.radix_cache is not None:
        g_infer_context.radix_cache.free_radix_cache_to_get_enough_token(input_ids.shape[0] - padded_req_num)
    mem_indexes = g_infer_context.req_manager.mem_manager.alloc(input_ids.shape[0] - padded_req_num).cuda()
    g_infer_state_lock.release()
    if padded_req_num > 0:
        padding_mem_indexs = torch.full(
            (padded_req_num,),
            fill_value=g_infer_context.req_manager.mem_manager.HOLD_TOKEN_MEMINDEX,
            dtype=torch.int32,
            device="cuda",
        )
        mem_indexes = torch.cat((mem_indexes, padding_mem_indexs), dim=0)

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

    return kwargs, run_reqs, padded_req_num


def padded_prepare_decode_inputs(req_objs: List[InferReq], max_decode_num: int, is_multimodal=False):
    assert max_decode_num != 0
    run_reqs = []
    nopad_total_token_num = 0
    nopad_max_len_in_batch = 0
    start_loc = 0
    input_ids = []
    nopad_b_req_idx = []
    nopad_b_start_loc = []
    nopad_b_seq_len = []
    padded_req_num = 1 if len(req_objs) == 0 else 0
    for req in req_objs:
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

    # padding fake req for decode
    for _ in range(padded_req_num):
        input_ids.append(1)
        seq_len = 2
        nopad_b_req_idx.append(g_infer_context.req_manager.HOLD_REQUEST_ID)
        nopad_b_start_loc.append(start_loc)
        nopad_b_seq_len.append(seq_len)
        start_loc += seq_len
        nopad_total_token_num += seq_len
        nopad_max_len_in_batch = max(nopad_max_len_in_batch, seq_len)

    input_ids = torch.tensor(input_ids, dtype=torch.int64, device="cuda")
    nopad_b_req_idx = torch.tensor(nopad_b_req_idx, dtype=torch.int32, device="cuda")
    nopad_b_start_loc = torch.tensor(nopad_b_start_loc, dtype=torch.int32, device="cuda")
    nopad_b_seq_len = torch.tensor(nopad_b_seq_len, dtype=torch.int32, device="cuda")

    # dynamic prompt cache 准备 token
    g_infer_state_lock.acquire()
    if g_infer_context.radix_cache is not None:
        g_infer_context.radix_cache.free_radix_cache_to_get_enough_token(input_ids.shape[0] - padded_req_num)
    mem_indexes = g_infer_context.req_manager.mem_manager.alloc(input_ids.shape[0] - padded_req_num).cuda()
    g_infer_state_lock.release()

    if padded_req_num > 0:
        padding_indexs = torch.full(
            (padded_req_num,),
            fill_value=g_infer_context.req_manager.mem_manager.HOLD_TOKEN_MEMINDEX,
            dtype=torch.int32,
            device="cuda",
        )
        mem_indexes = torch.cat((mem_indexes, padding_indexs), dim=0)

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
    return kwargs, run_reqs, padded_req_num


def padded_overlap_prepare_decode_inputs(req_objs: List[InferReq], max_decode_num: int, is_multimodal=False):
    assert max_decode_num != 0
    micro_batch_size = triton.cdiv(max_decode_num, 2)
    micro_batch1_req_num = triton.cdiv(len(req_objs), 2)
    micro_batch, run_reqs, padded_req_num = _padded_prepare_decode_micro_batch(
        req_objs[0:micro_batch1_req_num], micro_batch_size, is_multimodal=is_multimodal
    )
    micro_batch1, run_reqs1, padded_req_num1 = _padded_prepare_decode_micro_batch(
        req_objs[micro_batch1_req_num:], micro_batch_size, is_multimodal=is_multimodal
    )

    return micro_batch, run_reqs, padded_req_num, micro_batch1, run_reqs1, padded_req_num1


def _padded_prepare_decode_micro_batch(req_objs: List[InferReq], micro_batch_size: int, is_multimodal=False):
    run_reqs = []
    nopad_total_token_num = 0
    nopad_max_len_in_batch = 0
    start_loc = 0
    input_ids = []
    nopad_b_req_idx = []
    nopad_b_start_loc = []
    nopad_b_seq_len = []
    padded_req_num = micro_batch_size - len(req_objs)
    for req in req_objs:
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

    # padding fake req for decode
    for _ in range(padded_req_num):
        input_ids.append(1)
        seq_len = 2
        nopad_b_req_idx.append(g_infer_context.req_manager.HOLD_REQUEST_ID)
        nopad_b_start_loc.append(start_loc)
        nopad_b_seq_len.append(seq_len)
        start_loc += seq_len
        nopad_total_token_num += seq_len
        nopad_max_len_in_batch = max(nopad_max_len_in_batch, seq_len)

    input_ids = torch.tensor(input_ids, dtype=torch.int64, device="cuda")
    nopad_b_req_idx = torch.tensor(nopad_b_req_idx, dtype=torch.int32, device="cuda")
    nopad_b_start_loc = torch.tensor(nopad_b_start_loc, dtype=torch.int32, device="cuda")
    nopad_b_seq_len = torch.tensor(nopad_b_seq_len, dtype=torch.int32, device="cuda")

    # dynamic prompt cache 准备 token
    g_infer_state_lock.acquire()
    if g_infer_context.radix_cache is not None:
        g_infer_context.radix_cache.free_radix_cache_to_get_enough_token(input_ids.shape[0] - padded_req_num)
    mem_indexes = g_infer_context.req_manager.mem_manager.alloc(input_ids.shape[0] - padded_req_num).cuda()
    g_infer_state_lock.release()

    if padded_req_num > 0:
        padding_indexs = torch.full(
            (padded_req_num,),
            fill_value=g_infer_context.req_manager.mem_manager.HOLD_TOKEN_MEMINDEX,
            dtype=torch.int32,
            device="cuda",
        )
        mem_indexes = torch.cat((mem_indexes, padding_indexs), dim=0)

    micro_batch = DecodeMicroBatch(
        batch_size=nopad_b_seq_len.shape[0],
        total_token_num=nopad_total_token_num,
        max_len_in_batch=nopad_max_len_in_batch,
        input_ids=input_ids,
        mem_indexes=mem_indexes,
        b_req_idx=nopad_b_req_idx,
        b_start_loc=nopad_b_start_loc,
        b_seq_len=nopad_b_seq_len,
    )

    return micro_batch, run_reqs, padded_req_num


def padded_overlap_prepare_prefill_inputs(req_objs: List[InferReq], max_prefill_num: int, is_multimodal=False):
    assert max_prefill_num != 0
    micro_batch1_req_num = triton.cdiv(len(req_objs), 2)
    micro_batch, run_reqs, padded_req_num = _padded_prepare_prefill_micro_batch(
        req_objs[0:micro_batch1_req_num], is_multimodal=is_multimodal
    )
    micro_batch1, run_reqs1, padded_req_num1 = _padded_prepare_prefill_micro_batch(
        req_objs[micro_batch1_req_num:], is_multimodal=is_multimodal
    )

    return micro_batch, run_reqs, padded_req_num, micro_batch1, run_reqs1, padded_req_num1


def _padded_prepare_prefill_micro_batch(req_objs: List[InferReq], is_multimodal=False):
    run_reqs = []
    nopad_total_token_num = 0
    nopad_max_len_in_batch = 0
    start_loc = 0
    input_ids = []
    nopad_b_req_idx = []
    nopad_b_start_loc = []
    nopad_b_seq_len = []
    # prefill 只需要 padding 一个请求形成 micro_batch, 并不需要两个
    # micro batch 的 batch_size 相同。
    padded_req_num = 1 if len(req_objs) == 0 else 0
    b_ready_cache_len = []
    batch_multimodal_params = []
    for req in req_objs:
        run_reqs.append(req)
        batch_multimodal_params.append(req.multimodal_params)
        nopad_b_req_idx.append(req.req_idx)
        nopad_b_start_loc.append(start_loc)

        input_token_ids = req.get_chuncked_input_token_ids()
        seq_len = len(input_token_ids)
        input_token_len = seq_len - req.cur_kv_len
        input_id = input_token_ids[req.cur_kv_len :]

        nopad_b_seq_len.append(seq_len)
        input_ids.extend(input_id)
        nopad_total_token_num += seq_len
        nopad_max_len_in_batch = max(nopad_max_len_in_batch, input_token_len)
        b_ready_cache_len.append(req.cur_kv_len)
        start_loc += input_token_len

    # padding fake req for decode
    for _ in range(padded_req_num):
        input_ids.append(1)
        nopad_b_req_idx.append(g_infer_context.req_manager.HOLD_REQUEST_ID)
        nopad_b_start_loc.append(start_loc)
        start_loc += 1
        nopad_b_seq_len.append(1)
        b_ready_cache_len.append(0)
        nopad_total_token_num += 1
        nopad_max_len_in_batch = max(nopad_max_len_in_batch, 1)

    input_ids = torch.tensor(input_ids, dtype=torch.int64, device="cuda")
    nopad_b_req_idx = torch.tensor(nopad_b_req_idx, dtype=torch.int32, device="cuda")
    nopad_b_start_loc = torch.tensor(nopad_b_start_loc, dtype=torch.int32, device="cuda")
    nopad_b_seq_len = torch.tensor(nopad_b_seq_len, dtype=torch.int32, device="cuda")
    b_ready_cache_len = torch.tensor(b_ready_cache_len, dtype=torch.int32, device="cuda")

    # dynamic prompt cache 准备 token
    g_infer_state_lock.acquire()
    if g_infer_context.radix_cache is not None:
        g_infer_context.radix_cache.free_radix_cache_to_get_enough_token(input_ids.shape[0] - padded_req_num)
    mem_indexes = g_infer_context.req_manager.mem_manager.alloc(input_ids.shape[0] - padded_req_num).cuda()
    g_infer_state_lock.release()
    if padded_req_num > 0:
        padding_indexs = torch.full(
            (padded_req_num,),
            fill_value=g_infer_context.req_manager.mem_manager.HOLD_TOKEN_MEMINDEX,
            dtype=torch.int32,
            device="cuda",
        )
        mem_indexes = torch.cat((mem_indexes, padding_indexs), dim=0)

    micro_batch = PrefillMicroBatch(
        batch_size=nopad_b_seq_len.shape[0],
        total_token_num=nopad_total_token_num,
        max_len_in_batch=nopad_max_len_in_batch,
        input_ids=input_ids,
        mem_indexes=mem_indexes,
        b_req_idx=nopad_b_req_idx,
        b_start_loc=nopad_b_start_loc,
        b_seq_len=nopad_b_seq_len,
        b_ready_cache_len=b_ready_cache_len,
    )

    return micro_batch, run_reqs, padded_req_num
