import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
import triton
from typing import List, Optional, Tuple
from lightllm.server.router.model_infer.infer_batch import g_infer_context, InferReq
from lightllm.utils.infer_utils import calculate_time
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.common.basemodel.infer_lock import g_infer_state_lock
from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput


def padded_prepare_prefill_inputs(
    req_objs: List[InferReq], dest_batch_size: Optional[int] = None, is_multimodal=False
) -> Tuple[ModelInput, List[InferReq], int]:

    if dest_batch_size is None:
        req_num = len(req_objs)
        if req_num > 0:
            dest_batch_size = req_num
        else:
            dest_batch_size = 1
    else:
        assert len(req_objs) <= dest_batch_size

    run_reqs = []
    total_token_num = 0
    max_len_in_batch = 0
    padded_req_num = dest_batch_size - len(req_objs)
    input_ids = []
    b_req_idx = []
    b_seq_len = []
    batch_multimodal_params = []
    b_ready_cache_len = []
    for req in req_objs:

        run_reqs.append(req)
        batch_multimodal_params.append(req.multimodal_params)
        b_req_idx.append(req.req_idx)

        input_token_ids = req.get_chuncked_input_token_ids()
        seq_len = len(input_token_ids)
        input_token_len = seq_len - req.cur_kv_len
        input_id = input_token_ids[req.cur_kv_len :]

        b_seq_len.append(seq_len)
        input_ids.append(input_id)
        total_token_num += seq_len
        max_len_in_batch = max(max_len_in_batch, input_token_len)
        b_ready_cache_len.append(req.cur_kv_len)

    # padding fake req for prefill
    for _ in range(padded_req_num):
        input_ids.append([1])
        b_req_idx.append(g_infer_context.req_manager.HOLD_REQUEST_ID)
        b_seq_len.append(1)
        b_ready_cache_len.append(0)
        total_token_num += 1
        max_len_in_batch = max(max_len_in_batch, 1)

    input_ids = np.concatenate(input_ids, dtype=np.int64)
    input_ids = torch.tensor(input_ids, dtype=torch.int64, device="cuda")
    b_req_idx = torch.tensor(b_req_idx, dtype=torch.int32, device="cuda")
    b_seq_len = torch.tensor(b_seq_len, dtype=torch.int32, device="cuda")
    b_ready_cache_len = torch.tensor(b_ready_cache_len, dtype=torch.int32, device="cuda")

    # dynamic prompt cache 准备 token
    g_infer_state_lock.acquire()
    if g_infer_context.radix_cache is not None:
        g_infer_context.radix_cache.free_radix_cache_to_get_enough_token(input_ids.shape[0] - padded_req_num)
    mem_indexes = g_infer_context.req_manager.mem_manager.alloc(input_ids.shape[0] - padded_req_num).cuda()
    g_infer_state_lock.release()

    if padded_req_num > 0:
        mem_indexes = F.pad(
            input=mem_indexes,
            pad=(0, padded_req_num),
            mode="constant",
            value=g_infer_context.req_manager.mem_manager.HOLD_TOKEN_MEMINDEX,
        )

    model_input = ModelInput(
        batch_size=b_seq_len.shape[0],
        total_token_num=total_token_num,
        max_len_in_batch=max_len_in_batch,
        input_ids=input_ids,
        mem_indexes=mem_indexes,
        b_req_idx=b_req_idx,
        b_seq_len=b_seq_len,
        b_ready_cache_len=b_ready_cache_len,
        is_prefill=True,
    )
    if is_multimodal:
        model_input.multimodal_params = batch_multimodal_params

    return model_input, run_reqs, padded_req_num


def padded_prepare_decode_inputs(
    req_objs: List[InferReq], dest_batch_size: Optional[int] = None, is_multimodal=False
) -> Tuple[ModelInput, List[InferReq], int]:
    run_reqs = []
    total_token_num = 0
    max_len_in_batch = 0
    input_ids = []
    b_req_idx = []
    b_seq_len = []

    for req in req_objs:
        run_reqs.append(req)
        b_req_idx.append(req.req_idx)
        input_token_ids = req.get_input_token_ids()
        input_id = input_token_ids[-1]
        seq_len = len(input_token_ids)
        assert req.cur_kv_len == seq_len - 1
        b_seq_len.append(seq_len)
        input_ids.append(input_id)
        total_token_num += seq_len
        max_len_in_batch = max(max_len_in_batch, seq_len)
        # process the draft tokens.
        for step in range(len(req.mtp_gen_token_ids)):
            run_reqs.append(req)
            b_req_idx.append(req.req_idx)
            seq_len += 1
            b_seq_len.append(seq_len)
            input_ids.append(req.mtp_gen_token_ids[step])
            total_token_num += seq_len
            max_len_in_batch = max(max_len_in_batch, seq_len)

    if dest_batch_size is None:
        if len(run_reqs) == 0:
            dest_batch_size = 1
        else:
            dest_batch_size = len(run_reqs)
    else:
        assert len(run_reqs) <= dest_batch_size

    padded_req_num = dest_batch_size - len(run_reqs)

    # padding fake req for decode
    for _ in range(padded_req_num):
        input_ids.append(1)
        seq_len = 2
        b_req_idx.append(g_infer_context.req_manager.HOLD_REQUEST_ID)
        b_seq_len.append(seq_len)
        total_token_num += seq_len
        max_len_in_batch = max(max_len_in_batch, seq_len)

    input_ids = torch.tensor(input_ids, dtype=torch.int64, device="cuda")
    b_req_idx = torch.tensor(b_req_idx, dtype=torch.int32, device="cuda")
    b_seq_len = torch.tensor(b_seq_len, dtype=torch.int32, device="cuda")

    # dynamic prompt cache 准备 token
    g_infer_state_lock.acquire()
    if g_infer_context.radix_cache is not None:
        g_infer_context.radix_cache.free_radix_cache_to_get_enough_token(input_ids.shape[0] - padded_req_num)
    mem_indexes = g_infer_context.req_manager.mem_manager.alloc(input_ids.shape[0] - padded_req_num).cuda()
    g_infer_state_lock.release()

    if padded_req_num > 0:
        mem_indexes = F.pad(
            input=mem_indexes,
            pad=(0, padded_req_num),
            mode="constant",
            value=g_infer_context.req_manager.mem_manager.HOLD_TOKEN_MEMINDEX,
        )

    model_input = ModelInput(
        batch_size=b_seq_len.shape[0],
        total_token_num=total_token_num,
        max_len_in_batch=max_len_in_batch,
        input_ids=input_ids,
        mem_indexes=mem_indexes,
        b_req_idx=b_req_idx,
        b_seq_len=b_seq_len,
        is_prefill=False,
    )
    return model_input, run_reqs, padded_req_num


def padded_overlap_prepare_decode_inputs(req_objs: List[InferReq], is_multimodal=False):
    split_req_bound = triton.cdiv(len(req_objs), 2)
    req_objs_0 = req_objs[0:split_req_bound]
    req_objs_1 = req_objs[split_req_bound:]

    enable_mtp = get_env_start_args().mtp_mode is not None
    if enable_mtp:
        micro_batch_size = max(
            sum([len(req.mtp_gen_token_ids) + 1 for req in req_objs_0]),
            sum([len(req.mtp_gen_token_ids) + 1 for req in req_objs_1]),
        )
    else:
        micro_batch_size = triton.cdiv(len(req_objs), 2)

    micro_batch_size = max(1, micro_batch_size)

    micro_input, run_reqs, padded_req_num = padded_prepare_decode_inputs(
        req_objs_0, dest_batch_size=micro_batch_size, is_multimodal=is_multimodal
    )
    micro_input1, run_reqs1, padded_req_num1 = padded_prepare_decode_inputs(
        req_objs_1, dest_batch_size=micro_batch_size, is_multimodal=is_multimodal
    )
    return micro_input, run_reqs, padded_req_num, micro_input1, run_reqs1, padded_req_num1


def padded_overlap_prepare_prefill_inputs(req_objs: List[InferReq], is_multimodal=False):
    micro_batch1_req_num = triton.cdiv(len(req_objs), 2)

    micro_input, run_reqs, padded_req_num = padded_prepare_prefill_inputs(
        req_objs[0:micro_batch1_req_num], is_multimodal=is_multimodal
    )

    micro_input1, run_reqs1, padded_req_num1 = padded_prepare_prefill_inputs(
        req_objs[micro_batch1_req_num:], is_multimodal=is_multimodal
    )

    return micro_input, run_reqs, padded_req_num, micro_input1, run_reqs1, padded_req_num1
