import torch
import numpy as np
from typing import List, Tuple
from lightllm.server.router.model_infer.infer_batch import InferReq, g_infer_context
from lightllm.common.basemodel.infer_lock import g_infer_state_lock
from lightllm.common.basemodel.batch_objs import ModelInput


def prepare_prefill_inputs(
    req_objs: List[InferReq], is_chuncked_mode: bool, is_multimodal: bool = False
) -> Tuple[ModelInput, List[InferReq]]:
    run_reqs = []
    total_token_num = 0
    max_len_in_batch = 0
    input_ids = []
    b_req_idx = []
    b_seq_len = []
    batch_multimodal_params = []
    b_ready_cache_len = []
    for req in req_objs:
        run_reqs.append(req)
        batch_multimodal_params.append(req.multimodal_params)
        b_req_idx.append(req.req_idx)

        if is_chuncked_mode:
            input_token_ids = req.get_chuncked_input_token_ids()
        else:
            input_token_ids = req.get_input_token_ids()

        seq_len = len(input_token_ids)
        input_token_len = seq_len - req.cur_kv_len

        input_id = input_token_ids[req.cur_kv_len :]

        b_seq_len.append(seq_len)
        input_ids.append(input_id)
        total_token_num += seq_len
        max_len_in_batch = max(max_len_in_batch, input_token_len)
        b_ready_cache_len.append(req.cur_kv_len)

    input_ids = np.concatenate(input_ids, dtype=np.int64)

    input_ids = torch.tensor(input_ids, dtype=torch.int64, device="cuda")
    b_req_idx = torch.tensor(b_req_idx, dtype=torch.int32, device="cuda")
    b_seq_len = torch.tensor(b_seq_len, dtype=torch.int32, device="cuda")
    b_ready_cache_len = torch.tensor(b_ready_cache_len, dtype=torch.int32, device="cuda")

    # dynamic prompt cache 准备 token
    g_infer_state_lock.acquire()
    if g_infer_context.radix_cache is not None:
        g_infer_context.radix_cache.free_radix_cache_to_get_enough_token(input_ids.shape[0])
    mem_indexes = g_infer_context.req_manager.mem_manager.alloc(input_ids.shape[0]).cuda()
    g_infer_state_lock.release()

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

    return model_input, run_reqs


def prepare_decode_inputs(req_objs: List[InferReq]) -> Tuple[ModelInput, List[InferReq]]:
    run_reqs = []
    total_token_num = 0
    max_len_in_batch = 0
    input_ids = []
    b_req_idx = []
    b_seq_len = []
    for req in req_objs:
        run_reqs.append(req)
        b_req_idx.append(req.req_idx)
        input_id = req.get_last_gen_token()
        seq_len = req.get_cur_total_len()
        assert req.cur_kv_len == seq_len - 1, f"{req.cur_kv_len} {seq_len}"
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

    input_ids = torch.tensor(input_ids, dtype=torch.int64, device="cuda")
    b_req_idx = torch.tensor(b_req_idx, dtype=torch.int32, device="cuda")
    b_seq_len = torch.tensor(b_seq_len, dtype=torch.int32, device="cuda")

    # dynamic prompt cache 准备 token
    g_infer_state_lock.acquire()
    if g_infer_context.radix_cache is not None:
        g_infer_context.radix_cache.free_radix_cache_to_get_enough_token(input_ids.shape[0])
    mem_indexes = g_infer_context.req_manager.mem_manager.alloc(input_ids.shape[0]).cuda()
    g_infer_state_lock.release()

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
    return model_input, run_reqs
