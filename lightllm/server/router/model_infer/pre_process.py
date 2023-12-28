import torch
import numpy as np
from .infer_batch import requests_mapping, InferReq, InferBatch
from lightllm.server.io_struct import ReqRunStatus
from lightllm.utils.infer_utils import calculate_time

#@calculate_time(show=True, min_cost_ms=1)
def prepare_prefill_inputs(batch:InferBatch, is_multimodal=False):
    run_reqs, not_run_reqs = [], []
    nopad_total_token_num = 0
    nopad_max_len_in_batch = 0
    start_loc = 0
    input_ids = []
    nopad_b_req_idx = []
    nopad_b_start_loc = []
    nopad_b_seq_len = []
    batch_multimodal_params = []
    for request_id in batch.request_ids:
        req : InferReq = requests_mapping[request_id]
        assert req.req_status == ReqRunStatus.RUNNING
        # 当请求已经存在 cur_kv_len 不为 0 的时候，就不需要做全 prefill 操作了，
        # 说明是从 RERUNNING_FROM_KVKEEP 中 恢复的请求
        if req.cur_kv_len != 0: 
            not_run_reqs.append(req)
            continue
        
        run_reqs.append(req)
        batch_multimodal_params.append(req.multimodal_params)
        nopad_b_req_idx.append(req.req_idx)
        nopad_b_start_loc.append(start_loc)
        
        seq_len = len(req.input_token_ids)
        input_id = req.input_token_ids
        
        nopad_b_seq_len.append(seq_len)
        input_ids.append(input_id)
        nopad_total_token_num += seq_len
        nopad_max_len_in_batch = max(nopad_max_len_in_batch, seq_len)
        start_loc += seq_len
    
    if len(run_reqs) >= 1:
        
        input_ids = np.concatenate(input_ids, dtype=np.int64)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, device='cuda')
        nopad_b_req_idx = torch.tensor(nopad_b_req_idx, dtype=torch.int32, device='cuda')
        nopad_b_start_loc = torch.tensor(nopad_b_start_loc, dtype=torch.int32, device='cuda')
        nopad_b_seq_len = torch.tensor(nopad_b_seq_len, dtype=torch.int32, device='cuda')
        kwargs = {
            "batch_size": len(batch),
            "total_token_num": nopad_total_token_num,
            "max_len_in_batch": nopad_max_len_in_batch,
            "input_ids": input_ids,
            "b_req_idx": nopad_b_req_idx,
            "b_start_loc": nopad_b_start_loc,
            "b_seq_len": nopad_b_seq_len,
            "is_prefill": True,
        }
        if is_multimodal:
            kwargs["multimodal_params"] = batch_multimodal_params
        return kwargs, run_reqs, not_run_reqs
    else:
        return {}, run_reqs, not_run_reqs
    
#@calculate_time(show=True, min_cost_ms=1)
def prepare_decode_inputs(batch:InferBatch):
    run_reqs, not_run_reqs = [], []
    nopad_total_token_num = 0
    nopad_max_len_in_batch = 0
    start_loc = 0
    input_ids = []
    nopad_b_req_idx = []
    nopad_b_start_loc = []
    nopad_b_seq_len = []
    for request_id in batch.request_ids:
        req : InferReq = requests_mapping[request_id]
        assert req.req_status == ReqRunStatus.RUNNING
        run_reqs.append(req)
        nopad_b_req_idx.append(req.req_idx)
        nopad_b_start_loc.append(start_loc)
        input_id = req.input_token_ids[-1]
        seq_len = len(req.input_token_ids)
        assert req.cur_kv_len == seq_len - 1
        nopad_b_seq_len.append(seq_len)
        input_ids.append(input_id)
        nopad_total_token_num += seq_len
        nopad_max_len_in_batch = max(nopad_max_len_in_batch, seq_len)
        start_loc += seq_len
    
    if len(run_reqs) >= 1:

        input_ids = torch.tensor(input_ids, dtype=torch.int64, device='cuda')
        nopad_b_req_idx = torch.tensor(nopad_b_req_idx, dtype=torch.int32, device='cuda')
        nopad_b_start_loc = torch.tensor(nopad_b_start_loc, dtype=torch.int32, device='cuda')
        nopad_b_seq_len = torch.tensor(nopad_b_seq_len, dtype=torch.int32, device='cuda')
        kwargs = {
            "batch_size": len(batch),
            "total_token_num": nopad_total_token_num,
            "max_len_in_batch": nopad_max_len_in_batch,
            "input_ids": input_ids,
            "b_req_idx": nopad_b_req_idx,
            "b_start_loc": nopad_b_start_loc,
            "b_seq_len": nopad_b_seq_len,
            "is_prefill": False            
        }
        return kwargs, run_reqs, not_run_reqs
    else:
        return {}, run_reqs, not_run_reqs

#@calculate_time(show=True, min_cost_ms=1)
def splitfuse_prepare_decode_inputs(batch:InferBatch, splitfuse_block_size):    
    decode_reqs, prefill_reqs = [], []
    for request_id in batch.request_ids:
        req : InferReq = requests_mapping[request_id]
        if req.cur_kv_len == len(req.input_token_ids) - 1:
            decode_reqs.append(req)
        elif req.cur_kv_len < len(req.input_token_ids) - 1:
            prefill_reqs.append(req)
        else:
            assert False, "error state"

    input_ids = []
    decode_req_num = len(decode_reqs)
    decode_total_token_num = 0
    decode_b_req_idx = []
    decode_b_start_loc = []
    decode_b_seq_len = []
    decode_max_len_in_batch = 0
    start_loc = 0

    for req in decode_reqs:
        seq_len = len(req.input_token_ids)
        decode_b_start_loc.append(start_loc)
        start_loc += seq_len
        decode_total_token_num += seq_len
        decode_b_req_idx.append(req.req_idx)
        decode_b_seq_len.append(seq_len)
        decode_max_len_in_batch = max(decode_max_len_in_batch, seq_len)
        input_ids.append(req.input_token_ids[-1])
    
    prefill_req_num = len(prefill_reqs)
    prefill_b_req_idx = []
    prefill_b_split_start_loc = []
    split_start_loc = 0
    prefill_b_split_seq_len = []
    prefill_max_split_seq_len_in_batch = 0
    prefill_b_seq_len = []

    for req in prefill_reqs:
        prefill_b_req_idx.append(req.req_idx)
        split_len = min(len(req.input_token_ids) - req.cur_kv_len, splitfuse_block_size)
        prefill_b_split_start_loc.append(split_start_loc)
        split_start_loc += split_len
        prefill_b_split_seq_len.append(split_len)
        prefill_max_split_seq_len_in_batch = max(prefill_max_split_seq_len_in_batch, split_len)
        seq_len = req.cur_kv_len + split_len
        prefill_b_seq_len.append(seq_len)
        input_ids.extend(req.input_token_ids[seq_len - split_len : seq_len])
    
    kwargs = {
            "input_ids": torch.tensor(input_ids, dtype=torch.int64, device='cuda'),
            "decode_req_num": decode_req_num,
            "decode_total_token_num": decode_total_token_num,
            "decode_b_req_idx": torch.tensor(decode_b_req_idx, dtype=torch.int32, device='cuda'),
            "decode_b_start_loc": torch.tensor(decode_b_start_loc, dtype=torch.int32, device="cuda"),
            "decode_b_seq_len": torch.tensor(decode_b_seq_len, dtype=torch.int32, device="cuda"),
            "decode_max_len_in_batch": decode_max_len_in_batch,

            "prefill_req_num": prefill_req_num,
            "prefill_b_req_idx": torch.tensor(prefill_b_req_idx, dtype=torch.int32, device="cuda"),
            "prefill_b_split_start_loc" : torch.tensor(prefill_b_split_start_loc, dtype=torch.int32, device="cuda"),
            "prefill_b_split_seq_len" : torch.tensor(prefill_b_split_seq_len, dtype=torch.int32, device="cuda"),
            "prefill_max_split_seq_len_in_batch" : prefill_max_split_seq_len_in_batch,
            "prefill_b_seq_len" : torch.tensor(prefill_b_seq_len, dtype=torch.int32, device="cuda")
        }
    return kwargs, decode_reqs, prefill_reqs
    
