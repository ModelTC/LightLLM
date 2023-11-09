import torch
import numpy as np
from .infer_batch import requests_mapping, InferReq, InferBatch
from lightllm.server.io_struct import ReqRunStatus
from lightllm.utils.infer_utils import calculate_time

#@calculate_time(show=True, min_cost_ms=1)
def prepare_prefill_inputs(batch:InferBatch):
    run_req_ids, not_run_req_ids = [], []
    nopad_total_token_num = 0
    nopad_max_len_in_batch = 0
    start_loc = 0
    input_ids = []
    nopad_b_req_idx = []
    nopad_b_start_loc = []
    nopad_b_seq_len = []
    for request_id in batch.request_ids:
        req : InferReq = requests_mapping[request_id]
        assert req.req_status in [ReqRunStatus.RUNNING, ReqRunStatus.RERUNNING_FROM_OFFLOAD, ReqRunStatus.RERUNNING_FROM_KVKEEP]
        if req.req_status == ReqRunStatus.RERUNNING_FROM_KVKEEP:
            # 这个场景下不需要重新进行prefill了，所以不需要运行了
            not_run_req_ids.append(request_id)
            continue
        
        run_req_ids.append(request_id)
        nopad_b_req_idx.append(req.req_idx)
        nopad_b_start_loc.append(start_loc)
        if req.req_status == ReqRunStatus.RERUNNING_FROM_OFFLOAD:
            seq_len = req.offload_kv_len
            input_id = req.input_token_ids[0 : req.offload_kv_len]
        else:
            seq_len = req.seq_len
            input_id = req.input_token_ids
        
        nopad_b_seq_len.append(seq_len)
        input_ids.append(input_id)
        nopad_total_token_num += seq_len
        nopad_max_len_in_batch = max(nopad_max_len_in_batch, seq_len)
        start_loc += seq_len
    
    if len(run_req_ids) >= 1:
        if len(input_ids) > 1:
            input_ids = np.concatenate(input_ids, dtype=np.int64)
        else:
            input_ids = input_ids[0]

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
            "is_prefill": True            
        }
        return kwargs, run_req_ids, not_run_req_ids
    else:
        return {}, run_req_ids, not_run_req_ids
    
#@calculate_time(show=True, min_cost_ms=1)
def prepare_decode_inputs(batch:InferBatch):
    run_req_ids, not_run_req_ids = [], []
    nopad_total_token_num = 0
    nopad_max_len_in_batch = 0
    start_loc = 0
    input_ids = []
    nopad_b_req_idx = []
    nopad_b_start_loc = []
    nopad_b_seq_len = []
    for request_id in batch.request_ids:
        req = requests_mapping[request_id]
        assert req.req_status == ReqRunStatus.RUNNING
        run_req_ids.append(request_id)
        nopad_b_req_idx.append(req.req_idx)
        nopad_b_start_loc.append(start_loc)
        seq_len = req.seq_len
        input_id = req.input_token_ids[-1]
        
        nopad_b_seq_len.append(seq_len)
        input_ids.append(input_id)
        nopad_total_token_num += seq_len
        nopad_max_len_in_batch = max(nopad_max_len_in_batch, seq_len)
        start_loc += seq_len
    
    if len(run_req_ids) >= 1:

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
        return kwargs, run_req_ids, not_run_req_ids
    else:
        return {}, run_req_ids, not_run_req_ids