import torch
import numpy as np
from lightllm.server.router.model_infer.infer_batch import requests_mapping, InferReq, InferBatch
from lightllm.server.io_struct import ReqRunStatus
from lightllm.utils.infer_utils import calculate_time
from lightllm.server.router.dynamic_prompt.radix_cache import RadixCache
from lightllm.common.mem_manager import MemoryManager

# @calculate_time(show=True, min_cost_ms=1)
def splitfuse_prepare_decode_inputs(batch: InferBatch, splitfuse_block_size, radix_cache: RadixCache):
    decode_reqs, prefill_reqs = [], []
    for request_id in batch.request_ids:
        req: InferReq = requests_mapping[request_id]
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
    prefill_b_split_ready_cache_len = []
    prefill_max_split_seq_len_in_batch = 0
    prefill_b_seq_len = []

    for req in prefill_reqs:
        prefill_b_req_idx.append(req.req_idx)
        split_len = min(len(req.input_token_ids) - req.cur_kv_len, splitfuse_block_size)
        prefill_b_split_start_loc.append(split_start_loc)
        split_start_loc += split_len
        prefill_b_split_ready_cache_len.append(req.cur_kv_len)
        prefill_max_split_seq_len_in_batch = max(prefill_max_split_seq_len_in_batch, split_len)
        seq_len = req.cur_kv_len + split_len
        prefill_b_seq_len.append(seq_len)
        input_ids.extend(req.input_token_ids[seq_len - split_len : seq_len])

    input_ids = torch.tensor(input_ids, dtype=torch.int64, device="cuda")
    kwargs = {
        "input_ids": input_ids,
        "decode_req_num": decode_req_num,
        "decode_total_token_num": decode_total_token_num,
        "decode_b_req_idx": torch.tensor(decode_b_req_idx, dtype=torch.int32, device="cuda"),
        "decode_b_start_loc": torch.tensor(decode_b_start_loc, dtype=torch.int32, device="cuda"),
        "decode_b_seq_len": torch.tensor(decode_b_seq_len, dtype=torch.int32, device="cuda"),
        "decode_max_len_in_batch": decode_max_len_in_batch,
        "prefill_req_num": prefill_req_num,
        "prefill_b_req_idx": torch.tensor(prefill_b_req_idx, dtype=torch.int32, device="cuda"),
        "prefill_b_split_start_loc": torch.tensor(prefill_b_split_start_loc, dtype=torch.int32, device="cuda"),
        "prefill_b_split_ready_cache_len": torch.tensor(
            prefill_b_split_ready_cache_len, dtype=torch.int32, device="cuda"
        ),
        "prefill_max_split_seq_len_in_batch": prefill_max_split_seq_len_in_batch,
        "prefill_b_seq_len": torch.tensor(prefill_b_seq_len, dtype=torch.int32, device="cuda"),
    }

    # dynamic prompt cache 准备 token
    if radix_cache is not None:
        radix_cache.free_radix_cache_to_get_enough_token(input_ids.shape[0])

    return kwargs, decode_reqs, prefill_reqs
