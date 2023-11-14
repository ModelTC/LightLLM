import time
import torch
import numpy as np
import collections

from dataclasses import dataclass, field
from typing import List, Dict
from lightllm.common.req_manager import ReqManager
from lightllm.utils.infer_utils import mark_start, mark_end
from lightllm.server.io_struct import ReqRunStatus


requests_mapping = {}

class InferSamplingParams:

    def __init__(
        self,
        do_sample: bool = False,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        vocab_size: int = -1,
    ) -> None:
        self.do_sample = do_sample
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.repetition_penalty = repetition_penalty
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        if self.top_k == -1:
            self.top_k = vocab_size
        return


class InferReq:

    def __init__(
        self,
        input_token_ids=[],
        out_token_id_count={},
        sampling_param=None,
        req_idx=-1,
        seq_len=0,
        prompt_len=0,
        req_status:ReqRunStatus=ReqRunStatus.RUNNING,
    ) -> None:
        self.out_token_id_count = out_token_id_count
        self.sampling_param = sampling_param
        self.req_idx = req_idx
        self.seq_len = seq_len
        self.prompt_len = prompt_len
        self.offload_kv_len = None
        self.input_token_ids = input_token_ids
        self.req_status = req_status
        return


@dataclass
class InferBatch:
    batch_id: int
    request_ids: List
    req_manager: ReqManager
    
    @classmethod
    @torch.no_grad()
    def init_batch(cls, batch_id, requests, dtype: torch.dtype, device: torch.device, req_manager:ReqManager, vocab_size: int):

        request_ids = []
        need_alloc_size = len([r for r in requests if r['request_id'] not in requests_mapping])
        nopad_b_req_idx = req_manager.alloc(need_alloc_size)
        nopad_b_req_idx = nopad_b_req_idx.cpu().numpy()
        
        index = 0
        for r in requests:
            # request id -> idx in list mapping
            r_id = r['request_id']

            if r_id not in requests_mapping.keys():
                tokenized_input = r['input_id']
                input_length = len(tokenized_input)
                # postprocessor
                sampling_param = r["sampling_param"]
                sampling_param["vocab_size"] = vocab_size
                requests_mapping[r_id] = InferReq(input_token_ids=tokenized_input,
                                                    out_token_id_count=collections.defaultdict(int), 
                                                    sampling_param=InferSamplingParams(**sampling_param), 
                                                    req_idx=nopad_b_req_idx[index], 
                                                    seq_len=input_length,
                                                    prompt_len=input_length,
                                                    req_status=ReqRunStatus.RUNNING)
                index += 1
            else:
                if requests_mapping[r_id].req_status == ReqRunStatus.PAUSED_AND_OFFLOAD:
                    requests_mapping[r_id].req_status = ReqRunStatus.RERUNNING_FROM_OFFLOAD
                elif requests_mapping[r_id].req_status == ReqRunStatus.PAUSED_AND_KVKEEP:
                    requests_mapping[r_id].req_status = ReqRunStatus.RERUNNING_FROM_KVKEEP
                else:
                    assert False, f"should not exist {requests_mapping[r_id].req_status}"
            
            request_ids.append(r_id)
            

        return cls(
            batch_id=batch_id,
            request_ids=request_ids,
            req_manager=req_manager,
        )
    
    @torch.no_grad()
    def free_self(self):
        free_req_index = []
        free_token_index = []
        for request_id in self.request_ids:
            req = requests_mapping.pop(request_id)
            free_req_index.append(req.req_idx)
            free_token_index.append(self.req_manager.req_to_token_indexs[req.req_idx][:req.seq_len - 1])
            
        free_token_index = torch.cat(free_token_index, dim=-1)
        self.req_manager.free(free_req_index, free_token_index)
        if len(requests_mapping) == 0:
            requests_mapping.clear()
        return
    
    @torch.no_grad()
    def filter(self, request_ids: List[str], finished_request_ids: List[str]):
        if len(requests_mapping) == 0:
            raise ValueError("Batch must have at least one request")
        if len(request_ids) == len(self):
            return self
        if len(request_ids) == 0:
            self.free_self()
            return InferBatch(
                batch_id=self.batch_id,
                request_ids=[],
                req_manager=self.req_manager
            )
        free_req_index = []
        free_token_index = []
        for request_id in finished_request_ids:
            req = requests_mapping.pop(request_id)
            free_req_index.append(req.req_idx)
            free_token_index.append(self.req_manager.req_to_token_indexs[req.req_idx][:req.seq_len - 1])
        free_token_index = torch.cat(free_token_index, dim=-1)
        self.req_manager.free(free_req_index, free_token_index)
        
        return InferBatch(
            batch_id=self.batch_id,
            request_ids=request_ids,
            req_manager=self.req_manager,
        )

    @torch.no_grad()
    def pause_reqs(self, pause_reqs: List[str]):
        for request_id, pause_way, offload_kv_len in pause_reqs:
            req = requests_mapping[request_id]
            req.req_status = pause_way
            self.request_ids.remove(request_id)
            if pause_way == ReqRunStatus.PAUSED_AND_OFFLOAD:
                self.req_manager.free_token(self.req_manager.req_to_token_indexs[req.req_idx][:offload_kv_len])
                req.offload_kv_len = offload_kv_len
        return self

    @classmethod
    @torch.no_grad()
    def merge(cls, batch1, batch2):
        request_ids = batch1.request_ids + batch2.request_ids
        
        return InferBatch(
            batch_id=batch1.batch_id,
            request_ids=request_ids,
            req_manager=batch1.req_manager,
        )

    def __len__(self):
        return len(self.request_ids)
    
