from .sampling_params import SamplingParams
from typing import Dict, List, Optional, Tuple
import asyncio


class Req:
    def __init__(self, request_id, prompt_ids, sample_params: SamplingParams):
        self.request_id = request_id
        self.prompt_ids = prompt_ids
        self.input_len = len(prompt_ids)
        self.max_output_len = sample_params.max_new_tokens
        self.sample_params = sample_params
        self.output_ids = []
        self.has_generate_finished = False
        self.aborted = False

    def to_rpc_obj(self):
        return {"request_id": self.request_id,
                "input_id": self.prompt_ids,
                "output_len": self.max_output_len,
                "sampling_param": self.sample_params.to_dict() }

    def to_req_detokenization_state(self):
        out = ReqDetokenizationState(self.request_id, self.prompt_ids, self.max_output_len, self.sample_params.ignore_eos)
        return out

    def __repr__(self):
        return (f"request_id(n={self.request_id}, "
                f"prompt_ids={self.prompt_ids}, ")
        

class ReqDetokenizationState:
    def __init__(
        self,
        request_id: str,
        prompt_ids: List[int],
        max_output_len: int,
        ignore_eos: bool,
    ) -> None:
        self.request_id = request_id
        self.prompt_ids = prompt_ids
        self.output_ids = []
        self.output_tokens = []
        self.output_str = ""
        self.sub_texts = []
        self.current_sub_text = []
        self.max_output_len = max_output_len
        self.ignore_eos = ignore_eos

class Batch:
    def __init__(self, batch_id, reqs: List[Req]):
        self.batch_id = batch_id
        self.reqs = reqs
        self.id_to_reqs = {req.request_id: req for req in reqs}

    def calcu_max_tokens(self):
        tokens = 0
        for req in self.reqs:
            tokens += req.input_len + req.max_output_len
        return tokens
    
    def calcu_used_tokens(self):
        tokens = 0
        for req in self.reqs:
            tokens += req.input_len + len(req.output_ids)
        return tokens

    def mark_finished_req(self, eos_id):
        has_new_finish = False
        for req in self.reqs:
            if req.output_ids[-1] == eos_id and req.sample_params.ignore_eos == False:
                req.has_generate_finished = True
                has_new_finish = True
            if len(req.output_ids) >= req.max_output_len:
                req.has_generate_finished = True
                has_new_finish = True
        return has_new_finish

    def filter_finished(self):
        unfinished_req = []
        for req in self.reqs:
            if not req.has_generate_finished:
                unfinished_req.append(req)
        self.reqs = unfinished_req
        self.id_to_reqs = {req.request_id: req for req in self.reqs}

    def is_clear(self):
        return len(self.reqs) == 0

    def merge(self, mini_batch):
        for _req in mini_batch.reqs:
            self.reqs.append(_req)
        self.id_to_reqs = {req.request_id: req for req in self.reqs}
        return

    def __repr__(self):
        return (f"batch_id={self.batch_id}, "
                f"reqs={self.reqs}, ")
        
class BatchTokenIdOut:
    def __init__(self):
        self.reqs_infs: List[Tuple[str, int, bool, bool]] = []  # [req_id, new_token_id, finished_state, abort_state]

class BatchStrOut:
    def __init__(self):
        self.reqs_infs: List[Tuple[str, int, bool, bool]] = [] # [req_id, token_str, finished_state, abort_state]
        
class AbortReq:
    def __init__(self, req_id):
        self.req_id = req_id
        
