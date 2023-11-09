from .sampling_params import SamplingParams
from typing import Dict, List, Optional, Tuple
import asyncio
import enum

class ReqRunStatus(enum.Enum):
    WAIT_IN_QUEUE = 0 # 在队列中等待
    RUNNING = 1 # 运行
    PAUSED_AND_KVKEEP = 2 # 暂停保留KV
    PAUSED_AND_OFFLOAD = 3 # 暂停卸载KV
    RERUNNING_FROM_KVKEEP = 4 # 从暂停中恢复
    RERUNNING_FROM_OFFLOAD = 5 # 从卸载KV中恢复


class Req:
    def __init__(self, request_id, prompt_ids, sample_params: SamplingParams):
        self.request_id = request_id
        self.prompt_ids = prompt_ids
        self.input_len = len(prompt_ids)
        self.max_output_len = sample_params.max_new_tokens
        self.sample_params = sample_params
        self.output_ids = []
        self.output_metadata_list = []
        self.has_generate_finished = False
        self.aborted = False
        self.req_status = ReqRunStatus.WAIT_IN_QUEUE
        self.offload_kv_len = None # 卸载的kv长度

    def to_rpc_obj(self):
        return {"request_id": self.request_id,
                "input_id": self.prompt_ids,
                "output_len": self.max_output_len,
                "sampling_param": self.sample_params.to_dict() }

    def to_req_detokenization_state(self):
        out = ReqDetokenizationState(self.request_id, self.prompt_ids, self.max_output_len, self.sample_params.ignore_eos)
        if self.output_metadata_list:
            out.gen_metadata.update(self.output_metadata_list[-1])
        return out
    
    def stop_sequences_matched(self):
        for stop_token_ids in self.sample_params.stop_sequences:
            stop_len = len(stop_token_ids)
            if stop_len > 0:
                if len(self.output_ids) >= stop_len:
                    if all(self.output_ids[-(stop_len - i)] == stop_token_ids[i] for i in range(stop_len)):
                        return True
        return False

    def __repr__(self):
        return (f"request_id(n={self.request_id}, "
                f"prompt_ids={self.prompt_ids}, ")
    
    def get_tuple_tokens(self, is_busy, router_max_new_token_len):
        if self.sample_params.ignore_eos:
            cur_max_new_token_len = self.max_output_len
        elif is_busy:
            cur_max_new_token_len = self.max_output_len
        else:
            cur_max_new_token_len = min(self.max_output_len, router_max_new_token_len) 
        
        if self.req_status == ReqRunStatus.RUNNING:
            return (self.input_len + len(self.output_ids), max(0, cur_max_new_token_len - len(self.output_ids) - 1))
        elif self.req_status == ReqRunStatus.WAIT_IN_QUEUE:
            return (self.input_len + 1,  max(0, cur_max_new_token_len - 1 - 1))
        elif self.req_status == ReqRunStatus.PAUSED_AND_KVKEEP:
            return (self.input_len + len(self.output_ids), max(0, cur_max_new_token_len - len(self.output_ids) - 1))
        elif self.req_status == ReqRunStatus.PAUSED_AND_OFFLOAD:
            return (self.input_len + len(self.output_ids), max(0, cur_max_new_token_len - len(self.output_ids) - 1))
        else:
            assert False, "error state"
    
    def get_prefill_need_tokens(self):
        if self.req_status == ReqRunStatus.WAIT_IN_QUEUE:
            return self.input_len
        elif self.req_status == ReqRunStatus.RUNNING:
            return 0
        elif self.req_status == ReqRunStatus.PAUSED_AND_KVKEEP:
            return 0
        elif self.req_status == ReqRunStatus.PAUSED_AND_OFFLOAD:
            return self.offload_kv_len
        else:
            assert False, "error state"

    def get_used_tokens(self):
        if self.req_status == ReqRunStatus.RUNNING:
            return self.input_len + len(self.output_ids) - 1
        if self.req_status == ReqRunStatus.PAUSED_AND_OFFLOAD:
            return self.input_len + len(self.output_ids) - 1 - self.offload_kv_len
        elif self.req_status == ReqRunStatus.PAUSED_AND_KVKEEP:
            return self.input_len + len(self.output_ids) - 1
        elif self.req_status == ReqRunStatus.WAIT_IN_QUEUE:
            return 0
        else:
            assert False, "error state"

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
        self.gen_metadata = {}

class Batch:
    def __init__(self, batch_id, reqs: List[Req]):
        self.batch_id = batch_id
        self.reqs = reqs
        self.id_to_reqs = {req.request_id: req for req in reqs}
        self.batch_used_tokens = 0
        self.recalcu_batch_used_tokens()

    def input_tokens(self):
        batch_input_tokens = 0
        for req in self.reqs:
            batch_input_tokens += req.input_len
        return batch_input_tokens

    def calcu_max_tokens(self):
        tokens = 0
        for req in self.reqs:
            tokens += req.input_len + req.max_output_len
        return tokens

    def mark_and_get_finished_req(self, eos_id):
        finished_reqs, unfinished_req = [], []
        for req in self.reqs:
            if req.stop_sequences_matched():
                req.has_generate_finished = True
            if req.output_ids[-1] == eos_id and req.sample_params.ignore_eos == False:
                req.has_generate_finished = True
            if len(req.output_ids) >= req.max_output_len or req.aborted:
                req.has_generate_finished = True

            if req.has_generate_finished:
                finished_reqs.append(req)
            else:
                unfinished_req.append(req)
    
        return finished_reqs, unfinished_req
    
    def filter_out_finished_req(self, finished_reqs, unfinished_req):
        # update batch
        if len(finished_reqs) != 0:
            self.reqs = unfinished_req
            self.id_to_reqs = {req.request_id: req for req in unfinished_req}
        return
    
    def pop_req(self, req_id):
        self.reqs = [req for req in self.reqs if req.request_id != req_id]
        req = self.id_to_reqs[req_id]
        self.id_to_reqs.pop(req_id)
        self.batch_used_tokens -= req.get_used_tokens()
        return

    def is_clear(self):
        return len(self.reqs) == 0

    def merge(self, mini_batch):
        for _req in mini_batch.reqs:
            self.reqs.append(_req)
        self.id_to_reqs = {req.request_id: req for req in self.reqs}
        self.batch_used_tokens += mini_batch.batch_used_tokens
        return
    
    def recalcu_batch_used_tokens(self):
        total_tokens = 0
        for req in self.reqs:
            total_tokens += req.get_used_tokens()
        self.batch_used_tokens = total_tokens
        return self.batch_used_tokens
    
    def update_req_status_to_running(self):
        """
        当prefill完之后, 将所有请求的状态修改为 RUNNING, 这个函数只会在prefill之后进行调用,
        修改状态后，再调用 recalcu_batch_used_tokens 才能得到正常的值
        """
        for req in self.reqs:
            req.req_status = ReqRunStatus.RUNNING
            req.offload_kv_len = 0
        self.recalcu_batch_used_tokens()
        return
    
    def __repr__(self):
        return (f"batch_id={self.batch_id}, "
                f"reqs={self.reqs}, ")
        
class BatchTokenIdOut:
    def __init__(self):
        self.reqs_infs: List[Tuple[str, int, Dict, bool, bool]] = []  # [req_id, new_token_id, gen_metadata, finished_state, abort_state]

class BatchStrOut:
    def __init__(self):
        self.reqs_infs: List[Tuple[str, str, Dict, bool, bool]] = [] # [req_id, token_str, gen_metadata, finished_state, abort_state]
        
class AbortReq:
    def __init__(self, req_id):
        self.req_id = req_id
        
