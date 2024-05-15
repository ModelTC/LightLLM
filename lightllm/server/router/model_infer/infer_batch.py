import os
import copy
import time
import torch
import numpy as np
import collections

from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from lightllm.common.req_manager import ReqManager
from lightllm.common.mem_manager import MemoryManager
from lightllm.utils.infer_utils import mark_start, mark_end
from lightllm.server.io_struct import ReqRunStatus, FinishStatus
from lightllm.server.router.dynamic_prompt.radix_cache import RadixCache
from lightllm.utils.log_utils import init_logger
from lightllm.server.req_id_generator import convert_sub_id_to_group_id

logger = init_logger(__name__)
requests_mapping = {}
group_mapping = {}


class InferSamplingParams:
    def __init__(
        self,
        best_of: int = 1,
        do_sample: bool = False,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        exponential_decay_length_penalty: Tuple[int, float] = (1, 1.0),
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        vocab_size: int = -1,
        min_new_tokens: int = 1,
        max_new_tokens: int = 16,
        ignore_eos: bool = False,
        stop_sequences: List[List[int]] = [],
        input_penalty: bool = False,
    ) -> None:
        self.best_of = best_of
        self.do_sample = do_sample
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.repetition_penalty = repetition_penalty
        self.exponential_decay_length_penalty = exponential_decay_length_penalty
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_new_tokens = min_new_tokens
        self.max_new_tokens = max_new_tokens
        self.ignore_eos = ignore_eos
        self.stop_sequences = stop_sequences
        if self.top_k == -1:
            self.top_k = vocab_size
        self.input_penalty = input_penalty
        return


class InferReq:
    def __init__(
        self,
        r_id,
        group_req_id,
        input_token_ids=[],
        sampling_param=None,
        req_idx=-1,
        prompt_len=0,
        req_status=None,
        multimodal_params=None,
    ) -> None:
        self.r_id = r_id
        self.group_req_id = group_req_id
        self.sampling_param = sampling_param
        self.multimodal_params = multimodal_params
        self.req_idx = req_idx
        self.prompt_len = prompt_len
        self.input_token_ids = input_token_ids
        self.req_status = req_status
        self.cur_kv_len = 0  # 当前已经占用掉 token 现存的 kv len 长度
        self.shared_kv_node = None
        self.finish_status = FinishStatus.NO_FINISH
        self.logprobs = []  # logprob of each token, using for beamsearch and diverse_backend
        self.cum_logprob = 0.0  # cumulative logprob of each token, using for beamsearch and diverse_backend
        if self.sampling_param.input_penalty:
            self.out_token_id_count = collections.Counter(input_token_ids)
        else:
            self.out_token_id_count = collections.defaultdict(int)

        return

    def get_output_len(self):
        return len(self.input_token_ids) - self.prompt_len

    def update_finish_status(self, eos_ids):
        if self._stop_sequences_matched():
            self.finish_status = FinishStatus.FINISHED_STOP
        elif (
            len(self.input_token_ids) > self.prompt_len
            and self.input_token_ids[-1] in eos_ids
            and self.sampling_param.ignore_eos is False
        ):
            self.finish_status = FinishStatus.FINISHED_STOP
        elif len(self.input_token_ids) >= self.prompt_len + self.sampling_param.max_new_tokens:
            self.finish_status = FinishStatus.FINISHED_LENGTH
        return

    def _stop_sequences_matched(self):
        for stop_token_ids in self.sampling_param.stop_sequences:
            stop_len = len(stop_token_ids)
            output_len = len(self.input_token_ids) - self.prompt_len
            if stop_len > 0:
                if output_len >= stop_len:
                    if all(self.input_token_ids[i] == stop_token_ids[i] for i in range(-1, -(stop_len + 1), -1)):
                        return True
        return False


class InferReqGroup:
    def __init__(
        self,
        group_req_id,
        best_of=1,
    ) -> None:
        self.best_of = best_of
        self.refs = best_of
        self.group_req_id = group_req_id
        self.prev_beamid = [0] * best_of  # Record which beam the current token came from.
        self.min_score = 1e9  # The min score of beam results
        self.req_group = []
        self.res = []  # Results already finished
        self.filter_reqs_id = []  # filtered reqs
        self.finish_status = False  # If beamsearch is done
        self.has_beam = False

    def get_req(self, index):
        return requests_mapping[self.req_group[index]]

    def get_relative_index(self, index):
        return self.req_group[index] - self.group_req_id

    def add_req(self, req_id):
        self.req_group.append(req_id)

    def get_cumlogprobs(self):
        return [requests_mapping[r_id].cum_logprob for r_id in self.req_group]

    def decrease_refs(self, req_id):
        self.refs -= 1
        self.filter_reqs_id.append(req_id)
        return self.refs == 0

    def update_filter(self):
        filter_reqs_id_set = set(self.filter_reqs_id)
        new_req_group = [req_id for req_id in self.req_group if req_id not in filter_reqs_id_set]
        self.req_group = new_req_group
        self.best_of = len(self.req_group)

    def add_res(self, output_ids, logprobs, cum_logprob, finish_status):
        score = cum_logprob / len(output_ids)
        if len(self.res) < self.best_of or score < self.min_score:
            self.res.append([score, output_ids, logprobs, finish_status])
            if len(self.res) > self.best_of:
                sorted_scores = sorted([(s, idx) for idx, (s, _, _, _) in enumerate(self.res)])
                del self.res[sorted_scores[0][1]]
                self.min_score = sorted_scores[1][0]
            else:
                self.min_score = min(score, self.min_score)

    def beam_copy(self, req_manager, is_prefill):
        cache_req = {}
        cache_req_to_token = {}
        # record previous status
        for i, prev_ in enumerate(self.prev_beamid):
            prev_req = requests_mapping[self.req_group[prev_]]
            if prev_ not in cache_req_to_token:
                cache_req[prev_] = copy.deepcopy(prev_req)
                prev_tokens = req_manager.req_to_token_indexs[prev_req.req_idx][: len(prev_req.input_token_ids)].clone()
                cache_req_to_token[prev_] = prev_tokens
            req = self.get_req(i)
            if not is_prefill or i == 0:
                req_manager.mem_manager.free(req_manager.req_to_token_indexs[req.req_idx][: len(req.input_token_ids)])

        # update the InferReq status and mem_manager status for cache sharing
        for i, req_id in enumerate(self.req_group):
            prev_ = self.prev_beamid[i]
            prev_req = cache_req[prev_]
            req = requests_mapping[req_id]
            req.input_token_ids = copy.deepcopy(prev_req.input_token_ids)
            req.out_token_id_count = copy.deepcopy(req.out_token_id_count)
            req.logprobs = copy.deepcopy(req.logprobs)
            req.finish_status = FinishStatus.NO_FINISH
            req_manager.req_to_token_indexs[req.req_idx][: len(req.input_token_ids)] = cache_req_to_token[prev_]
            req_manager.mem_manager.add_refs(cache_req_to_token[prev_])

    def update_finish_status(self, best_new_score):
        if len(self.res) < self.best_of:
            self.finish_status = False
        else:
            req_obj = requests_mapping[self.req_group[0]]
            max_new_tokens = req_obj.sampling_param.max_new_tokens
            has_output_len = req_obj.cur_kv_len + 1 - req_obj.prompt_len  # 这个地方只能用 cur_kv_len 来计算，有beam流程函数存在依赖
            self.finish_status = best_new_score <= self.min_score or has_output_len >= max_new_tokens

        if self.finish_status:
            self.res = sorted(self.res, key=lambda i: i[0])


@dataclass
class InferBatch:
    batch_id: int
    request_ids: List
    req_manager: ReqManager
    radix_cache: RadixCache

    @classmethod
    @torch.no_grad()
    def init_batch(
        cls,
        batch_id,
        requests,
        dtype: torch.dtype,
        device: torch.device,
        req_manager: ReqManager,
        vocab_size: int,
        radix_cache: RadixCache = None,
    ):

        request_ids = []
        need_alloc_size = len([r for r in requests if r["request_id"] not in requests_mapping])
        nopad_b_req_idx = req_manager.alloc(need_alloc_size)
        nopad_b_req_idx = nopad_b_req_idx.cpu().numpy()

        index = 0
        for r in requests:
            # request id -> idx in list mapping
            r_id = r["request_id"]
            if r_id not in requests_mapping.keys():
                tokenized_input = r["input_id"]
                input_length = len(tokenized_input)
                # postprocessor
                sampling_param = r["sampling_param"]
                multimodal_params = r["multimodal_params"]
                sampling_param["vocab_size"] = vocab_size
                assert r["req_status"] == ReqRunStatus.WAIT_IN_QUEUE
                group_req_id = r["group_req_id"]
                r_obj = InferReq(
                    r_id,
                    group_req_id,
                    input_token_ids=tokenized_input,
                    sampling_param=InferSamplingParams(**sampling_param),
                    multimodal_params=multimodal_params,
                    req_idx=nopad_b_req_idx[index],
                    prompt_len=input_length,
                    req_status=r["req_status"],
                )
                requests_mapping[r_id] = r_obj
                index += 1
            else:
                if requests_mapping[r_id].req_status == ReqRunStatus.PAUSED_AND_OFFLOAD:
                    r_obj: InferReq = requests_mapping[r_id]
                    r_obj.req_status = ReqRunStatus.RERUNNING_FROM_OFFLOAD
                else:
                    assert False, f"should not exist {requests_mapping[r_id].req_status}"

            request_ids.append(r_id)

            # 如果是具有 prompt_cache 的使用特性则需要进行提前的填充和恢复操作。
            if r_obj.req_status in [ReqRunStatus.RERUNNING_FROM_OFFLOAD, ReqRunStatus.WAIT_IN_QUEUE]:
                if radix_cache is not None:
                    key = torch.tensor(r_obj.input_token_ids, dtype=torch.int64, device="cpu")
                    key = key[0 : len(key) - 1]  # 最后一个不需要，因为需要一个额外的token，让其在prefill的时候输出下一个token的值
                    share_node, kv_len, value_tensor = radix_cache.match_prefix(key, update_refs=True)
                    if share_node is not None:
                        r_obj.shared_kv_node = share_node
                        ready_cache_len = share_node.shared_idx_node.get_node_prefix_total_len()
                        mem_manager: MemoryManager = req_manager.mem_manager
                        value_tensor = value_tensor.long().cuda()
                        mem_manager.add_refs(value_tensor)  # 加 refs
                        req_manager.req_to_token_indexs[r_obj.req_idx, 0:ready_cache_len] = value_tensor
                        r_obj.cur_kv_len = ready_cache_len

            # 初始化之后 所有请求状态置换为 RUNNING 状态
            r_obj.req_status = ReqRunStatus.RUNNING

        return cls(
            batch_id=batch_id,
            request_ids=request_ids,
            req_manager=req_manager,
            radix_cache=radix_cache,
        )

    def _free_a_req_mem(self, free_token_index: List, req: InferReq):
        if self.radix_cache is None:
            free_token_index.append(self.req_manager.req_to_token_indexs[req.req_idx][: req.cur_kv_len])
        else:
            key = torch.tensor(req.input_token_ids[0 : req.cur_kv_len], dtype=torch.int64, device="cpu")
            value = self.req_manager.req_to_token_indexs[req.req_idx][: req.cur_kv_len].detach().cpu()
            prefix_len = self.radix_cache.insert(key, value)
            free_token_index.append(self.req_manager.req_to_token_indexs[req.req_idx][:prefix_len])
            if req.shared_kv_node is not None:
                assert req.shared_kv_node.shared_idx_node.get_node_prefix_total_len() <= prefix_len
                self.radix_cache.dec_node_ref_counter(req.shared_kv_node)
                req.shared_kv_node = None

    @torch.no_grad()
    def free_self(self):
        free_req_index = []
        free_token_index = []
        for request_id in self.request_ids:
            req: InferReq = requests_mapping.pop(request_id)
            group_req_id = convert_sub_id_to_group_id(req.r_id)
            if group_req_id in group_mapping:
                is_empty = group_mapping[group_req_id].decrease_refs(req.r_id)
                if is_empty:
                    del group_mapping[group_req_id]
            free_req_index.append(req.req_idx)
            self._free_a_req_mem(free_token_index, req)
            req.cur_kv_len = 0

        free_token_index = torch.cat(free_token_index, dim=-1)
        self.req_manager.free(free_req_index, free_token_index)
        if len(requests_mapping) == 0:
            requests_mapping.clear()
        if len(group_mapping) == 0:
            group_mapping.clear()

        if self.radix_cache is not None:
            logger.info(
                f"free a batch state:\n"
                f"radix refed token num {self.radix_cache.get_refed_tokens_num()}\n"
                f"radix hold token num {self.radix_cache.get_tree_total_tokens_num()}\n"
                f"mem manager can alloc token num {self.req_manager.mem_manager.can_use_mem_size}\n"
                f"mem manager total size {self.req_manager.mem_manager.size}"
            )
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
                batch_id=self.batch_id, request_ids=[], req_manager=self.req_manager, radix_cache=self.radix_cache
            )
        free_req_index = []
        free_token_index = []
        for request_id in finished_request_ids:
            req: InferReq = requests_mapping.pop(request_id)
            group_req_id = convert_sub_id_to_group_id(req.r_id)
            if group_req_id in group_mapping:
                is_empty = group_mapping[group_req_id].decrease_refs(req.r_id)
                if is_empty:
                    del group_mapping[group_req_id]
            free_req_index.append(req.req_idx)
            self._free_a_req_mem(free_token_index, req)
            req.cur_kv_len = 0

        free_token_index = torch.cat(free_token_index, dim=-1)
        self.req_manager.free(free_req_index, free_token_index)

        return InferBatch(
            batch_id=self.batch_id, request_ids=request_ids, req_manager=self.req_manager, radix_cache=self.radix_cache
        )

    @torch.no_grad()
    def pause_reqs(self, pause_reqs: List[str]):
        free_token_index = []
        for request_id, pause_way in pause_reqs:
            req: InferReq = requests_mapping[request_id]
            req.req_status = pause_way
            self.request_ids.remove(request_id)
            if pause_way == ReqRunStatus.PAUSED_AND_OFFLOAD:
                self._free_a_req_mem(free_token_index, req)
                req.cur_kv_len = 0

        if len(free_token_index) != 0:
            free_token_index = torch.cat(free_token_index, dim=-1)
            self.req_manager.free_token(free_token_index)

        return self

    @classmethod
    @torch.no_grad()
    def merge(cls, batch1, batch2):
        request_ids = batch1.request_ids + batch2.request_ids

        return InferBatch(
            batch_id=batch1.batch_id,
            request_ids=request_ids,
            req_manager=batch1.req_manager,
            radix_cache=batch1.radix_cache,
        )

    def __len__(self):
        return len(self.request_ids)
