import time
import torch
import numpy as np
import collections

from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from lightllm.common.req_manager import ReqManager
from lightllm.common.mem_manager import MemoryManager
from lightllm.utils.infer_utils import mark_start, mark_end
from lightllm.server.io_struct import ReqRunStatus
from lightllm.server.router.dynamic_prompt.radix_cache import RadixCache
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


requests_mapping = {}


class InferSamplingParams:
    def __init__(
        self,
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
    ) -> None:
        self.do_sample = do_sample
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.repetition_penalty = repetition_penalty
        self.exponential_decay_length_penalty = exponential_decay_length_penalty
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_new_tokens = min_new_tokens
        if self.top_k == -1:
            self.top_k = vocab_size
        return


class InferReq:
    def __init__(
        self,
        r_id,
        input_token_ids=[],
        out_token_id_count={},
        sampling_param=None,
        req_idx=-1,
        prompt_len=0,
        req_status=None,
        multimodal_params=None,
    ) -> None:
        self.r_id = r_id
        self.out_token_id_count = out_token_id_count
        self.sampling_param = sampling_param
        self.multimodal_params = multimodal_params
        self.req_idx = req_idx
        self.prompt_len = prompt_len
        self.input_token_ids = input_token_ids
        self.req_status = req_status
        self.cur_kv_len = 0  # 当前已经占用掉 token 现存的 kv len 长度

        self.shared_kv_node = None
        self.ready_cache_len = 0
        return


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
                r_obj = InferReq(
                    r_id,
                    input_token_ids=tokenized_input,
                    out_token_id_count=collections.defaultdict(int),
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
                        r_obj.ready_cache_len = share_node.shared_idx_node.get_node_prefix_total_len()
                        mem_manager: MemoryManager = req_manager.mem_manager
                        value_tensor = value_tensor.long().cuda()
                        mem_manager.add_refs(value_tensor)  # 加 refs
                        req_manager.req_to_token_indexs[r_obj.req_idx, 0 : r_obj.ready_cache_len] = value_tensor
                        r_obj.cur_kv_len = r_obj.ready_cache_len

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
                req.ready_cache_len = 0

    @torch.no_grad()
    def free_self(self):
        free_req_index = []
        free_token_index = []
        for request_id in self.request_ids:
            req: InferReq = requests_mapping.pop(request_id)
            free_req_index.append(req.req_idx)
            self._free_a_req_mem(free_token_index, req)
            req.cur_kv_len = 0

        free_token_index = torch.cat(free_token_index, dim=-1)
        self.req_manager.free(free_req_index, free_token_index)
        if len(requests_mapping) == 0:
            requests_mapping.clear()

        if self.radix_cache is not None:
            logger.info(
                f"""free a batch state:
                        radix refed token num {self.radix_cache.get_refed_tokens_num()}
                        radix hold token num {self.radix_cache.get_tree_total_tokens_num()}
                        mem manager can alloc token num {self.req_manager.mem_manager.can_use_mem_size}
                        mem manager total size {self.req_manager.mem_manager.size}"""
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
        for request_id, pause_way in pause_reqs:
            req: InferReq = requests_mapping[request_id]
            req.req_status = pause_way
            self.request_ids.remove(request_id)
            if pause_way == ReqRunStatus.PAUSED_AND_OFFLOAD:
                # 现在只支持全卸载一个请求的所有 kv 了
                free_token_index = []
                self._free_a_req_mem(free_token_index, req)
                self.req_manager.free_token(free_token_index[0])
                req.cur_kv_len = 0

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
