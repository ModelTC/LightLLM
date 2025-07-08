import os
import copy
import time
import torch
import torch.distributed as dist
import numpy as np
import collections

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union, Any
from lightllm.common.req_manager import ReqManager
from lightllm.utils.infer_utils import mark_start, mark_end
from lightllm.server.core.objs import Req, SamplingParams, FinishStatus, ShmReqManager
from lightllm.server.router.dynamic_prompt.radix_cache import RadixCache, TreeNode
from lightllm.utils.log_utils import init_logger
from lightllm.server.req_id_generator import convert_sub_id_to_group_id
from lightllm.common.basemodel.infer_lock import g_infer_state_lock
from lightllm.server.multimodal_params import MultimodalParams
from lightllm.utils.custom_kernel_utis import custom_cat
from lightllm.utils.envs_utils import get_env_start_args

logger = init_logger(__name__)


@dataclass
class InferenceContext:
    req_manager: ReqManager = None  # gpu 请求管理
    radix_cache: RadixCache = None
    shm_req_manager: ShmReqManager = None  # 共享内存请求对象管理
    requests_mapping: Dict[int, "InferReq"] = None
    group_mapping = None  # 只有进行多输出模式下才有真的使用
    infer_req_ids = None
    vocab_size = None

    overlap_stream: torch.cuda.Stream = None  # 一些情况下推理进程进行异步折叠操作的异步流对象。

    def register(
        self, req_manager: ReqManager, radix_cache: RadixCache, shm_req_manager: ShmReqManager, vocab_size: int
    ):
        self.req_manager = req_manager
        self.radix_cache = radix_cache
        self.shm_req_manager = shm_req_manager

        self.requests_mapping = {}
        self.group_mapping: Dict[int, InferReqGroup] = {}
        self.infer_req_ids = []

        self.vocab_size = vocab_size
        return

    def get_overlap_stream(self) -> torch.cuda.Stream:
        if self.overlap_stream is None:
            self.overlap_stream = torch.cuda.Stream()
        return self.overlap_stream

    def add_reqs(self, requests: List[Tuple[int, int, Any, int]], init_req_obj=True):
        request_ids = []
        for r in requests:

            r_id, r_index, multimodal_params, _ = r
            if r_id not in self.requests_mapping.keys():
                r_obj = InferReq(
                    req_id=r_id,
                    req_idx=self.req_manager.alloc(),
                    shm_index=r_index,
                    multimodal_params=multimodal_params,
                    vocab_size=self.vocab_size,
                )
                self.requests_mapping[r_id] = r_obj
            else:
                r_obj: InferReq = self.requests_mapping[r_id]
                assert r_obj.paused is True

            request_ids.append(r_id)

            if init_req_obj:
                r_obj.init_all()

        self.infer_req_ids.extend(request_ids)

        return

    def free_a_req_mem(self, free_token_index: List, req: "InferReq", is_group_finished: bool):
        if self.radix_cache is None:
            if is_group_finished:
                free_token_index.append(self.req_manager.req_to_token_indexs[req.req_idx][0 : req.cur_kv_len])
            else:
                free_token_index.append(
                    self.req_manager.req_to_token_indexs[req.req_idx][req.shm_req.input_len : req.cur_kv_len]
                )
        else:
            input_token_ids = req.get_input_token_ids()
            key = torch.tensor(input_token_ids[0 : req.cur_kv_len], dtype=torch.int64, device="cpu")
            # .cpu() 是 流内阻塞操作
            value = self.req_manager.req_to_token_indexs[req.req_idx][: req.cur_kv_len].detach().cpu()

            if is_group_finished:
                prefix_len = self.radix_cache.insert(key, value)
                old_prefix_len = 0 if req.shared_kv_node is None else req.shared_kv_node.node_prefix_total_len
                free_token_index.append(self.req_manager.req_to_token_indexs[req.req_idx][old_prefix_len:prefix_len])
                if req.shared_kv_node is not None:
                    assert req.shared_kv_node.node_prefix_total_len <= prefix_len
                    self.radix_cache.dec_node_ref_counter(req.shared_kv_node)
                    req.shared_kv_node = None
            else:
                free_token_index.append(
                    self.req_manager.req_to_token_indexs[req.req_idx][req.shm_req.input_len : req.cur_kv_len]
                )
                if req.shared_kv_node is not None:
                    self.radix_cache.dec_node_ref_counter(req.shared_kv_node)
                    req.shared_kv_node = None


    def _save_promptcache_kvbuffer(self):
        """
        save prompt cache kv buffer
        这个接口是用于保存非量化的缓存prompt cache资源，是定制场景使用的接口，当前代码中不会有调用。
        其保存的 kv 会配合量化推理模式, 加载到量化推理的prompt cache中, 提升量化推理的精度。
        like paper:
        https://arxiv.org/abs/2403.01241
        """
        prompt_cache_token_id = list(self.radix_cache.root_node.children.values())[0].token_id_key
        print(f"prompt_cache_token_id : {prompt_cache_token_id}")
        index = range(len(prompt_cache_token_id))
        prompt_cache_kv_buffer = self.radix_cache.mem_manager.get_index_kv_buffer(index)
        torch.save(prompt_cache_kv_buffer, f"prompt_cache_rank_{dist.get_rank()}.pt")

    @torch.no_grad()
    def filter(self, finished_request_ids: List[int]):
        if len(finished_request_ids) == 0:
            return

        free_req_index = []
        free_token_index = []
        for request_id in finished_request_ids:
            req: InferReq = self.requests_mapping.pop(request_id)
            group_req_id = convert_sub_id_to_group_id(req.shm_req.request_id)
            if group_req_id in self.group_mapping:
                is_group_finished = self.group_mapping[group_req_id].remove_req(req.shm_req.request_id)
                if is_group_finished:
                    del self.group_mapping[group_req_id]
                self.free_a_req_mem(free_token_index, req, is_group_finished)
            else:
                self.free_a_req_mem(free_token_index, req, True)
            free_req_index.append(req.req_idx)
            # logger.info(f"infer release req id {req.shm_req.request_id}")
            req.shm_req.shm_infer_released = True
            self.shm_req_manager.put_back_req_obj(req.shm_req)

        free_token_index = custom_cat(free_token_index)
        self.req_manager.free(free_req_index, free_token_index)

        finished_req_ids_set = set(finished_request_ids)
        self.infer_req_ids = [_id for _id in self.infer_req_ids if _id not in finished_req_ids_set]

        if self.radix_cache is not None and len(self.infer_req_ids) == 0:
            logger.debug(
                f"free a batch state:\n"
                f"radix refed token num {self.radix_cache.get_refed_tokens_num()}\n"
                f"radix hold token num {self.radix_cache.get_tree_total_tokens_num()}\n"
                f"mem manager can alloc token num {self.req_manager.mem_manager.can_use_mem_size}\n"
                f"mem manager total size {self.req_manager.mem_manager.size}"
            )

        return

    def filter_reqs(self, finished_reqs: List["InferReq"]):
        if finished_reqs:
            self.filter([req.req_id for req in finished_reqs])
        return

    @torch.no_grad()
    def pause_reqs(self, pause_req_ids: List[int]):
        free_token_index = []
        for request_id in pause_req_ids:
            req: InferReq = self.requests_mapping[request_id]
            self.infer_req_ids.remove(request_id)

            if req.initialized:
                # 不支持多输出的情况的暂停
                self.free_a_req_mem(free_token_index, req, is_group_finished=True)
                req.cur_kv_len = 0
                req.shm_req.shm_cur_kv_len = req.cur_kv_len
                req.paused = True  # 暂停信息标记。
            else:
                req.paused = True

        if len(free_token_index) != 0:
            free_token_index = custom_cat(free_token_index)
            self.req_manager.free_token(free_token_index)

        return self


g_infer_context = InferenceContext()


class InferSamplingParams:
    def __init__(
        self,
        shm_req: Req,
        vocab_size: int,
    ) -> None:
        self.shm_param = shm_req.sample_params
        if self.shm_param.top_k == -1:
            self.shm_param.top_k = vocab_size

        # output constraint states
        self.regular_constraint = self.shm_param.regular_constraint.to_str()
        self.guided_grammar = self.shm_param.guided_grammar.to_str()
        self.guided_json = self.shm_param.guided_json.to_str()
        if len(self.regular_constraint) == 0:
            self.regular_constraint = None
        if len(self.guided_grammar) == 0:
            self.guided_grammar = None
        if len(self.guided_json) == 0:
            self.guided_json = None

        self.fsm_current_state: int = 0
        self.allowed_token_ids = self.shm_param.allowed_token_ids.to_list()
        if len(self.allowed_token_ids) == 0:
            self.allowed_token_ids = None

        # p d mode use params
        if self.shm_param.move_kv_to_decode_node.exists:
            self.move_kv_to_decode_node = self.shm_param.move_kv_to_decode_node.to_dict()
        else:
            self.move_kv_to_decode_node = None

        # this check is not very good to placed here. to do...
        if self.allowed_token_ids is not None:
            if not all(e < vocab_size for e in self.allowed_token_ids):
                logger.error("allowed_token_ids contain tokenid >= vobsize, we remove these token ids")
                self.allowed_token_ids = [e for e in self.allowed_token_ids if e < vocab_size]
        return

    def has_constraint_setting(self) -> bool:
        return (
            self.regular_constraint is not None
            or self.allowed_token_ids is not None
            or self.guided_grammar is not None
            or self.guided_json is not None
        )


class InferReq:
    def __init__(
        self,
        req_id: int,
        req_idx: int,
        shm_index: int,
        multimodal_params=None,
        vocab_size: int = -1,
    ):
        self.req_id = req_id
        self.req_idx = req_idx
        self.shm_index = shm_index
        self.multimodal_params = multimodal_params
        self.vocab_size = vocab_size
        self.initialized = False
        self.paused = False
        self.need_out_token_id_statistics = True
        self.out_token_id_count: Dict[int, int] = None

        # mtp_gen_token_ids 用于处理一个请求可以通过mtp进行很多token的预先生成
        # 的技术，在没有开启 mtp 功能的时候，这个成员变量不会有任何的实际实用意义。
        # 当开启后，mtp_gen_token_ids 保存多生成的多余的token_id,但是在后面的
        # 步骤中需要重新进行校验。
        self.mtp_gen_token_ids: List[int] = []

    def init_all(self):
        if self.initialized is False:
            self.shm_req = g_infer_context.shm_req_manager.get_req_obj_by_index(self.shm_index)
            self.shm_req.link_prompt_ids_shm_array()
            self.shm_req.link_logprobs_shm_array()
            self.sampling_param: InferSamplingParams = InferSamplingParams(self.shm_req, self.vocab_size)

            g_infer_context.req_manager.req_sampling_params_manager.init_req_sampling_params(self)

            self.stop_sequences = self.sampling_param.shm_param.stop_sequences.to_list()
            # token healing mode 才被使用的管理对象
            if self.shm_req.prefix_token_ids.size != 0:
                self.prefix_token_ids = self.shm_req.prefix_token_ids.get_token_ids()
            else:
                self.prefix_token_ids = []
            self.multimodal_params = self.multimodal_params.to_dict()
            self.shared_kv_node: TreeNode = None
            self.cur_kv_len = 0
            self.cur_output_len = 0
            self.finish_status = FinishStatus()

        if self.paused or not self.initialized:
            # 如果是具有 prompt_cache 的使用特性则需要进行提前的填充和恢复操作。
            if g_infer_context.radix_cache is not None and self.get_cur_total_len() > 1:
                input_token_ids = self.shm_req.shm_prompt_ids.arr[0 : self.get_cur_total_len()]
                key = torch.tensor(input_token_ids, dtype=torch.int64, device="cpu")
                key = key[0 : len(key) - 1]  # 最后一个不需要，因为需要一个额外的token，让其在prefill的时候输出下一个token的值
                share_node, kv_len, value_tensor = g_infer_context.radix_cache.match_prefix(key, update_refs=True)
                if share_node is not None:
                    self.shared_kv_node = share_node
                    ready_cache_len = share_node.node_prefix_total_len
                    # 从 cpu 到 gpu 是流内阻塞操作
                    g_infer_context.req_manager.req_to_token_indexs[self.req_idx, 0:ready_cache_len] = value_tensor
                    self.cur_kv_len = int(ready_cache_len)  # 序列化问题, 该对象可能为numpy.int64，用 int(*)转换
                    self.shm_req.prompt_cache_len = self.cur_kv_len  # 记录 prompt cache 的命中长度

            self.shm_req.shm_cur_kv_len = self.cur_kv_len

        self.initialized = True
        self.paused = False
        return

    def is_uninitialized(self):
        return not self.initialized or self.paused

    def get_output_len(self):
        return self.cur_output_len

    def get_cur_total_len(self):
        return self.shm_req.input_len + self.cur_output_len

    def get_input_token_ids(self):
        return self.shm_req.shm_prompt_ids.arr[0 : self.get_cur_total_len()]

    def get_chuncked_input_token_ids(self):
        chunked_start = self.cur_kv_len
        chunked_end = min(self.get_cur_total_len(), chunked_start + self.shm_req.chunked_prefill_size)
        return self.shm_req.shm_prompt_ids.arr[0:chunked_end]

    def get_chuncked_input_token_len(self):
        chunked_start = self.cur_kv_len
        chunked_end = min(self.get_cur_total_len(), chunked_start + self.shm_req.chunked_prefill_size)
        return chunked_end

    def set_next_gen_token_id(self, next_token_id: int, logprob: float):
        index = self.get_cur_total_len()
        self.shm_req.shm_prompt_ids.arr[index] = next_token_id
        self.shm_req.shm_logprobs.arr[index] = logprob
        return

    def update_mtp_accepted_token_num(self, accept_token_num: int):
        # 用于统计 mtp 的接受率
        self.shm_req.mtp_accepted_token_num += accept_token_num

    def get_last_gen_token(self):
        return self.shm_req.shm_prompt_ids.arr[self.shm_req.input_len + self.cur_output_len - 1]

    def update_finish_status(self, eos_ids):
        if self._stop_sequences_matched():
            self.finish_status.set_status(FinishStatus.FINISHED_STOP)
        elif (
            self.cur_output_len > 0
            and self.get_last_gen_token() in eos_ids
            and self.sampling_param.shm_param.ignore_eos is False
        ):
            self.finish_status.set_status(FinishStatus.FINISHED_STOP)
        elif self.cur_output_len >= self.sampling_param.shm_param.max_new_tokens:
            self.finish_status.set_status(FinishStatus.FINISHED_LENGTH)
        return

    def is_finished_or_aborted(self):
        return self.finish_status.is_finished() or self.shm_req.router_aborted

    def _stop_sequences_matched(self):
        for stop_token_ids in self.stop_sequences:
            stop_len = len(stop_token_ids)
            output_len = self.cur_output_len
            if stop_len > 0:
                if output_len >= stop_len:
                    input_token_ids = self.shm_req.shm_prompt_ids.arr[
                        0 : (self.shm_req.input_len + self.cur_output_len)
                    ]
                    if all(input_token_ids[i] == stop_token_ids[i] for i in range(-1, -(stop_len + 1), -1)):
                        return True
        return False


class InferReqGroup:
    def __init__(
        self,
        group_req_id: int,
    ) -> None:
        self.group_req_id = group_req_id
        self.req_ids_group = []

    def get_req(self, index):
        return g_infer_context.requests_mapping[self.req_ids_group[index]]

    def get_all_reqs(self):
        return [g_infer_context.requests_mapping[self.req_ids_group[i]] for i in range(len(self.req_ids_group))]

    def add_req(self, req_id):
        self.req_ids_group.append(req_id)

    def remove_req(self, req_id):
        assert req_id in self.req_ids_group
        self.req_ids_group.remove(req_id)
        return len(self.req_ids_group) == 0

    def best_of(self):
        return len(self.req_ids_group)

    def diverse_copy(self, req_manager, is_prefill):
        # record previous status
        prev_req = g_infer_context.requests_mapping[convert_sub_id_to_group_id(self.req_ids_group[0])]
        if prev_req.shared_kv_node is not None:
            prefix_len = prev_req.shared_kv_node.node_prefix_total_len
        else:
            prefix_len = 0
        prefix_len = max(prefix_len, prev_req.cur_kv_len)
        pre_input_len = prev_req.get_chuncked_input_token_len()
        cache_token_id = req_manager.req_to_token_indexs[prev_req.req_idx][prefix_len:pre_input_len]
        # update the InferReq status and mem_manager status for cache sharing
        for req_id in self.req_ids_group[:]:
            if req_id == convert_sub_id_to_group_id(req_id):
                continue
            req = g_infer_context.requests_mapping[req_id]
            req.finish_status.set_status(FinishStatus.NO_FINISH)
            input_len = req.get_chuncked_input_token_len()
            req_manager.req_to_token_indexs[req.req_idx][prefix_len:input_len] = cache_token_id
            assert input_len == pre_input_len
