from typing import List, Dict
from typing_extensions import override
import torch

from lightllm.common.req_manager import ReqManager
from lightllm.models.qwen3next.mem_manager import Qwen3NextMemoryManager
from lightllm.utils.envs_utils import get_env_start_args


class Qwen3NextReqManager(ReqManager):
    def __init__(self, max_request_num, max_sequence_length, mem_manager: Qwen3NextMemoryManager):
        super().__init__(max_request_num, max_sequence_length, mem_manager)
        self.mem_manager: Qwen3NextMemoryManager = self.mem_manager
        self.enable_dynamic_prompt_cache = not get_env_start_args().disable_dynamic_prompt_cache

        self.req_to_buffer_indexes = torch.zeros((max_request_num + 1), dtype=torch.int32, device="cuda")
        self.req_to_buffer_indexes[:] = self.mem_manager.EMPTY_BUFFER_INDEX
        self.req_to_buffer_indexes[self.HOLD_REQUEST_ID] = self.mem_manager.HOLD_BUFFER_INDEX

    @override
    def alloc(self):
        from lightllm.server.router.model_infer.infer_batch import g_infer_state_lock, g_infer_context

        req_idx = super().alloc()
        g_infer_state_lock.acquire()
        if g_infer_context.radix_cache is not None:
            g_infer_context.radix_cache.free_radix_cache_to_get_enough_token(0, 1)
        self.req_to_buffer_indexes[req_idx] = self.mem_manager.alloc_state_cache_buffer(1)
        g_infer_state_lock.release()
        return req_idx

    @override
    def free(self, free_req_indexes: List[int], free_token_index):
        super().free(free_req_indexes, free_token_index)
        self.req_to_buffer_indexes[free_req_indexes] = self.mem_manager.EMPTY_BUFFER_INDEX

    @override
    def free_all(self):
        super().free_all()
        self.req_to_buffer_indexes[:] = self.mem_manager.EMPTY_BUFFER_INDEX
        self.req_to_buffer_indexes[self.HOLD_REQUEST_ID] = self.mem_manager.HOLD_BUFFER_INDEX
        return
