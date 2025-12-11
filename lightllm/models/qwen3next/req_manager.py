from typing import override, List

import torch

from lightllm.common.req_manager import ReqManager
from lightllm.models.qwen3next.mem_manager import Qwen3NextMemoryManager


class Qwen3NextReqManager(ReqManager):
    def __init__(self, max_request_num, max_sequence_length, mem_manager: Qwen3NextMemoryManager):
        super().__init__(max_request_num, max_sequence_length, mem_manager)
        self.EMPTY_BUFFER_INDEX = -1
        self.req_to_buffer_indexes = torch.zeros((self.max_request_num + 1), dtype=torch.int32, device="cuda")
        self.req_to_buffer_indexes[:] = self.EMPTY_BUFFER_INDEX

    @override
    def free(self, free_req_indexes: List[int], free_token_index):
        self.free_buffer(free_req_indexes)
        super().free(free_req_indexes, free_token_index)

    @override
    def free_all(self):
        self.req_to_buffer_indexes[:] = self.EMPTY_BUFFER_INDEX
        super().free_all()
        return

    @override
    def alloc(self):
        from lightllm.common.basemodel.infer_lock import g_infer_state_lock
        from lightllm.server.router.model_infer.infer_batch import g_infer_context

        req_index = super().alloc()

        g_infer_state_lock.acquire()
        if g_infer_context.radix_cache is not None:
            g_infer_context.radix_cache.free_radix_cache_to_get_enough_buffer(1)
        new_buffer_index = self.mem_manager.alloc_buffer(1)
        self.req_to_buffer_indexes[req_index] = new_buffer_index
        g_infer_state_lock.release()

        return req_index

    def free_buffer(self, free_req_indexes: List[int]):
        from lightllm.server.router.model_infer.infer_batch import g_infer_context

        if g_infer_context.radix_cache is None:
            self.mem_manager.free_buffer(self.req_to_buffer_indexes[free_req_indexes])
        self.req_to_buffer_indexes[free_req_indexes] = self.EMPTY_BUFFER_INDEX
        return
