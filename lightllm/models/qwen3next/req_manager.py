from typing import List
from lightllm.common.req_manager import ReqManager
from lightllm.models.qwen3next.mem_manager import Qwen3NextMemoryManager


class Qwen3NextReqManager(ReqManager):
    def __init__(self, max_request_num, max_sequence_length, mem_manager: Qwen3NextMemoryManager):
        super().__init__(max_request_num, max_sequence_length, mem_manager)

    def free(self, free_req_indexes: List[int], free_token_index):
        super().free(free_req_indexes, free_token_index)
        assert isinstance(self.mem_manager, Qwen3NextMemoryManager)
        self.mem_manager.free_mamba_state_buffer(free_req_indexes)
