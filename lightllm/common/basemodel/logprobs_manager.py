import os
from typing import Optional, Tuple

import numpy as np
import torch

from lightllm.common.basemodel.triton_kernel.logprobs_capture import scatter_prompt_logprobs_to_cpu
from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)

_MAX_PROMPT_LOGPROBS = int(os.getenv("LIGHTLLM_MAX_PROMPT_LOGPROBS", 1024))


class PromptLogprobsCaptureManager:
    def __init__(self, kv_cache_size: int):
        self.kv_cache_size = kv_cache_size
        self.max_topk = _MAX_PROMPT_LOGPROBS
        self.top_token_ids = torch.empty(
            (kv_cache_size, self.max_topk), dtype=torch.int32, device="cpu", pin_memory=True
        )
        self.top_logprobs = torch.empty(
            (kv_cache_size, self.max_topk), dtype=torch.float32, device="cpu", pin_memory=True
        )
        self.top_token_ids_ptr = self._cuda_pointer(self.top_token_ids)
        self.top_logprobs_ptr = self._cuda_pointer(self.top_logprobs)

        pinned_bytes = self.top_token_ids.numel() * self.top_token_ids.element_size()
        pinned_bytes += self.top_logprobs.numel() * self.top_logprobs.element_size()
        logger.info(
            f"PromptLogprobsCaptureManager initialized: kv_cache_size={kv_cache_size}, "
            f"max_topk={self.max_topk}, pinned_memory={pinned_bytes / 1024 / 1024:.2f}MB"
        )

    @staticmethod
    def _cuda_pointer(buffer: torch.Tensor) -> torch.Tensor:
        return torch.tensor([buffer.data_ptr()], dtype=torch.uint64, device="cuda")

    @staticmethod
    def _cpu_indexes(mem_indexes: torch.Tensor) -> torch.Tensor:
        return mem_indexes.cpu() if mem_indexes.is_cuda else mem_indexes

    def capture(
        self,
        mem_indexes: torch.Tensor,
        top_token_ids: torch.Tensor,
        top_logprobs: torch.Tensor,
    ) -> None:
        scatter_prompt_logprobs_to_cpu(
            mem_indexes=mem_indexes,
            top_token_ids=top_token_ids,
            top_logprobs=top_logprobs,
            top_token_ids_buffer_ptr=self.top_token_ids_ptr,
            top_logprobs_buffer_ptr=self.top_logprobs_ptr,
            kv_cache_size=self.kv_cache_size,
            max_topk=self.max_topk,
        )

    def extract(self, mem_indexes: torch.Tensor, topk: int) -> Tuple[np.ndarray, np.ndarray]:
        indexes = self._cpu_indexes(mem_indexes)
        return (
            self.top_token_ids[indexes, :topk].numpy(),
            self.top_logprobs[indexes, :topk].numpy(),
        )

    def copy_slots(
        self,
        source_indexes: torch.Tensor,
        destination_indexes: torch.Tensor,
        topk: int,
    ) -> None:
        source = self._cpu_indexes(source_indexes)
        destination = self._cpu_indexes(destination_indexes)
        self.top_token_ids[destination, :topk] = self.top_token_ids[source, :topk]
        self.top_logprobs[destination, :topk] = self.top_logprobs[source, :topk]


g_prompt_logprobs_capture_manager: Optional[PromptLogprobsCaptureManager] = None


def init_prompt_logprobs_capture(mem_manager) -> PromptLogprobsCaptureManager:
    global g_prompt_logprobs_capture_manager
    if g_prompt_logprobs_capture_manager is None:
        g_prompt_logprobs_capture_manager = PromptLogprobsCaptureManager(kv_cache_size=mem_manager.size + 1)
    return g_prompt_logprobs_capture_manager
