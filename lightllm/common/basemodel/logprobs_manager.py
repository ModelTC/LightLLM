"""Prompt logprobs 捕获与配置管理。

本模块为 ``--enable_prompt_logprobs`` 提供端到端支持：在 prefill 过程中记录
prompt 各位置的 top-k token id 与对应 logprob，并在请求结束时写入 final token
metadata shm，供 HTTP 进程编码进 API 返回的 ``prompt_logprobs``。

背景与目标
----------
开启 ``enable_prompt_logprobs`` 后，请求可通过 ``prompt_logprobs=k`` 要求返回
每个 prompt 位置的 top-k 候选。推理侧需要在 prefill 产出 logits 时把 top-k
结果按 KV 槽位落到 CPU pinned buffer，避免阻塞 GPU，并在请求结束时按 mem
indexes 导出。

``prompt_logprobs=0`` 走另一条路径（返回真实命中 token 的 logprob/rank），
不经过本 manager 的 top-k buffer。

核心职责
--------
1. **配置（phase-1）**
   记录 ``max_topk``（由环境变量 ``LIGHTLLM_MAX_PROMPT_LOGPROBS`` 控制上限）。

2. **捕获缓冲（phase-2，可选）**
   在 infer 进程按 KV cache 槽位分配 pinned CPU buffer：
   ``top_token_ids[kv_slot, max_topk]`` / ``top_logprobs[kv_slot, max_topk]``。
   Triton kernel 将 GPU 上的 top-k 结果 scatter 写入该 buffer。

3. **捕获 / 导出 / 槽位拷贝**
   - ``capture``：prefill 时写入对应 mem indexes
   - ``extract``：请求结束时按 mem indexes 取出 ``(token_ids, logprobs)``
   - ``copy_slots``：radix cache 等场景下在 KV 槽位间复制已捕获数据

进程与初始化
------------
- 进程内单例：``PromptLogprobsCaptureManager.get_instance()``。
  仅在 ``enable_prompt_logprobs`` 时创建，且只做 phase-1 配置初始化。
- **Infer 进程（通常 dp master）**：在 phase-1 之后调用
  ``init_capture_buffer(kv_cache_size)``，再参与 capture / extract。

数据流简图::

    prefill logits → topk
           │
           ▼
        capture(mem_indexes)  ──scatter──►  pinned CPU buffers
           │
           ▼  (request finished)
        extract(mem_indexes, topk)
           │
           ▼
    final_token_metadata shm  ──HTTP──►  response["prompt_logprobs"]
"""

import os
from typing import ClassVar, Optional, Tuple

import numpy as np
import torch

from lightllm.common.basemodel.triton_kernel.logprobs_capture import scatter_prompt_logprobs_to_cpu
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

_MAX_PROMPT_LOGPROBS = int(os.getenv("LIGHTLLM_MAX_PROMPT_LOGPROBS", 128))


class PromptLogprobsCaptureManager:
    """管理 prompt top-k logprobs 的配置、捕获缓冲与导出。

    详见模块文档字符串。
    """

    _instance: ClassVar[Optional["PromptLogprobsCaptureManager"]] = None

    @classmethod
    def get_instance(cls) -> Optional["PromptLogprobsCaptureManager"]:
        """Return the process singleton with phase-1 (config) init only.

        Capture buffer is optional and must be allocated separately via
        ``init_capture_buffer`` when needed (infer process).
        """
        if cls._instance is not None:
            return cls._instance

        from lightllm.utils.envs_utils import get_env_start_args

        args = get_env_start_args()
        if not args.enable_prompt_logprobs:
            return None

        cls._instance = cls(max_topk=_MAX_PROMPT_LOGPROBS)
        return cls._instance

    def __init__(self, max_topk: int):
        """Phase-1 init: config metadata only. Call init_capture_buffer() when capture is needed."""
        self.max_topk = max_topk
        self.kv_cache_size: Optional[int] = None
        self.top_token_ids: Optional[torch.Tensor] = None
        self.top_logprobs: Optional[torch.Tensor] = None
        self.top_token_ids_ptr: Optional[torch.Tensor] = None
        self.top_logprobs_ptr: Optional[torch.Tensor] = None

        logger.info(f"PromptLogprobsCaptureManager created: max_topk={max_topk}")

    def is_buffer_initialized(self) -> bool:
        """Whether phase-2 capture buffers have been allocated."""
        return self.top_token_ids is not None

    def init_capture_buffer(self, kv_cache_size: int) -> None:
        """Phase-2 init: allocate pinned CPU buffers for capture/extract."""
        if self.is_buffer_initialized():
            return

        self.kv_cache_size = kv_cache_size
        self.top_token_ids = torch.empty(
            (kv_cache_size, self.max_topk), dtype=torch.int32, device="cpu", pin_memory=True
        )
        self.top_logprobs = torch.empty(
            (kv_cache_size, self.max_topk), dtype=torch.float32, device="cpu", pin_memory=True
        )
        self.top_token_ids_ptr = torch.tensor([self.top_token_ids.data_ptr()], dtype=torch.uint64, device="cuda")
        self.top_logprobs_ptr = torch.tensor([self.top_logprobs.data_ptr()], dtype=torch.uint64, device="cuda")

        pinned_bytes = self.top_token_ids.numel() * self.top_token_ids.element_size()
        pinned_bytes += self.top_logprobs.numel() * self.top_logprobs.element_size()
        logger.info(
            f"PromptLogprobsCaptureManager capture buffer ready: kv_cache_size={kv_cache_size}, "
            f"max_topk={self.max_topk}, pinned_memory={pinned_bytes / 1024 / 1024:.2f}MB"
        )

    def capture(
        self,
        mem_indexes: torch.Tensor,
        top_token_ids: torch.Tensor,
        top_logprobs: torch.Tensor,
    ) -> None:
        if not self.is_buffer_initialized():
            return
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
        if not self.is_buffer_initialized():
            return
        indexes = mem_indexes.cpu() if mem_indexes.is_cuda else mem_indexes
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
        if not self.is_buffer_initialized():
            return
        source = source_indexes.cpu() if source_indexes.is_cuda else source_indexes
        destination = destination_indexes.cpu() if destination_indexes.is_cuda else destination_indexes
        self.top_token_ids[destination, :topk] = self.top_token_ids[source, :topk]
        self.top_logprobs[destination, :topk] = self.top_logprobs[source, :topk]
