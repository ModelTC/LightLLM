"""InferReq 功能扩展包裹对象。

将部分与 InferReq 强相关、但不适合继续堆在 InferReq / InferenceContext
类体上的逻辑拆出，由 InferReq 在初始化时挂载为成员，调用方通过
``req.<ext>`` 访问。

当前包含：
- :class:`PromptSelectedLogprobsExt`：prompt logprobs 相关辅助
  （``topk=0`` 异步落盘 + ``topk>0`` mem slot 拷贝）
- :class:`FinalTokenMetadataExt`：请求结束时汇总并写出 final token metadata
"""

from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import torch

from lightllm.common.basemodel.logprobs_manager import PromptLogprobsCaptureManager
from lightllm.common.basemodel.moe_route_info_manager import MoeRouteInfoManager

if TYPE_CHECKING:
    from lightllm.server.router.model_infer.infer_batch import InferReq


class PromptSelectedLogprobsExt:
    """InferReq 上的 prompt logprobs 辅助对象。

    覆盖两条路径中与本请求强绑定的操作：

    - ``topk == 0``：:meth:`add_chunk` / :meth:`flush` —— 热路径异步 D2H，
      结束时写入 ``shm_logprobs``（HTTP 直接读）。
    - mem 重映射：:meth:`copy_capture_slots_if_needed` —— radix / linear
      att 匹配把 KV 拷到新 mem slot 时，同步拷贝 CaptureManager 中已有的
      top-k 数据，保证后续请求复用 radix 时元数据仍完整。

    为何 ``topk==0`` 需要缓冲而不是 prefill 当场写 shm
    -----------------------------------------------
    prefill（含 chunked prefill）热路径上立刻 ``.cpu()`` / synchronize 会
    打断 overlap。因此只做非阻塞 D2H；``FinalTokenMetadataExt.dump`` →
    ``flush`` 时再 sync 写 shm。

    chunk 语义（``topk==0``）
    ------------------------
    chunked prefill 多次 :meth:`add_chunk`，每段对应
    ``[target_start, target_end)``；flush 时按段写回。

    生命周期
    --------
    InferReq 初始化时创建；dump metadata 前 flush；必须在
    ``shm_infer_released=True`` 之前完成，避免 HTTP 读到未写完数据。
    """

    # (target_start, target_end, logprobs_cpu, ranks_cpu, copy_done_event)
    _Chunk = Tuple[int, int, torch.Tensor, torch.Tensor, torch.cuda.Event]

    def __init__(self, req: "InferReq") -> None:
        self._req = req
        self._chunks: List[PromptSelectedLogprobsExt._Chunk] = []

    def add_chunk(
        self,
        target_start: int,
        target_end: int,
        logprobs: torch.Tensor,
        ranks: torch.Tensor,
    ) -> None:
        """登记一段 GPU 上的 selected logprobs，异步拷到 pinned CPU。

        Args:
            target_start: 写入 ``shm_logprobs`` 的起始 prompt 下标（含）。
            target_end: 写入 ``shm_logprobs`` 的结束 prompt 下标（不含）。
            logprobs: shape ``[end - start]``，GPU tensor。
            ranks: shape ``[end - start]``，GPU tensor（1-based rank）。
        """
        logprobs_cpu = torch.empty(logprobs.shape, dtype=logprobs.dtype, device="cpu", pin_memory=True)
        ranks_cpu = torch.empty(ranks.shape, dtype=ranks.dtype, device="cpu", pin_memory=True)
        logprobs_cpu.copy_(logprobs, non_blocking=True)
        ranks_cpu.copy_(ranks, non_blocking=True)
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        self._chunks.append((target_start, target_end, logprobs_cpu, ranks_cpu, event))

    def flush(self) -> None:
        """等待所有异步 D2H 完成，将结果写入 ``shm_logprobs``。

        调用时机：请求结束、写出 final token metadata 之前。
        HTTP 进程对 ``prompt_logprobs=0`` 依赖 ``shm_logprobs``，因此必须在
        标记 ``shm_infer_released`` 之前完成提交。
        """
        if not self._chunks:
            return

        shm_logprobs = self._req.shm_req.shm_logprobs
        for target_start, target_end, logprobs_cpu, ranks_cpu, event in self._chunks:
            event.synchronize()
            shm_logprobs.arr["logprob"][target_start:target_end] = logprobs_cpu.numpy()
            shm_logprobs.arr["rank"][target_start:target_end] = ranks_cpu.numpy()

        self._chunks.clear()

    def copy_capture_slots_if_needed(
        self,
        source_indexes: torch.Tensor,
        destination_indexes: torch.Tensor,
    ) -> None:
        """KV mem slot 重映射时，拷贝 CaptureManager 中的 prompt top-k 数据。

        场景：linear att radix 小页命中后，尾部 KV 从共享小页 mem 拷到新申请的
        ``tail_mems``。top-k prompt logprobs 按 mem slot 索引存放，必须随 KV
        一并拷到新 slot。

        不能按「当前请求的 prompt_logprobs」决定是否拷贝：source slot 上的
        数据可能来自更早的 ``topk>0`` 请求；destination 之后还可能插回
        radix cache，被后续 ``topk>0`` 请求复用。若此处因当前 ``topk<=0``
        跳过拷贝，后续复用方会 extract 到空/错误数据。只要 capture buffer
        已初始化，就按 ``max_topk`` 全宽拷贝以保留可复用元数据。
        """
        mgr = PromptLogprobsCaptureManager.get_instance()
        if mgr is None or not mgr.is_buffer_initialized():
            return

        mgr.copy_slots(
            source_indexes=source_indexes,
            destination_indexes=destination_indexes,
            topk=mgr.max_topk,
        )


class FinalTokenMetadataExt:
    """请求结束时汇总可选元信息，并写入 final token metadata shm。

    调用时机
    --------
    仅 DP master 在 ``InferenceContext._filter(modify_shm_finish_state=True)``
    路径、真正释放请求前调用 :meth:`dump`。必须在 ``shm_infer_released=True``
    之前完成，供 HTTP 侧编码进响应。

    与 :class:`PromptSelectedLogprobsExt` 的分工
    ------------------------------------------
    - ``prompt_logprobs=0``：热路径写入 ``PromptSelectedLogprobsExt``，dump
      时 flush 到 ``shm_logprobs``（HTTP 直接读该 shm）。
    - ``prompt_logprobs>0`` / routed experts：prefill/decode 期间按 mem slot
      落在对应 CaptureManager；dump 时按本请求 mem_indexes extract，再与
      其它字段一并 pickle 进 metadata shm。
    """

    def __init__(self, req: "InferReq") -> None:
        self._req = req

    def _mem_indexes(self) -> torch.Tensor:
        # 延迟导入，避免与 infer_batch 形成模块级循环依赖。
        from lightllm.server.router.model_infer.infer_batch import g_infer_context

        return g_infer_context.req_manager.req_to_token_indexs[self._req.req_idx]

    def collect_prompt_logprobs(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """从 PromptLogprobsCaptureManager 提取 ``prompt_logprobs>0`` 的 top-k。

        Returns:
            ``(top_token_ids, top_logprobs)``，或无需收集时返回 ``None``
            （``topk<=0`` / prompt 过短）。
        """
        req = self._req
        topk = req.sampling_param.shm_param.prompt_logprobs
        if topk <= 0 or req.shm_req.input_len <= 1:
            return None

        mgr = PromptLogprobsCaptureManager.get_instance()
        mem_indexes = self._mem_indexes()[: req.shm_req.input_len - 1]
        return mgr.extract(mem_indexes, topk)

    def collect_routed_experts(self) -> Optional[np.ndarray]:
        """从 MoeRouteInfoManager 提取已完成请求的 routed expert 信息。

        仅在请求已 finished / stop_str_matched，且存在可导出长度时返回数据。
        """
        req = self._req
        if not (req.shm_req.finish_status.is_finished() or req.shm_req.stop_str_matched):
            return None

        visible_total_len = req.shm_req.input_len + req.shm_req.shm_cur_output_len
        capture_len = min(req.cur_kv_len, visible_total_len - 1)
        if capture_len <= 0:
            return None

        mem_indexes = self._mem_indexes()[0:capture_len]
        return MoeRouteInfoManager.get_instance().extract(mem_indexes)

    def dump(self) -> None:
        """汇总各路径元信息并写入 final token metadata shm。"""
        req = self._req
        prompt_top_token_ids = None
        prompt_top_logprobs = None
        routed_experts = None

        # 阶段 1：落盘 prompt_logprobs=0 的异步缓冲。
        # prefill 热路径只做了非阻塞 D2H，这里 sync 后写入 shm_logprobs，
        # HTTP 对该模式直接从 shm_logprobs 读 logprob/rank。
        req.prompt_selected_logprobs.flush()

        # 阶段 2：收集 prompt_logprobs>0 的 top-k 结果。
        # 数据按 KV mem slot 存在 PromptLogprobsCaptureManager 中，
        # 按当前 req 的 mem_indexes extract 后交给 metadata shm。
        prompt_logprobs_mgr = PromptLogprobsCaptureManager.get_instance()
        if prompt_logprobs_mgr is not None and prompt_logprobs_mgr.is_buffer_initialized():
            collected = self.collect_prompt_logprobs()
            if collected is not None:
                prompt_top_token_ids, prompt_top_logprobs = collected

        # 阶段 3：收集 MoE routed experts（若开启 enable_return_routed_experts）。
        # 同样按 mem slot 从 MoeRouteInfoManager 导出。
        moe_mgr = MoeRouteInfoManager.get_instance()
        if moe_mgr is not None and moe_mgr.is_buffer_initialized():
            routed_experts = self.collect_routed_experts()

        # 阶段 4：统一 pickle 写入 final token metadata shm，供 HTTP 编码进响应。
        req.shm_req.get_final_token_metadata().save(
            prompt_top_token_ids=prompt_top_token_ids,
            prompt_top_logprobs=prompt_top_logprobs,
            routed_experts=routed_experts,
        )
