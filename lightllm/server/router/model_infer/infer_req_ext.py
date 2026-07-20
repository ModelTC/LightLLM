"""InferReq 功能扩展包裹对象。

将部分与 InferReq 强相关、但不适合继续堆在 InferReq 类体上的逻辑拆出，
由 InferReq 在初始化时挂载为成员，调用方通过 ``req.<ext>`` 访问。

当前包含：
- :class:`PromptSelectedLogprobsExt`：``prompt_logprobs=0`` 路径的异步落盘缓冲
"""
import torch

from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from lightllm.server.router.model_infer.infer_batch import InferReq


class PromptSelectedLogprobsExt:
    """``prompt_logprobs=0`` 时，真实命中 prompt token 的 logprob/rank 异步缓冲。

    背景
    ----
    ``prompt_logprobs`` 有两条互斥路径：

    - ``topk > 0``：走 :class:`PromptLogprobsCaptureManager`，按 KV mem slot
      写入 pinned buffer，请求结束时再 extract。
    - ``topk == 0``：返回真实命中的 prompt token 的 logprob 与全词表 rank，
      HTTP 侧从 ``shm_logprobs`` 读取。本类服务后者。

    为何需要缓冲而不是 prefill 当场写 shm
    ------------------------------------
    prefill（含 chunked prefill）热路径上，logprob/rank 刚在 GPU 上算完。
    若立刻 ``.cpu()`` / synchronize 再写 shm，会打断 overlap、拖慢调度。
    因此这里只发起 **非阻塞 D2H**，把结果挂到 chunk 列表；真正需要交给
    HTTP 进程前（``_dump_final_token_metadata`` → ``flush``）再
    ``event.synchronize()`` 并写入 ``shm_logprobs``。

    chunk 语义
    ----------
    chunked prefill 会多次调用 :meth:`add_chunk`，每次对应一段
    ``[target_start, target_end)`` 的 prompt 位置。flush 时按段写回 shm。

    生命周期
    --------
    InferReq 初始化时创建本对象；请求结束 dump metadata 时 flush；
    之后 chunks 清空。必须在 ``shm_infer_released=True`` 之前完成 flush，
    否则 HTTP recycle 可能读到未写完的 shm_logprobs。
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
