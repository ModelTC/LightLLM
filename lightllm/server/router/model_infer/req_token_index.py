"""InferReq 的 token 索引扩展，仅保存请求槽位和索引状态。"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, List, Optional, Sequence

import torch

if TYPE_CHECKING:
    from lightllm.server.router.model_infer.infer_batch import InferReq


class ReqTokenIndex(ABC):
    """单个请求的 token 索引，使用调用方指定的流维护；区间右端点不包含在内。

    调用方保证 token 区间有效，并在构造时传入统一的索引流；在线服务在请求构造入口传入当前流。
    GPU 索引读写固定在索引流，返回 GPU 视图前让调用流等待，Helper 不维护逐请求 CUDA 事件。
    """

    def __init__(self, infer_req: "InferReq", index_stream: torch.cuda.Stream):
        # 仅保存固定槽位编号和 GPU 行视图，避免反向持有 InferReq 形成循环引用。
        self.req_idx = infer_req.req_idx
        self.gpu_indexes = infer_req.req_manager.req_to_token_indexs[self.req_idx]
        # 索引流由外部选择，后续 GPU 索引操作不随调用流切换。
        self._index_stream = index_stream
        # 当前连续记录的 token 索引长度，包含页内预留槽位；
        # CPU 扩容时复制 [0, token_index_len)，它不是缓存容量，也不是已算好 KV 的长度。
        self.token_index_len = 0
        # [0, gpu_index_submitted_len) 的最新索引写入已提交，后续索引流操作可按序使用。
        # 前缀重建前由 reset 清零；它不是累计上传量，也不表示 GPU 传输已经完成。
        self.gpu_index_submitted_len = 0

    @abstractmethod
    def append_token_indexes(self, mem_indexes: torch.Tensor):
        """将 token 索引追加到当前记录的末尾，追加长度由 mem_indexes.numel() 决定。

        非空输入必须是 CPU Tensor；CPU 缓存模式直接复制到自己的常驻缓存。
        pure_gpu 按需 pin 后异步上传；已 pinned 的输入在 H2D 完成前不得改写，但可释放引用。
        pin_mem_page_cache 仅接受从页边界追加完整页，调用方保证每页内物理 token 索引连续。
        前缀替换先 reset，再按顺序追加重建。
        """
        pass

    @abstractmethod
    def get_cpu_token_indexes(self, start: int, end: int) -> torch.Tensor:
        """读取有效长度内任意 token 区间，返回可跨请求生命周期持有的独立 CPU 快照。"""
        pass

    @abstractmethod
    def get_gpu_token_indexes(self, start: int, end: int) -> torch.Tensor:
        """在索引流准备 GPU 表视图；跨请求槽位复用持有时需另取快照。"""
        pass

    @abstractmethod
    def get_upload_indexes(self) -> Optional[torch.Tensor]:
        """返回 [gpu_index_submitted_len, token_index_len) 的 pinned CPU token 索引，不更新提交进度。

        完整 CPU 缓存每次复制到新申请的 pinned 区间，页缓存按需展开到独立暂存区。
        返回值不借用常驻缓存，调用方可直接批量上传；GPU 模式返回 None。
        """
        pass

    @abstractmethod
    def get_free_token_indexes(self, start: int, end: int) -> torch.Tensor:
        """返回仅用于释放的 token 索引，可能是 CPU 或 GPU Tensor。

        两种 CPU 缓存模式返回 CPU 快照；pure_gpu 返回借用的 GPU 请求表视图。

        GPU 视图借用请求表，返回前让调用流等待索引流。重写请求槽位前，须在同一流
        按序消费，或让索引流等待消费流。此接口不提供跨槽位复用的快照语义，
        缓存和元数据仍使用 CPU getter。
        """
        pass

    @abstractmethod
    def reset(self):
        """重置有效索引长度和 GPU 提交进度，释放 CPU 缓存引用。"""
        pass

    @staticmethod
    def submit_pending_uploads(
        token_indexes: Sequence["ReqTokenIndex"],
        copy_batch: Optional[Callable[[List[torch.Tensor], List[torch.Tensor]], None]] = None,
    ) -> None:
        """统一准备待上传区间、提交拷贝并确认进度，随后让调用流承接依赖。

        copy_batch 接收 GPU 目标和 pinned CPU 源列表，必须在当前索引流提交拷贝，
        并保护异步源的生命周期；未提供时使用 PyTorch copy_。
        只有整批提交成功才推进进度；提交不表示传输完成，无新上传也需等待已有写入。
        """
        if not token_indexes:
            return
        index_stream = token_indexes[0]._index_stream
        assert all(
            token_index._index_stream == index_stream for token_index in token_indexes
        ), "All requests in a batch must use the same index stream"
        cpu_slices, gpu_slices, pending_uploads = [], [], []
        for token_index in token_indexes:
            start, end = token_index.gpu_index_submitted_len, token_index.token_index_len
            if start < end:
                source = token_index.get_upload_indexes()
                cpu_slices.append(source)
                gpu_slices.append(token_index.gpu_indexes[start:end])
                pending_uploads.append((token_index, end))
        if pending_uploads:
            with torch.cuda.stream(index_stream):
                if copy_batch is None:
                    for destination, source in zip(gpu_slices, cpu_slices):
                        destination.copy_(source, non_blocking=True)
                else:
                    copy_batch(gpu_slices, cpu_slices)
            for token_index, end in pending_uploads:
                token_index._mark_gpu_indexes_submitted(end)
        token_indexes[0]._wait_for_index_stream()

    def _mark_gpu_indexes_submitted(self, end: int) -> None:
        """只确认已成功提交的连续前缀，不把 GPU 完成状态混入长度计数。"""
        assert self.gpu_index_submitted_len <= end <= self.token_index_len
        self.gpu_index_submitted_len = end

    def _wait_for_index_stream(self) -> None:
        current_stream = torch.cuda.current_stream(self._index_stream.device)
        if current_stream != self._index_stream:
            current_stream.wait_stream(self._index_stream)

    @staticmethod
    def _alloc_return_pinned_tensor(need_size: int) -> torch.Tensor:
        """申请 int32 token 索引的 pinned 区间，slab 容量为 1M token（4 MiB）。"""
        buffer_token_num = 1024 * 1024
        if need_size == 0 or need_size > buffer_token_num:
            return torch.empty(need_size, dtype=torch.int32, device="cpu", requires_grad=False, pin_memory=True)
        # 在函数对象上惰性保存 slab 和 offset，所有索引实例共享，由推理线程串行申请。
        allocator = ReqTokenIndex._alloc_return_pinned_tensor
        buffer, start = getattr(allocator, "return_buffer_state", (None, 0))
        end = start + need_size
        if buffer is None or end > buffer_token_num:
            buffer = torch.empty(
                buffer_token_num, dtype=torch.int32, device="cpu", requires_grad=False, pin_memory=True
            )
            start = 0
            end = need_size
        # 每次只创建返回值这一个视图；offset 只前进，换新 slab 时才归零，不覆写在途 H2D 的源。
        allocator.return_buffer_state = (buffer, end)
        return buffer[start:end]


class PinMemCacheReqTokenIndex(ReqTokenIndex):
    """按请求保存普通 CPU token 缓存，上传时复制待提交区间到独立 pinned 区间。"""

    def __init__(self, infer_req: "InferReq", index_stream: torch.cuda.Stream):
        super().__init__(infer_req, index_stream)
        self.input_len = infer_req.shm_req.input_len
        self._cpu_buffer: Optional[torch.Tensor] = None

    def append_token_indexes(self, mem_indexes: torch.Tensor):
        """在 CPU 索引末尾追加，GPU 提交进度由上传流程维护。"""
        append_len = mem_indexes.numel()
        if append_len == 0:
            return
        assert mem_indexes.device.type == "cpu", "mem_indexes must be a CPU Tensor"
        start = self.token_index_len
        end = start + append_len
        self._reserve(end)
        self._cpu_buffer[start:end].copy_(mem_indexes)
        self.token_index_len = end

    def get_cpu_token_indexes(self, start: int, end: int) -> torch.Tensor:
        assert end <= self.token_index_len, "Read range exceeds recorded token indexes"
        if start == end:
            return torch.empty(0, dtype=self.gpu_indexes.dtype, device="cpu", requires_grad=False)
        return self._cpu_buffer[start:end].clone()

    def get_gpu_token_indexes(self, start: int, end: int) -> torch.Tensor:
        """在索引流提交全部待上传索引，让调用流承接依赖后返回 GPU 视图。"""
        assert end <= self.token_index_len, "Read range exceeds recorded token indexes"
        if self.gpu_index_submitted_len < end:
            self.submit_pending_uploads([self])
        else:
            self._wait_for_index_stream()
        return self.gpu_indexes[start:end]

    def get_upload_indexes(self) -> torch.Tensor:
        """每次为待上传区间创建 pinned 快照，不保留常驻缓存或更新提交进度。"""
        start, end = self.gpu_index_submitted_len, self.token_index_len
        upload_indexes = self._alloc_return_pinned_tensor(end - start)
        if start < end:
            upload_indexes.copy_(self._cpu_buffer[start:end])
        return upload_indexes

    def get_free_token_indexes(self, start: int, end: int) -> torch.Tensor:
        """返回用于释放的 CPU token 索引快照。"""
        return self.get_cpu_token_indexes(start, end)

    def reset(self):
        # 上传使用独立 pinned 暂存区，常驻缓存可直接释放，无需等待在途 H2D。
        self.token_index_len = self.gpu_index_submitted_len = 0
        self._cpu_buffer = None

    def _reserve(self, end: int):
        """将 CPU 缓存扩至至少 end 个 token，只复制已有的有效索引。"""
        old_capacity = 0 if self._cpu_buffer is None else self._cpu_buffer.numel()
        if old_capacity >= end:
            return
        if old_capacity == 0:
            # 首次按完整输入长度额外预留 2K 输出 token，同时覆盖本次写入区间。
            reserve_len = max(self.input_len + 2 * 1024, end)
        else:
            # 容量不足时在所需长度上额外预留 2K token，减少后续 decode 扩容。
            reserve_len = end + 2 * 1024
        # 按 1K token 向上对齐，最多申请到 GPU 请求表允许的长度。
        new_capacity = min(self.gpu_indexes.numel(), (reserve_len + 1023) // 1024 * 1024)
        new_buffer = torch.empty(new_capacity, dtype=self.gpu_indexes.dtype, device="cpu", requires_grad=False)
        if self.token_index_len:
            new_buffer[: self.token_index_len].copy_(self._cpu_buffer[: self.token_index_len])
        self._cpu_buffer = new_buffer


class PinMemPageCacheReqTokenIndex(PinMemCacheReqTokenIndex):
    """每个逻辑页仅保存首 token 的 pinned CPU 索引，对外仍使用 token 单位。"""

    # 同一进程使用统一的 page_size，所有请求共享首次初始化的页内偏移。
    _page_offsets: Optional[torch.Tensor] = None

    def __init__(self, infer_req: "InferReq", index_stream: torch.cuda.Stream):
        super().__init__(infer_req, index_stream)
        self.page_size = infer_req.req_manager.page_size
        if PinMemPageCacheReqTokenIndex._page_offsets is None:
            PinMemPageCacheReqTokenIndex._page_offsets = torch.arange(
                self.page_size, dtype=self.gpu_indexes.dtype, device="cpu", requires_grad=False
            )

    def append_token_indexes(self, mem_indexes: torch.Tensor):
        """从页边界追加完整页；调用方保证同一逻辑页内物理 token 索引连续。"""
        append_len = mem_indexes.numel()
        if append_len == 0:
            return
        assert mem_indexes.device.type == "cpu", "mem_indexes must be a CPU Tensor"
        start = self.token_index_len
        end = start + append_len
        assert start % self.page_size == 0, "Append start must be page-aligned"
        assert end % self.page_size == 0, "Append end must be page-aligned"
        self._reserve(end)
        # 输入从页首开始，每隔 page_size 个 token 保存一个页首索引。
        first_page = start // self.page_size
        last_page = (end + self.page_size - 1) // self.page_size
        self._cpu_buffer[first_page:last_page].copy_(mem_indexes[:: self.page_size])
        self.token_index_len = end

    def get_cpu_token_indexes(self, start: int, end: int) -> torch.Tensor:
        assert end <= self.token_index_len, "Read range exceeds recorded token indexes"
        return self._expand_token_indexes(start, end, pin_memory=False)

    def get_upload_indexes(self) -> torch.Tensor:
        return self._expand_token_indexes(self.gpu_index_submitted_len, self.token_index_len, pin_memory=True)

    def _reserve(self, end: int):
        """首次按 GPU 请求表的最大长度申请全部页索引，后续追加直接复用。"""
        if self._cpu_buffer is not None:
            return
        page_capacity = (self.gpu_indexes.numel() + self.page_size - 1) // self.page_size
        self._cpu_buffer = torch.empty(
            page_capacity, dtype=self.gpu_indexes.dtype, device="cpu", pin_memory=True, requires_grad=False
        )

    def _expand_token_indexes(self, start: int, end: int, pin_memory: bool) -> torch.Tensor:
        """展开覆盖读取区间的完整页，再截取精确 token 范围，返回值不借用压缩缓存。"""
        if start == end:
            return torch.empty(
                0, dtype=self.gpu_indexes.dtype, device="cpu", pin_memory=pin_memory, requires_grad=False
            )
        first_page = start // self.page_size
        last_page = (end + self.page_size - 1) // self.page_size
        token_num = (last_page - first_page) * self.page_size
        if pin_memory:
            # 上传使用一次性的 slab 区间，后续追加、扩容或 reset 均不会覆写在途 H2D 的源。
            token_indexes = self._alloc_return_pinned_tensor(token_num)
        else:
            token_indexes = torch.empty(token_num, dtype=self.gpu_indexes.dtype, device="cpu", requires_grad=False)
        page_starts = self._cpu_buffer[first_page:last_page]
        # 将每个 page 的首索引加上页内偏移，直接展开到 token 级别的目标缓冲区。
        torch.add(page_starts[:, None], self._page_offsets, out=token_indexes.view(-1, self.page_size))
        # 读取可从页内任意位置开始或结束，页内偏移只影响返回范围，不改变压缩页缓存。
        offset = start % self.page_size
        return token_indexes[offset : offset + end - start]


class PureGpuReqTokenIndex(ReqTokenIndex):
    """直接维护 GPU 请求表，不持有 CPU 索引缓存；CPU 读取会等待 D2H 完成。"""

    def append_token_indexes(self, mem_indexes: torch.Tensor):
        append_len = mem_indexes.numel()
        if append_len == 0:
            return
        assert mem_indexes.device.type == "cpu", "mem_indexes must be a CPU Tensor"
        # 普通 CPU 输入在这里转为独立 pinned storage；已有 pinned 输入直接复用。
        if not mem_indexes.is_pinned():
            mem_indexes = mem_indexes.pin_memory()
        start = self.token_index_len
        end = start + append_len
        # copy_ 会向 PyTorch pinned allocator 登记流使用，H2D 完成前不会回收源内存。
        # 普通 CPU 输入已形成独立 pinned 副本；直接复用的 pinned 输入在传输完成前不能改写。
        with torch.cuda.stream(self._index_stream):
            self.gpu_indexes[start:end].copy_(mem_indexes, non_blocking=True)
        self.token_index_len = end
        self._mark_gpu_indexes_submitted(end)
        self._wait_for_index_stream()

    def get_cpu_token_indexes(self, start: int, end: int) -> torch.Tensor:
        assert end <= self.token_index_len, "Read range exceeds recorded token indexes"
        # 阻塞到 D2H 完成，GPU 索引拷回后得到独立 CPU 快照。
        with torch.cuda.stream(self._index_stream):
            return self.gpu_indexes[start:end].detach().cpu()

    def get_gpu_token_indexes(self, start: int, end: int) -> torch.Tensor:
        # append 已在索引流提交写入，读取只校验范围，不推进提交进度。
        assert end <= self.gpu_index_submitted_len, "Read range exceeds submitted GPU indexes"
        self._wait_for_index_stream()
        return self.gpu_indexes[start:end]

    def get_upload_indexes(self) -> None:
        return None

    def get_free_token_indexes(self, start: int, end: int) -> torch.Tensor:
        """返回用于释放的 GPU token 索引视图，让调用流先等待索引写入。"""
        # 复用 GPU getter 的范围校验和流等待，避免跨流释放读取尚未就绪的索引。
        # allocator.free 会按页取首索引并阻塞 D2H；借用视图仍须在槽位重写前消费。
        return self.get_gpu_token_indexes(start, end)

    def reset(self):
        # 保留 GPU 行内容，统一释放流程还需在槽位重写前消费借用的索引视图。
        self.token_index_len = self.gpu_index_submitted_len = 0


def create_req_token_index(infer_req: "InferReq", index_stream: torch.cuda.Stream) -> ReqTokenIndex:
    """从请求参数选择实现；索引流由调用方显式指定，内部不选择默认流或当前流。"""
    mode = infer_req.args.req_token_index_mode
    if mode == "pin_mem_cache":
        return PinMemCacheReqTokenIndex(infer_req, index_stream)
    if mode == "pin_mem_page_cache":
        return PinMemPageCacheReqTokenIndex(infer_req, index_stream)
    if mode == "pure_gpu":
        return PureGpuReqTokenIndex(infer_req, index_stream)
    raise ValueError(
        f"Unknown request token index mode: {mode}; expected pin_mem_cache, pin_mem_page_cache or pure_gpu"
    )
