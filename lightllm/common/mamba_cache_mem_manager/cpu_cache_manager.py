from typing import List, Optional, Tuple, Union

import torch

from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class CpuMambaCacheManager:
    """CPU-offloaded Mamba state cache using pinned (or plain) CPU memory.

    Provides a stack-based slot allocator (identical pattern to MambaCacheManager)
    and per-slot GPU<->CPU transfer helpers.

    CPU buffers use shape **(size, layer_num, \\*state_shape)** — slot-first
    layout — so that ``buffer[slot_i]`` is a single contiguous block in
    memory, enabling direct DMA without intermediate staging buffers.

    Note: the GPU-side ``MambaCacheManager`` uses (layer_num, size,
    \\*state_shape) because inference kernels access per-layer slices.  The
    shape difference is handled transparently by the transfer methods.
    """

    def __init__(
        self,
        size: int,
        layer_num: int,
        conv_state_dtype: torch.dtype,
        ssm_state_dtype: torch.dtype,
        conv_kernel_size: int,
        num_linear_k_heads: int,
        num_linear_v_heads: int,
        head_linear_k_dim: int,
        head_linear_v_dim: int,
        shm_key_conv: Optional[int] = None,
        shm_key_ssm: Optional[int] = None,
    ):
        self.size = size
        self.layer_num = layer_num
        self.conv_state_dtype = conv_state_dtype
        self.ssm_state_dtype = ssm_state_dtype
        self.conv_kernel_size = conv_kernel_size
        self.num_linear_k_heads = num_linear_k_heads
        self.num_linear_v_heads = num_linear_v_heads
        self.head_linear_k_dim = head_linear_k_dim
        self.head_linear_v_dim = head_linear_v_dim

        # Derived dimensions (same formulas as MambaCacheManager)
        self.conv_dim = head_linear_k_dim * num_linear_k_heads * 2 + head_linear_v_dim * num_linear_v_heads
        self.conv_state_shape: Tuple[int, ...] = (self.conv_dim, conv_kernel_size - 1)
        self.ssm_state_shape: Tuple[int, ...] = (num_linear_v_heads, head_linear_k_dim, head_linear_v_dim)

        # ------------------------------------------------------------------
        # Stack-based slot allocator
        # ------------------------------------------------------------------
        self.mem_state = torch.arange(0, size, dtype=torch.int32, device="cpu", requires_grad=False)
        self.mark_start = 0
        self.mark_end = size
        self.can_use_mem_size = size

        # ------------------------------------------------------------------
        # CPU buffers: shape (size, layer_num, *state_shape)
        # Slot-first so buffer[slot_i] is contiguous for direct DMA.
        # ------------------------------------------------------------------
        conv_buf_shape = (size, layer_num, *self.conv_state_shape)
        ssm_buf_shape = (size, layer_num, *self.ssm_state_shape)

        if shm_key_conv is not None and shm_key_ssm is not None:
            self._init_shm_buffers(shm_key_conv, shm_key_ssm, conv_buf_shape, ssm_buf_shape)
        else:
            self._init_plain_buffers(conv_buf_shape, ssm_buf_shape)

        # Lazy transfer stream for async offload
        self._transfer_stream: Optional[torch.cuda.Stream] = None

    # ------------------------------------------------------------------
    # Buffer initialisation helpers
    # ------------------------------------------------------------------

    def _init_shm_buffers(
        self,
        shm_key_conv: int,
        shm_key_ssm: int,
        conv_buf_shape: Tuple[int, ...],
        ssm_buf_shape: Tuple[int, ...],
    ):
        """Create CPU buffers backed by SysV shared memory + cudaHostRegister."""
        from lightllm.common.cpu_cache.creator import CpuCacheCreator, CpuCacheTensorSpec
        import numpy as np

        conv_size = int(np.prod(conv_buf_shape)) * torch._utils._element_size(self.conv_state_dtype)
        ssm_size = int(np.prod(ssm_buf_shape)) * torch._utils._element_size(self.ssm_state_dtype)

        conv_spec = CpuCacheTensorSpec(
            shm_key=shm_key_conv, shape=conv_buf_shape, dtype=self.conv_state_dtype, size_bytes=conv_size
        )
        ssm_spec = CpuCacheTensorSpec(
            shm_key=shm_key_ssm, shape=ssm_buf_shape, dtype=self.ssm_state_dtype, size_bytes=ssm_size
        )

        conv_creator = CpuCacheCreator(conv_spec)
        ssm_creator = CpuCacheCreator(ssm_spec)

        self.conv_state_buffer, self._pin_handle_conv = conv_creator.create_or_attach(
            init_shm_data=True, pin=True, pin_no_blocking=True
        )
        self.ssm_state_buffer, self._pin_handle_ssm = ssm_creator.create_or_attach(
            init_shm_data=True, pin=True, pin_no_blocking=True
        )

    def _init_plain_buffers(
        self,
        conv_buf_shape: Tuple[int, ...],
        ssm_buf_shape: Tuple[int, ...],
    ):
        """Create plain CPU tensors (optionally pinned if CUDA is available)."""
        self._pin_handle_conv = None
        self._pin_handle_ssm = None

        conv_buf = torch.zeros(conv_buf_shape, dtype=self.conv_state_dtype, device="cpu")
        ssm_buf = torch.zeros(ssm_buf_shape, dtype=self.ssm_state_dtype, device="cpu")

        try:
            self.conv_state_buffer = conv_buf.pin_memory()
            self.ssm_state_buffer = ssm_buf.pin_memory()
        except Exception:
            # pin_memory() may fail without CUDA; fall back to plain CPU tensors
            self.conv_state_buffer = conv_buf
            self.ssm_state_buffer = ssm_buf

    # ------------------------------------------------------------------
    # Slot allocator
    # ------------------------------------------------------------------

    def alloc(self, need_size: int) -> torch.Tensor:
        """Allocate *need_size* slots; returns a CPU int32 tensor of slot indices."""
        if need_size > self.mark_end - self.mark_start:
            logger.error(f"no enough cpu mamba cache: need_size={need_size}, left={self.can_use_mem_size}")
            assert False, "error alloc cpu mamba state"

        start = self.mark_start
        end = start + need_size
        self.mark_start = end
        self.can_use_mem_size -= need_size

        return self.mem_state[start:end].clone()

    def free(self, free_index: Union[torch.Tensor, List[int]]):
        """Return slots back to the pool."""
        if isinstance(free_index, list):
            free_len = len(free_index)
        else:
            free_len = free_index.numel()

        if free_len == 0:
            return

        end = self.mark_start
        start = end - free_len
        assert start >= 0, f"error free cpu state: mark_start={self.mark_start}, free_len={free_len}"

        if isinstance(free_index, list):
            self.mem_state[start:end] = torch.tensor(free_index, dtype=torch.int32)
        elif free_index.device.type == "cpu":
            self.mem_state[start:end] = free_index.to(dtype=torch.int32)
        else:
            self.mem_state[start:end] = free_index.cpu().to(dtype=torch.int32)

        self.mark_start -= free_len
        self.can_use_mem_size += free_len

        # Zero freed slots — buffer[slot] is contiguous, fast.
        if isinstance(free_index, list):
            for idx in free_index:
                self.conv_state_buffer[idx] = 0
                self.ssm_state_buffer[idx] = 0
        else:
            idx_list = free_index.tolist() if hasattr(free_index, "tolist") else list(free_index)
            for idx in idx_list:
                self.conv_state_buffer[idx] = 0
                self.ssm_state_buffer[idx] = 0

    def free_all(self):
        """Reset pool and zero out buffers."""
        self.conv_state_buffer.zero_()
        self.ssm_state_buffer.zero_()
        self.mem_state[:] = torch.arange(0, self.size, dtype=torch.int32)
        self.mark_start = 0
        self.mark_end = self.size
        self.can_use_mem_size = self.size

    # ------------------------------------------------------------------
    # GPU <-> CPU transfers
    # ------------------------------------------------------------------

    def set_gpu_buffers(self, gpu_conv_buffer: torch.Tensor, gpu_ssm_buffer: torch.Tensor):
        """Bind GPU-side cache buffers (call once after MambaCacheManager is created).

        After binding, offload_to_cpu / load_to_gpu only need slot indexes.
        """
        self._gpu_conv = gpu_conv_buffer
        self._gpu_ssm = gpu_ssm_buffer

    def _get_transfer_stream(self) -> torch.cuda.Stream:
        if self._transfer_stream is None:
            self._transfer_stream = torch.cuda.Stream()
        return self._transfer_stream

    def _to_cpu_long(self, idx: Union[List[int], torch.Tensor]) -> torch.Tensor:
        if isinstance(idx, list):
            return torch.tensor(idx, dtype=torch.long)
        return idx.to(dtype=torch.long, device="cpu")

    def _to_gpu_long(self, idx: Union[List[int], torch.Tensor]) -> torch.Tensor:
        if isinstance(idx, list):
            return torch.tensor(idx, dtype=torch.long, device=self._gpu_conv.device)
        return idx.to(dtype=torch.long, device=self._gpu_conv.device)

    def offload_to_cpu(
        self,
        gpu_buffer_indexes: Union[List[int], torch.Tensor],
        cpu_slot_indexes: Union[List[int], torch.Tensor],
        direct: bool = False,
    ):
        """Per-slot GPU -> CPU transfer.

        **direct=False** (default, source buffer remains live for compute):
            Creates contiguous snapshots on the compute stream, then async
            DMA to CPU on the transfer stream.  Safe when subsequent compute
            kernels will modify the source buffer in-place.

        **direct=True** (source buffer will be freed / reallocated after call):
            Async DMA on the transfer stream, then a GPU-side ``wait_event``
            on the default stream.  The host returns immediately; the default
            stream only stalls if it catches up before the DMA finishes — so
            the GPU slot can be freed, and the actual wait only happens when
            the slot is about to be zeroed or reused.
        """
        gpu_buffer_indexes = self._to_gpu_long(gpu_buffer_indexes)
        cpu_slot_indexes = self._to_cpu_long(cpu_slot_indexes)

        n = int(gpu_buffer_indexes.shape[0])
        if n == 0:
            return

        # Both paths use the transfer stream; they differ only in whether
        # the DMA source is a snapshot or the live GPU buffer.
        stream = self._get_transfer_stream()

        if direct:
            # No snapshot needed — buffer won't be modified by compute after
            # this call.  Record event to order DMA after prior compute.
            compute_done = torch.cuda.current_stream().record_event()

            with torch.cuda.stream(stream):
                stream.wait_event(compute_done)
                for i in range(n):
                    g = gpu_buffer_indexes[i]
                    c = int(cpu_slot_indexes[i])
                    self.conv_state_buffer[c].copy_(self._gpu_conv[:, g], non_blocking=True)
                    self.ssm_state_buffer[c].copy_(self._gpu_ssm[:, g], non_blocking=True)
                dma_done = stream.record_event()

            # GPU-side barrier: the default stream won't execute past this
            # point until the DMA finishes.  The host returns immediately.
            # In practice the DMA completes during subsequent scheduling /
            # bookkeeping, so the wait is usually a no-op.
            torch.cuda.current_stream().wait_event(dma_done)
        else:
            # Snapshot on compute stream, then async DMA on transfer stream.
            snapshots_conv = []
            snapshots_ssm = []
            for i in range(n):
                g = gpu_buffer_indexes[i]
                snapshots_conv.append(self._gpu_conv[:, g].contiguous())
                snapshots_ssm.append(self._gpu_ssm[:, g].contiguous())

            event = torch.cuda.current_stream().record_event()

            with torch.cuda.stream(stream):
                stream.wait_event(event)
                for i in range(n):
                    c = int(cpu_slot_indexes[i])
                    self.conv_state_buffer[c].copy_(snapshots_conv[i], non_blocking=True)
                    snapshots_conv[i].record_stream(stream)
                    self.ssm_state_buffer[c].copy_(snapshots_ssm[i], non_blocking=True)
                    snapshots_ssm[i].record_stream(stream)

    def load_to_gpu(
        self,
        cpu_slot_indexes: Union[List[int], torch.Tensor],
        gpu_buffer_indexes: Union[List[int], torch.Tensor],
    ):
        """Strictly synchronous per-slot CPU -> GPU transfer on the default stream.

        When this method returns the GPU buffers are immediately safe for
        in-place modification by subsequent compute kernels (guaranteed by
        CUDA stream ordering on the default stream).
        """
        cpu_slot_indexes = self._to_cpu_long(cpu_slot_indexes)
        gpu_buffer_indexes = self._to_gpu_long(gpu_buffer_indexes)

        n = int(cpu_slot_indexes.shape[0])
        if n == 0:
            return

        # Ensure any pending offload DMA is complete so CPU buffer is up-to-date.
        self.sync_transfer()

        # Per-slot DMA on the default (compute) stream.
        # cpu_buffer[c] is contiguous pinned — efficient DMA source.
        for i in range(n):
            c = int(cpu_slot_indexes[i])
            g = gpu_buffer_indexes[i]
            self._gpu_conv[:, g].copy_(self.conv_state_buffer[c])
            self._gpu_ssm[:, g].copy_(self.ssm_state_buffer[c])

    def sync_transfer(self):
        """Block until any pending async offload DMA completes."""
        if self._transfer_stream is not None:
            self._transfer_stream.synchronize()

    def wait_for_pin(self):
        """Wait for async cudaHostRegister handles to complete."""
        if self._pin_handle_conv is not None:
            self._pin_handle_conv.wait()
        if self._pin_handle_ssm is not None:
            self._pin_handle_ssm.wait()
