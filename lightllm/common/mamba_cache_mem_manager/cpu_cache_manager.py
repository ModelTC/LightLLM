from typing import List, Optional, Tuple, Union

import torch

from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class CpuMambaCacheManager:
    """CPU-offloaded Mamba state cache using pinned (or plain) CPU memory.

    Provides a stack-based slot allocator (identical pattern to MambaCacheManager)
    and async GPU<->CPU transfer helpers.
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
        # CPU buffers: shape (layer_num, size, *state_shape)
        # ------------------------------------------------------------------
        conv_buf_shape = (layer_num, size, *self.conv_state_shape)
        ssm_buf_shape = (layer_num, size, *self.ssm_state_shape)

        if shm_key_conv is not None and shm_key_ssm is not None:
            self._init_shm_buffers(shm_key_conv, shm_key_ssm, conv_buf_shape, ssm_buf_shape)
        else:
            self._init_plain_buffers(conv_buf_shape, ssm_buf_shape)

        # Lazy transfer stream
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

        # Zero the CPU buffers for freed slots (defensive, mirrors GPU MambaCacheManager)
        if isinstance(free_index, list):
            for idx in free_index:
                self.conv_state_buffer[:, idx] = 0
                self.ssm_state_buffer[:, idx] = 0
        else:
            idx_list = free_index.tolist() if hasattr(free_index, "tolist") else list(free_index)
            for idx in idx_list:
                self.conv_state_buffer[:, idx] = 0
                self.ssm_state_buffer[:, idx] = 0

    def free_all(self):
        """Reset pool and zero out buffers."""
        self.conv_state_buffer.zero_()
        self.ssm_state_buffer.zero_()
        self.mem_state[:] = torch.arange(0, self.size, dtype=torch.int32)
        self.mark_start = 0
        self.mark_end = self.size
        self.can_use_mem_size = self.size

    # ------------------------------------------------------------------
    # Async GPU <-> CPU transfers
    # ------------------------------------------------------------------

    def _get_transfer_stream(self) -> torch.cuda.Stream:
        if self._transfer_stream is None:
            self._transfer_stream = torch.cuda.Stream()
        return self._transfer_stream

    def offload_to_cpu(
        self,
        gpu_conv_state: torch.Tensor,
        gpu_ssm_state: torch.Tensor,
        gpu_buffer_indexes: Union[List[int], torch.Tensor],
        cpu_slot_indexes: Union[List[int], torch.Tensor],
    ):
        """Batch GPU -> CPU copy on a dedicated CUDA stream.

        gpu_conv_state / gpu_ssm_state have shape (layer_num, buffer_size, *state_shape).
        Uses index_select + index_copy_ to reduce CUDA kernel launches from 2*N to 2.
        """
        stream = self._get_transfer_stream()
        if not isinstance(gpu_buffer_indexes, torch.Tensor):
            gpu_buffer_indexes = torch.tensor(gpu_buffer_indexes, dtype=torch.long, device=gpu_conv_state.device)
        if not isinstance(cpu_slot_indexes, torch.Tensor):
            cpu_slot_indexes = torch.tensor(cpu_slot_indexes, dtype=torch.long)
        with torch.cuda.stream(stream):
            self.conv_state_buffer.index_copy_(
                1, cpu_slot_indexes, gpu_conv_state.index_select(1, gpu_buffer_indexes).cpu()
            )
            self.ssm_state_buffer.index_copy_(
                1, cpu_slot_indexes, gpu_ssm_state.index_select(1, gpu_buffer_indexes).cpu()
            )

    def load_to_gpu(
        self,
        gpu_conv_state: torch.Tensor,
        gpu_ssm_state: torch.Tensor,
        cpu_slot_indexes: Union[List[int], torch.Tensor],
        gpu_buffer_indexes: Union[List[int], torch.Tensor],
    ):
        """Batch CPU -> GPU copy on a dedicated CUDA stream.

        Uses index_select + index_copy_ to reduce CUDA kernel launches from 2*N to 2.
        """
        stream = self._get_transfer_stream()
        if not isinstance(cpu_slot_indexes, torch.Tensor):
            cpu_slot_indexes = torch.tensor(cpu_slot_indexes, dtype=torch.long)
        if not isinstance(gpu_buffer_indexes, torch.Tensor):
            gpu_buffer_indexes = torch.tensor(gpu_buffer_indexes, dtype=torch.long, device=gpu_conv_state.device)
        with torch.cuda.stream(stream):
            gpu_conv_state.index_copy_(
                1, gpu_buffer_indexes, self.conv_state_buffer.index_select(1, cpu_slot_indexes).cuda()
            )
            gpu_ssm_state.index_copy_(
                1, gpu_buffer_indexes, self.ssm_state_buffer.index_select(1, cpu_slot_indexes).cuda()
            )

    def sync_transfer(self):
        """Synchronize the transfer stream."""
        if self._transfer_stream is not None:
            self._transfer_stream.synchronize()

    def wait_for_pin(self):
        """Wait for async cudaHostRegister handles to complete."""
        if self._pin_handle_conv is not None:
            self._pin_handle_conv.wait()
        if self._pin_handle_ssm is not None:
            self._pin_handle_ssm.wait()
