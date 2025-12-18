# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.6.3.post1/vllm/distributed/device_communicators/custom_all_reduce.py
# of the vllm-project/vllm GitHub repository.
#
# Copyright 2023 ModelTC Team
# Copyright 2023 vLLM Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import ctypes
from contextlib import contextmanager
from typing import List, Optional, Union
import os
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from lightllm.common.cuda_wrapper import CudaRTLibrary
from lightllm.utils.log_utils import init_logger
from lightllm.utils.device_utils import has_nvlink
from lightllm.utils.sgl_utils import sgl_allreduce_ops
from lightllm.utils.vllm_utils import vllm_ops

logger = init_logger(__name__)

use_vllm_custom_allreduce = os.getenv("LIGHTLLM_USE_VLLM_CUSTOM_ALLREDUCE", "0").upper() in ["ON", "TRUE", "1"]
if use_vllm_custom_allreduce:
    # Use vllm custom allreduce
    ops = vllm_ops
else:
    # Use sgl custom allreduce
    ops = sgl_allreduce_ops

if ops is not None:
    ops.meta_size()


def is_weak_contiguous(inp: torch.Tensor):
    return inp.is_contiguous() or (
        inp.storage().nbytes() - inp.storage_offset() * inp.element_size() == inp.numel() * inp.element_size()
    )


class CustomAllreduce:

    _SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]

    # max_size: max supported allreduce size
    def __init__(self, group: ProcessGroup, device: Union[int, str, torch.device], max_size=8192 * 1024) -> None:
        """
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the CustomAllreduce to. If None,
                it will be bind to f"cuda:{local_rank}".
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device, and all communicators in this group
        are in the same node.
        """
        self._IS_CAPTURING = False
        self.disabled = True

        if ops is None:
            # disable because of missing custom allreduce library
            # e.g. in a non-cuda environment
            return
        self.group = group
        assert dist.get_backend(group) != dist.Backend.NCCL, "CustomAllreduce should be attached to a non-NCCL group."

        rank = dist.get_rank(group=self.group)
        world_size = dist.get_world_size(group=self.group)
        if world_size == 1:
            # No need to initialize custom allreduce for single GPU case.
            return

        if world_size not in CustomAllreduce._SUPPORTED_WORLD_SIZES:
            logger.warning(
                "Custom allreduce is disabled due to an unsupported world"
                " size: %d. Supported world sizes: %s. To silence this "
                "warning, specify disable_custom_all_reduce=True explicitly.",
                world_size,
                str(CustomAllreduce._SUPPORTED_WORLD_SIZES),
            )
            return

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        # now `device` is a `torch.device` object
        assert isinstance(device, torch.device)
        self.device = device

        cuda_visible_devices = None
        if cuda_visible_devices:
            device_ids = list(map(int, cuda_visible_devices.split(",")))
        else:
            device_ids = list(range(torch._C._cuda_getDeviceCount()))

        physical_device_id = device_ids[device.index]
        tensor = torch.tensor([physical_device_id], dtype=torch.int, device="cpu")
        gather_list = [torch.tensor([0], dtype=torch.int, device="cpu") for _ in range(world_size)]
        dist.all_gather(gather_list, tensor, group=self.group)
        # physical_device_ids = [t.item() for t in gather_list]

        full_nvlink = has_nvlink()
        if world_size > 2 and not full_nvlink:
            logger.warning(
                "Custom allreduce is disabled because it's not supported on"
                " more than two PCIe-only GPUs. To silence this warning, "
                "specify disable_custom_all_reduce=True explicitly."
            )
            return

        self.disabled = False
        # Buffers memory are owned by this Python class and passed to C++.
        # Meta data composes of two parts: meta data for synchronization and a
        # temporary buffer for storing intermediate allreduce results.
        self.meta_ptrs = self.create_shared_buffer(ops.meta_size() + max_size, group=group)
        # This is a pre-registered IPC buffer. In eager mode, input tensors
        # are first copied into this buffer before allreduce is performed
        self.buffer_ptrs = self.create_shared_buffer(max_size, group=group)
        # This is a buffer for storing the tuples of pointers pointing to
        # IPC buffers from all ranks. Each registered tuple has size of
        # 8*world_size bytes where world_size is at most 8. Allocating 8MB
        # is enough for 131072 such tuples. The largest model I've seen only
        # needs less than 10000 of registered tuples.
        self.rank_data = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        self.max_size = max_size
        self.rank = rank
        self.world_size = world_size
        self.full_nvlink = full_nvlink
        self._ptr = ops.init_custom_ar(self.meta_ptrs, self.rank_data, rank, self.full_nvlink)
        ops.register_buffer(self._ptr, self.buffer_ptrs)

    @staticmethod
    def create_shared_buffer(size_in_bytes: int, group: Optional[ProcessGroup] = None) -> List[int]:
        """
        Creates a shared buffer and returns a list of pointers
        representing the buffer on all processes in the group.
        """
        lib = CudaRTLibrary()
        pointer = lib.cudaMalloc(size_in_bytes)
        handle = lib.cudaIpcGetMemHandle(pointer)
        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        handles = [None] * world_size
        dist.all_gather_object(handles, handle, group=group)

        pointers: List[int] = []
        for i, h in enumerate(handles):
            if i == rank:
                pointers.append(pointer.value)  # type: ignore
            else:
                pointers.append(lib.cudaIpcOpenMemHandle(h).value)  # type: ignore

        return pointers

    @staticmethod
    def free_shared_buffer(pointers: List[int], group: Optional[ProcessGroup] = None) -> None:
        rank = dist.get_rank(group=group)
        lib = CudaRTLibrary()
        lib.cudaFree(ctypes.c_void_p(pointers[rank]))

    @contextmanager
    def capture(self):
        """
        The main responsibility of this context manager is the
        `register_graph_buffers` call at the end of the context.
        It records all the buffer addresses used in the CUDA graph.
        """
        try:
            self._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False
            if not self.disabled:
                self.register_graph_buffers()

    def register_graph_buffers(self):
        handle, offset = ops.get_graph_buffer_ipc_meta(self._ptr)
        # We cannot directly use `dist.all_gather_object` here
        # because it is incompatible with `gloo` backend under inference mode.
        # see https://github.com/pytorch/pytorch/issues/126032 for details.
        all_data = [[None, None] for _ in range(dist.get_world_size(group=self.group))]
        all_data[self.rank] = [handle, offset]
        ranks = sorted(dist.get_process_group_ranks(group=self.group))
        for i, rank in enumerate(ranks):
            dist.broadcast_object_list(all_data[i], src=rank, group=self.group, device="cpu")
        # Unpack list of tuples to tuple of lists.
        handles = [d[0] for d in all_data]  # type: ignore
        offsets = [d[1] for d in all_data]  # type: ignore
        ops.register_graph_buffers(self._ptr, handles, offsets)

    def should_custom_ar(self, inp: torch.Tensor):
        if self.disabled:
            return False
        inp_size = inp.numel() * inp.element_size()
        # custom allreduce requires input byte size to be multiples of 16
        if inp_size % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        # for 4 or more non NVLink-capable GPUs, custom allreduce provides
        # little performance improvement over NCCL.
        if self.world_size == 2 or self.full_nvlink:
            return inp_size < self.max_size
        return False

    def all_reduce(self, inp: torch.Tensor, *, out: torch.Tensor = None, registered: bool = False):
        """Performs an out-of-place all reduce.

        If registered is True, this assumes inp's pointer is already
        IPC-registered. Otherwise, inp is first copied into a pre-registered
        buffer.
        """
        if out is None:
            # fix circle import
            from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager

            out = g_cache_manager.alloc_tensor(inp.shape, inp.dtype, device=inp.device)
        if registered:
            ops.all_reduce(self._ptr, inp, out, 0, 0)
        else:
            ops.all_reduce(self._ptr, inp, out, self.buffer_ptrs[self.rank], self.max_size)
        return out

    def custom_all_reduce(self, input: torch.Tensor) -> Optional[torch.Tensor]:
        """The main allreduce API that provides support for cuda graph."""
        # When custom allreduce is disabled, this will be None.
        if self.disabled or not self.should_custom_ar(input):
            return None
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self.all_reduce(input, registered=True)
            else:
                # If warm up, mimic the allocation pattern since custom
                # allreduce is out-of-place.
                # fix circle import
                from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager

                out = g_cache_manager.alloc_tensor(input.shape, input.dtype, device=input.device)
                return out
        else:
            # Note: outside of cuda graph context, custom allreduce incurs a
            # cost of cudaMemcpy, which should be small (<=1% of overall
            # latency) compared to the performance gain of using custom kernels
            return self.all_reduce(input, registered=False)

    def close(self):
        if not self.disabled and self._ptr:
            ops.dispose(self._ptr)
            self._ptr = 0
            self.free_shared_buffer(self.meta_ptrs)
            self.free_shared_buffer(self.buffer_ptrs)

    def __del__(self):
        self.close()
