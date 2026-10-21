import math
from dataclasses import dataclass
from typing import Dict, Iterable, Union

import torch


Shape = Union[int, torch.Size, Iterable[int]]


class TensorBufferManager:
    """从一块连续的底层 buffer 中申请和复用 tensor。

    ``alloc`` 返回与底层 buffer 共享 storage 的连续 tensor view；``free``
    将其占用的区间归还给管理器。每次申请的起始地址和预留空间都会按
    ``alignment_bytes`` 对齐，默认对齐到 256 bytes。

    调用方必须使用 ``free`` 释放 ``alloc`` 原样返回的 tensor。释放后不能
    继续使用该 tensor、它的别名，或者尚未完成的异步操作；本管理器不负责
    CUDA stream 间的生命周期同步。

    使用示例::

        buffer = torch.empty(1024 * 1024, dtype=torch.uint8, device="cuda")
        manager = TensorBufferManager(buffer)
        tensor = manager.alloc((128, 256), torch.bfloat16)
        manager.free(tensor)
    """

    def __init__(self, buffer: torch.Tensor, alignment_bytes: int = 256):
        """初始化连续内存池，并将可用区间调整到指定的地址对齐边界。"""
        # 底层 buffer 必须是一块不参与 autograd 的连续内存，才能安全地按字节切分。
        if not isinstance(buffer, torch.Tensor):
            raise TypeError("buffer must be a torch.Tensor")
        if not buffer.is_contiguous():
            raise ValueError("buffer must be contiguous")
        if buffer.requires_grad:
            raise ValueError("buffer must not require gradients")
        if alignment_bytes <= 0 or alignment_bytes & (alignment_bytes - 1):
            raise ValueError("alignment_bytes must be a positive power of two")

        self._alignment_bytes = alignment_bytes
        # 统一转换成一维 byte tensor，后续 offset 和 size 都使用字节作为单位。
        byte_buffer = buffer.view(torch.uint8).view(-1)

        # CPU buffer 的起始地址不一定满足指定对齐要求。构造阶段一次性切掉未对齐的
        # 前缀，使后续所有 allocation offset 都能从对齐后的地址 0 开始计算。
        address_remainder = buffer.data_ptr() % self._alignment_bytes
        if address_remainder != 0:
            aligned_offset = self._alignment_bytes - address_remainder
            byte_buffer = byte_buffer[aligned_offset:]

        assert byte_buffer.numel() > 0, "buffer has no usable bytes after alignment"
        self._byte_buffer = byte_buffer
        # 初始状态下，整个对齐后的 byte buffer 都是一个连续空闲块。
        self._free_blocks = [_FreeBlock(offset=0, size=byte_buffer.numel())]
        # 使用 data_ptr 定位 allocation，同时保存 tensor 对象用于释放时的身份校验。
        self._allocations: Dict[int, _Allocation] = {}

    def alloc(self, shape: Shape, dtype: torch.dtype) -> torch.Tensor:
        """从内存池申请指定 shape 和 dtype 的连续 tensor view。"""
        # tensor 的实际占用空间统一换算成字节数。
        shape = self._normalize_shape(shape)
        element_size = self._element_size(dtype)
        tensor_bytes = math.prod(shape) * element_size

        # 空 tensor 不占用底层 buffer，无需创建 allocation 记录。
        if tensor_bytes == 0:
            return torch.empty(shape, dtype=dtype, device=self._byte_buffer.device)

        # 实际 tensor 只使用 tensor_bytes；内存池按对齐后的 reserved_bytes 划出区间。
        reserved_bytes = self._align_up(tensor_bytes)
        offset = self._take_free_block(reserved_bytes)
        # 先截取精确字节范围，再转换为调用方需要的 dtype 和 shape。
        tensor = self._byte_buffer[offset : offset + tensor_bytes].view(dtype).view(shape)
        self._remember_allocation(tensor, offset, reserved_bytes)
        return tensor

    def free(self, tensor: torch.Tensor) -> None:
        """释放 ``alloc`` 返回的 tensor，并回收它的完整对齐预留区间。"""
        # 空 tensor 没有占用内存池空间，可以直接忽略。
        if tensor.numel() == 0:
            return

        # 先按地址查找，再校验 tensor 身份，避免旧 tensor 误释放地址相同的新 allocation。
        data_ptr = tensor.data_ptr()
        allocation = self._allocations.get(data_ptr)
        if allocation is None or allocation.tensor is not tensor:
            raise ValueError("tensor was not allocated by this manager or has already been freed")

        del self._allocations[data_ptr]
        # 归还的是 reserved_bytes，而不是 tensor 的实际字节数，确保对齐填充也被回收。
        self._free_blocks.append(_FreeBlock(allocation.offset, allocation.reserved_bytes))
        self._merge_adjacent_free_blocks()

    def _take_free_block(self, required_bytes: int) -> int:
        """使用 first-fit 策略取出一个连续空闲区间，并返回其起始 offset。"""
        for index, block in enumerate(self._free_blocks):
            if block.size < required_bytes:
                continue

            allocation_offset = block.offset
            # 大小完全相同则删除空闲块，否则从空闲块头部切出所需空间。
            if block.size == required_bytes:
                del self._free_blocks[index]
            else:
                block.offset += required_bytes
                block.size -= required_bytes
            return allocation_offset

        # 总空闲空间足够但最大连续块不足时，也会在这里报告碎片化信息。
        free_bytes = sum(block.size for block in self._free_blocks)
        largest_block = max((block.size for block in self._free_blocks), default=0)
        raise MemoryError(
            f"tensor buffer has no contiguous block for {required_bytes} bytes; "
            f"free={free_bytes} bytes, largest_free_block={largest_block} bytes"
        )

    def _merge_adjacent_free_blocks(self) -> None:
        """按 offset 排序空闲块，并将地址上相邻的区间合并。"""
        self._free_blocks.sort(key=lambda block: block.offset)
        if len(self._free_blocks) < 2:
            return

        merged_blocks = [self._free_blocks[0]]
        for current_block in self._free_blocks[1:]:
            previous_block = merged_blocks[-1]

            # 相邻区间直接合并；存在间隔则保留；发生重叠说明内部状态已损坏。
            if previous_block.end == current_block.offset:
                previous_block.size += current_block.size
            elif previous_block.end < current_block.offset:
                merged_blocks.append(current_block)
            else:
                raise RuntimeError("tensor buffer contains overlapping free blocks")

        self._free_blocks = merged_blocks

    def _remember_allocation(self, tensor: torch.Tensor, offset: int, reserved_bytes: int) -> None:
        """以 tensor 地址为 key，记录其原始对象和对应的内存池区间。"""
        self._allocations[tensor.data_ptr()] = _Allocation(tensor, offset, reserved_bytes)

    @staticmethod
    def _normalize_shape(shape: Shape) -> torch.Size:
        """将整数或整数序列统一转换成不包含负数维度的 ``torch.Size``。"""
        if isinstance(shape, int):
            shape = (shape,)
        try:
            shape = torch.Size(shape)
        except TypeError as exc:
            raise TypeError("shape must be an int or an iterable of ints") from exc
        if any(dim < 0 for dim in shape):
            raise ValueError("shape dimensions must be non-negative")
        return shape

    @staticmethod
    def _element_size(dtype: torch.dtype) -> int:
        """校验 dtype，并返回该 dtype 单个元素占用的字节数。"""
        if not isinstance(dtype, torch.dtype):
            raise TypeError("dtype must be a torch.dtype")
        return torch.empty((), dtype=dtype).element_size()

    def _align_up(self, size_bytes: int) -> int:
        """将字节数向上取整到 ``alignment_bytes`` 的整数倍。"""
        return (size_bytes + self._alignment_bytes - 1) // self._alignment_bytes * self._alignment_bytes


@dataclass
class _FreeBlock:
    """描述内存池中的一段连续空闲字节区间。"""

    offset: int
    size: int

    @property
    def end(self) -> int:
        """返回空闲区间的右边界，不包含该位置。"""
        return self.offset + self.size


@dataclass(frozen=True)
class _Allocation:
    """记录已申请 tensor 及其在内存池中的完整预留区间。"""

    tensor: torch.Tensor
    offset: int
    reserved_bytes: int
