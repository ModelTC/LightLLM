import pytest
import torch

from lightllm.utils.tensor_buffer_manager import TensorBufferManager


def _aligned_buffer(size: int) -> torch.Tensor:
    storage = torch.empty(size + 255, dtype=torch.uint8)
    aligned_offset = -storage.data_ptr() % 256
    return storage[aligned_offset : aligned_offset + size]


def test_allocate_different_shapes_and_dtypes_from_one_buffer():
    buffer = torch.empty(1024, dtype=torch.uint8)
    manager = TensorBufferManager(buffer)

    fp32_tensor = manager.alloc((3, 5), torch.float32)
    int16_tensor = manager.alloc((7,), torch.int16)

    assert fp32_tensor.shape == (3, 5)
    assert fp32_tensor.dtype == torch.float32
    assert fp32_tensor.is_contiguous()
    assert int16_tensor.shape == (7,)
    assert int16_tensor.dtype == torch.int16
    assert fp32_tensor.untyped_storage().data_ptr() == buffer.untyped_storage().data_ptr()
    assert int16_tensor.untyped_storage().data_ptr() == buffer.untyped_storage().data_ptr()
    assert fp32_tensor.data_ptr() % 256 == 0
    assert int16_tensor.data_ptr() % 256 == 0


def test_non_byte_backing_tensor_is_used_as_byte_storage():
    buffer = torch.empty(256, dtype=torch.float32)
    manager = TensorBufferManager(buffer)
    tensor = manager.alloc((96,), torch.int64)

    assert tensor.untyped_storage().data_ptr() == buffer.untyped_storage().data_ptr()
    assert tensor.nbytes == 96 * torch.int64.itemsize


def test_released_block_is_reused():
    manager = TensorBufferManager(_aligned_buffer(512))
    first = manager.alloc((16,), torch.float32)
    first_ptr = first.data_ptr()

    manager.free(first)
    replacement = manager.alloc((8, 2), torch.float32)

    assert replacement.data_ptr() == first_ptr


def test_adjacent_free_blocks_are_merged():
    manager = TensorBufferManager(_aligned_buffer(1024))
    first = manager.alloc((32,), torch.uint8)
    second = manager.alloc((32,), torch.uint8)
    third = manager.alloc((32,), torch.uint8)
    first_ptr = first.data_ptr()

    manager.free(second)
    manager.free(first)
    merged = manager.alloc((512,), torch.uint8)

    assert merged.data_ptr() == first_ptr
    manager.free(merged)
    manager.free(third)
    assert manager.alloc((768,), torch.uint8).data_ptr() == first_ptr


def test_release_rejects_unknown_and_already_released_tensors():
    manager = TensorBufferManager(_aligned_buffer(256))
    tensor = manager.alloc((8,), torch.float32)
    manager.free(tensor)

    with pytest.raises(ValueError, match="already been freed"):
        manager.free(tensor)
    with pytest.raises(ValueError, match="not allocated"):
        manager.free(torch.empty(1))


def test_stale_tensor_cannot_release_reused_address():
    manager = TensorBufferManager(_aligned_buffer(256))
    stale_tensor = manager.alloc((8,), torch.float32)
    manager.free(stale_tensor)
    current_tensor = manager.alloc((8,), torch.float32)

    assert stale_tensor.data_ptr() == current_tensor.data_ptr()
    with pytest.raises(ValueError, match="already been freed"):
        manager.free(stale_tensor)

    manager.free(current_tensor)


def test_allocation_reports_fragmentation_on_failure():
    manager = TensorBufferManager(_aligned_buffer(1024))
    first = manager.alloc((256,), torch.uint8)
    middle = manager.alloc((512,), torch.uint8)
    last = manager.alloc((256,), torch.uint8)
    manager.free(first)
    manager.free(last)

    with pytest.raises(MemoryError, match="largest_free_block=256 bytes"):
        manager.alloc((384,), torch.uint8)

    manager.free(middle)


def test_invalid_buffer_is_rejected():
    with pytest.raises(ValueError, match="contiguous"):
        TensorBufferManager(torch.empty((4, 4)).t())
    with pytest.raises(ValueError, match="positive power of two"):
        TensorBufferManager(torch.empty(256, dtype=torch.uint8), alignment_bytes=0)
    with pytest.raises(ValueError, match="positive power of two"):
        TensorBufferManager(torch.empty(256, dtype=torch.uint8), alignment_bytes=3)
    with pytest.raises(AssertionError, match="no usable bytes"):
        TensorBufferManager(torch.empty(0, dtype=torch.uint8))


def test_unaligned_buffer_prefix_is_skipped():
    buffer = torch.empty(512, dtype=torch.uint8)[1:]
    manager = TensorBufferManager(buffer)
    tensor = manager.alloc((128,), torch.uint8)

    assert tensor.data_ptr() % 256 == 0
    assert tensor.untyped_storage().data_ptr() == buffer.untyped_storage().data_ptr()


def test_empty_tensor_does_not_consume_buffer_space():
    manager = TensorBufferManager(_aligned_buffer(512))
    tensor = manager.alloc((0, 4), torch.float16)
    full_buffer = manager.alloc((256,), torch.uint8)

    assert tensor.shape == (0, 4)
    manager.free(tensor)
    manager.free(full_buffer)
