from types import SimpleNamespace

import pytest
import torch

from lightllm.common.kv_cache_mem_manager.mem_manager import MemoryManager


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for page transfer")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.uint8])
@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_read_page_reuses_index_buffer_on_copy_stream(dtype, tp_size):
    page_size, layer_num, head_dim = 32, 3, 64
    manager = MemoryManager.__new__(MemoryManager)
    manager.kv_move_buffer = torch.empty((2, page_size, layer_num, 2 * tp_size, head_dim), dtype=dtype, device="cuda")
    manager._buffer_mem_indexes_tensors = [torch.empty(page_size, dtype=torch.int64, pin_memory=True) for _ in range(2)]
    peers = [
        SimpleNamespace(kv_buffer=torch.zeros((layer_num, 96, 2, head_dim), dtype=dtype, device="cuda"))
        for _ in range(tp_size)
    ]
    stream = torch.cuda.Stream()
    for iteration, count in enumerate((5, 32, 1, 32, 5, 1)):
        page_index = iteration % 2
        indexes = torch.randperm(96)[:count].tolist()
        payload = torch.randint(0, 128, manager.kv_move_buffer[page_index].shape, device="cuda").to(dtype)
        for peer in peers:
            peer.kv_buffer.zero_()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            manager.kv_move_buffer[page_index].copy_(payload)
            manager.read_page_kv_move_buffer_to_mem(
                indexes, page_index, dp_index=0, mem_managers=peers, dp_world_size=tp_size
            )
            done = torch.cuda.Event()
            done.record(stream)
        # The service returns the page to its pool after this completion event.
        done.synchronize()
        selection = torch.tensor(indexes, dtype=torch.int64, device="cuda")
        for rank, peer in enumerate(peers):
            expected = torch.zeros_like(peer.kv_buffer)
            values = torch.stack((payload[:count, :, rank], payload[:count, :, tp_size + rank]), dim=2)
            expected.index_copy_(1, selection, values.permute(1, 0, 2, 3))
            assert torch.equal(peer.kv_buffer, expected)
