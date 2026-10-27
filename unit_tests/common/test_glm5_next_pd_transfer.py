"""Exercise packed FP8 MLA+index KV through the current P/D transfer API."""
from types import SimpleNamespace
from unittest import SkipTest
from unittest.mock import patch

import torch

from lightllm.common.kv_cache_mem_manager.glm5_next_mem_manager import Glm5NextMemManager
from lightllm.common.kv_cache_mem_manager.fp8_glm5_next_mem_manager import FP8Glm5NextMemManager
from lightllm.common.kv_cache_mem_manager.qwen3next_mem_manager import Qwen3NextMemManager


def test_packed_fp8_transfer_roundtrip_and_replication():
    if not torch.cuda.is_available():
        raise SkipTest('requires CUDA')
    layers, rows, width, page_size = 2, 67, 716, 8
    source = torch.arange(layers*rows*width, device='cuda').remainder(251).to(torch.uint8)
    source = source.reshape(layers, rows, 1, width)
    manager = FP8Glm5NextMemManager.__new__(FP8Glm5NextMemManager)
    manager.kv_move_buffer = torch.empty((1,page_size,layers,1,width),dtype=torch.uint8,device='cuda')
    manager._buffer_mem_indexes_tensors = [torch.empty(page_size,dtype=torch.int64,pin_memory=True)]
    memories = [SimpleNamespace(kv_buffer=torch.zeros_like(source)) for _ in range(4)]
    memories[2].kv_buffer.copy_(source)
    src = [2, 7, 17, 31, 43]
    count = manager.write_mem_to_page_kv_move_buffer(
        mem_indexes=src, page_index=0, dp_index=1, mem_managers=memories, dp_world_size=2,
        start_kv_index=8192, request_kv_len=8197,
    )
    assert count == manager.kv_move_buffer.numel()*manager.kv_move_buffer.element_size()
    expected = source[:,src].clone()
    dst = [5, 9, 41, 19, 24]
    destinations = [SimpleNamespace(kv_buffer=torch.full_like(source,255)) for _ in range(4)]
    manager.read_page_kv_move_buffer_to_mem(
        mem_indexes=dst, page_index=0, dp_index=1, mem_managers=destinations, dp_world_size=2,
        start_kv_index=8192, request_kv_len=8197,
    )
    torch.cuda.synchronize()
    for index in [2,3]:
        assert torch.equal(destinations[index].kv_buffer[:,dst],expected)
        assert torch.all(destinations[index].kv_buffer[:,0] == 255)
    for index in [0,1]:
        assert torch.all(destinations[index].kv_buffer == 255)


def test_linear_state_transfer_keeps_range_and_request_metadata():
    manager = Glm5NextMemManager.__new__(Glm5NextMemManager)
    params = dict(mem_indexes=[1],page_index=0,dp_index=1,mem_managers=[],dp_world_size=2,
                  start_kv_index=8192,request_kv_len=8193,page_kind='att_state',req_idx=7)
    for name in ['write_mem_to_page_kv_move_buffer','read_page_kv_move_buffer_to_mem']:
        with patch.object(Qwen3NextMemManager,name,return_value=123) as parent:
            assert getattr(manager,name)(**params) == 123
            assert parent.call_args.args == ([1],0,1,[],2,8192,8193,'att_state',7)
