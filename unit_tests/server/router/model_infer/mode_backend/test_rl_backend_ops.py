from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from lightllm.common.attention_state.allocators import FullAttKVAllocator
from lightllm.common.req_manager import ReqManager
from lightllm.common.req_manager.base import _ReqIndexPool
from lightllm.server.router.model_infer.mode_backend.rl_backend_ops import RlBackendOps
from lightllm.utils.torch_memory_saver_utils import MemoryTag


@pytest.mark.parametrize(
    ("tags", "should_init_hold_indexes"),
    [
        ([MemoryTag.WEIGHT], False),
        ([MemoryTag.GRAPH], False),
        ([], False),
        ([MemoryTag.KV_CACHE], True),
        (None, True),
    ],
)
def test_resume_memory_initializes_hold_indexes_only_for_kv_cache(
    tags, should_init_hold_indexes, kv_allocator_environment
):
    allocator = FullAttKVAllocator(16, page_size=4, pin_memory=False)
    req_manager = ReqManager.__new__(ReqManager)
    req_manager.req_list = _ReqIndexPool(2)
    req_manager.HOLD_REQUEST_ID = 2
    req_manager.page_size = allocator.page_size
    req_manager._cpu_token_index_blocks = np.empty((3, 2), dtype=np.int32)
    req_manager.req_to_token_indexs = torch.full((3, 8), -1, dtype=torch.int32)
    resumed = []

    def resume(tag):
        assert req_manager.req_to_token_indexs.eq(-1).all()
        resumed.append(tag)

    model = SimpleNamespace(
        torch_memory_saver=SimpleNamespace(resume=resume),
        req_manager=req_manager,
        att_state_manager=SimpleNamespace(full_att_allocator=allocator),
    )
    backend = SimpleNamespace(model=model, logger=Mock())
    ops = RlBackendOps(backend)
    ops._clear_cuda_cache = lambda: None

    success, message = ops.resume_memory_occupation(tags)

    assert success, message
    assert resumed == list(RlBackendOps.MEMORY_TAG_ORDER if tags is None else tags)
    expected = list(allocator.hold_token_indices) * 2 if should_init_hold_indexes else [-1] * 8
    assert req_manager.req_to_token_indexs[req_manager.HOLD_REQUEST_ID].tolist() == expected
    assert req_manager.req_to_token_indexs[:2].eq(-1).all()
    backend.logger.error.assert_not_called()
