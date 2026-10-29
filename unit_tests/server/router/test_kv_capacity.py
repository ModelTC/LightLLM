from __future__ import annotations

from types import SimpleNamespace

import pytest

from lightllm.common.att_resources.allocators import FullAttKVAllocator
from lightllm.common.att_resources.allocators import full_att_kv_allocator


pytestmark = pytest.mark.usefixtures("kv_allocator_environment")


@pytest.mark.parametrize(
    "tp,dp,nnodes,selected_ranks",
    [(4, 1, 1, [0]), (8, 4, 1, [0, 2, 4, 6]), (8, 4, 2, [0, 2]), (8, 8, 2, [0, 1, 2, 3]), (8, 1, 2, [0])],
)
@pytest.mark.parametrize("reader_first", [False, True])
def test_router_reads_each_local_dp_groups_allocator(
    monkeypatch: pytest.MonkeyPatch,
    tp: int,
    dp: int,
    nnodes: int,
    selected_ranks: list[int],
    reader_first: bool,
):
    monkeypatch.setattr(full_att_kv_allocator, "get_env_start_args", lambda: SimpleNamespace(tp=tp, dp=dp, nnodes=nnodes))
    reader = full_att_kv_allocator.KVCapacityReader() if reader_first else None
    allocators = []
    for rank in range(tp // nnodes):
        monkeypatch.setenv("LIGHTLLM_CURRENT_RANK_IN_NODE", str(rank))
        allocator = FullAttKVAllocator(16 + 4 * rank, 4, pin_memory=False)
        assert allocator.shared_can_use_token_num.get_value() == allocator.size
        allocator.alloc(4)
        allocators.append(allocator)
    reader = reader or full_att_kv_allocator.KVCapacityReader()
    assert len(reader.counters) == len(selected_ranks)
    for dp_rank, rank in enumerate(selected_ranks):
        allocator = allocators[rank]
        initial = allocator.size - 4
        assert reader.get_available_tokens(dp_rank) == initial
        tokens = allocator.alloc(4)
        assert reader.get_available_tokens(dp_rank) == initial - 4
        allocator.free(tokens)
        assert reader.get_available_tokens(dp_rank) == initial
        allocator.free_all()
        assert reader.get_available_tokens(dp_rank) == allocator.size


def test_failed_page_operations_leave_shared_capacity_unchanged():
    allocator = FullAttKVAllocator(16, 4, pin_memory=False)
    counter = allocator.shared_can_use_token_num
    tokens = allocator.alloc(16)
    with pytest.raises(AssertionError):
        allocator.alloc(4)
    with pytest.raises(AssertionError):
        allocator.free(tokens[:3])
    assert counter.get_value() == allocator.can_use_mem_size == 0
    allocator.free(tokens)
    assert counter.get_value() == allocator.can_use_mem_size == 16
