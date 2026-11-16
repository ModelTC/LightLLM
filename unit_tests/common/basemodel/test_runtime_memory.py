from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention_state.manager import capacity_for_budget
from lightllm.common.basemodel.runtime import _profile_layout
from lightllm.common.req_manager.base import _ReqIndexPool
from lightllm.platform import DeviceRuntime


def test_runtime_budget_honors_allocator_process_limit():
    api = SimpleNamespace(
        mem_get_info=lambda device: (8000, 10000),
        get_per_process_memory_fraction=lambda device: 0.4,
        memory_reserved=lambda device: 1000,
    )
    assert DeviceRuntime("cuda:0", api).memory_info() == (3000, 4000)


def test_capacity_admission_includes_hold_and_nonlinear_runtime_payload():
    reserve = lambda tokens: 4 * tokens + 20 * min(tokens, 24)
    for budget in (0, 47, 128, 512, 800, 1024, 2048):
        capacity = capacity_for_budget(budget, 8, 4, reserve)
        if capacity:
            assert (capacity + 4) * 8 + reserve(capacity + 4) <= budget
        assert (capacity + 8) * 8 + reserve(capacity + 8) > budget


@pytest.mark.parametrize("history,query,page", [(15, 2, 16), (16, 16, 16), (63, 130, 64), (0, 1, 1)])
def test_profile_aliases_complete_readonly_pages_and_privatises_every_write(history, query, page):
    pool = _ReqIndexPool(2)
    mapping = torch.zeros((2, (history + query + page - 1) // page * page), dtype=torch.int32)
    next_token = 0

    def alloc(count):
        nonlocal next_token
        result = torch.arange(next_token, next_token + count, dtype=torch.int32)
        next_token += count
        return result

    hold = torch.arange(10000, 10000 + page, dtype=torch.int32)
    model = SimpleNamespace(
        req_manager=SimpleNamespace(alloc=pool.alloc, req_to_token_indexs=mapping),
        att_state_manager=SimpleNamespace(
            reset_bounded_att_state=lambda requests: None,
            full_att_allocator=SimpleNamespace(page_size=page, hold_token_indices=hold, alloc=alloc),
        ),
        device_runtime=SimpleNamespace(device=torch.device("cpu")),
    )
    requests = _profile_layout(model, [query, query], [history, history])
    first = history // page * page
    count = mapping.shape[1] - first
    assert next_token == 2 * count
    for req in requests:
        row = mapping[req]
        assert torch.equal(row[:first].view(-1, page), hold.expand(first // page, -1))
        assert torch.equal(row[first : first + count] % page, torch.arange(count) % page)
    writes = torch.cat([mapping[req, history : history + query] for req in requests])
    assert writes.unique().numel() == 2 * query
