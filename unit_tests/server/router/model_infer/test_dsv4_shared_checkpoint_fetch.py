from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from lightllm.server.router.model_infer.mode_backend import generic_pre_process  # noqa: F401
from lightllm.common.kv_cache_mem_manager.deepseek4_mem_manager import DeepseekV4MemoryManager
from lightllm.common.kv_cache_mem_manager.mem_manager import MemoryManager
from lightllm.common.state_cache_manager.base import StateCacheManager
from lightllm.server.router.model_infer.mode_backend.dp_backend import dp_shared_kv_trans as transfer


class SlotPool(StateCacheManager):
    def get_state_cache(self, buffer_idx):
        raise NotImplementedError


def _module(big_tokens=2048):
    module = object.__new__(transfer.DPKVSharedMoudle)
    module.backend = NS(
        args=NS(linear_att_hash_page_size=256, linear_att_page_block_num=big_tokens // 256, max_req_total_len=8192),
        node_gloo_group=object(),
        model=NS(mem_manager=NS(big_page_buffers=SlotPool(8))),
        small_page_buffers=SlotPool(8),
        radix_cache=NS(),
    )
    return module


@pytest.mark.parametrize("start,end,expected_ids", [(0, 2048, [3]), (1792, 2304, [3, 5]), (2048, 2304, [5])])
def test_source_exchanges_integer_ids_and_no_checkpoint_payload(monkeypatch, start, end, expected_ids):
    module = _module()
    node = NS(small_page_buffer_idx=5)
    module.backend.radix_cache = NS(
        get_mem_index_value_by_node=lambda node, start, end: torch.arange(start, end),
        get_big_page_ids_by_node=lambda node: [3],
    )
    sent = []
    monkeypatch.setattr(transfer.dist, "get_rank", lambda group: 1)
    monkeypatch.setattr(transfer.dist, "get_global_rank", lambda group, rank: rank)
    monkeypatch.setattr(transfer.dist, "send", lambda tensor, dst, group: sent.append(tensor.clone()))
    module._transfer_dsv4_source_data([], {7: NS(node=node, matched_len=end)}, [[(1, 7, start, end)], []])
    assert len(sent) == 2
    assert all(tensor.dtype == torch.int64 for tensor in sent)
    assert torch.equal(sent[0], torch.arange(start, end))
    assert sent[1].tolist() == expected_ids
    assert sent[1].numel() * sent[1].element_size() == len(expected_ids) * 8


@pytest.mark.parametrize("terminal", [False, True])
def test_destination_adopts_both_checkpoint_kinds_when_shared_pools_are_full(monkeypatch, terminal):
    module = _module()
    big = module.backend.model.mem_manager.big_page_buffers
    small = module.backend.small_page_buffers
    big.alloc_state_cache(8)
    small.alloc_state_cache(8)
    req = NS(
        req_id=7,
        hybrid_cache_len=2304 if terminal else 2560,
        hybrid_len_to_big_page_id={},
        tail_small_page_buffer_id=None,
    )
    task = transfer.TransTask(
        req=req, mem_indexes=torch.arange(512), max_kv_len_dp_rank=1, max_kv_len_mem_manager_index=1
    )
    payloads = iter([torch.arange(1792, 2304), torch.tensor([3, 5])])
    monkeypatch.setattr(transfer.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(transfer.dist, "get_global_rank", lambda group, rank: rank)
    monkeypatch.setattr(transfer.dist, "recv", lambda tensor, src, group: tensor.copy_(next(payloads)))
    original_to = torch.Tensor.to

    def cpu_indexes(tensor, *args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu_indexes)
    module._transfer_dsv4_source_data([task], {}, [[(1, 7, 1792, 2304)], []])
    assert big.get_free_cache_num() == small.get_free_cache_num() == 0
    assert req.hybrid_len_to_big_page_id == {2048: 3}
    assert task.terminal_small_page_buffer_id == 5
    assert req.tail_small_page_buffer_id == (5 if terminal else None)
    assert big.refs[3] == small.refs[5] == 2
    assert task.max_kv_len_mem_indexes.dtype == torch.int32
    big.free_state_cache([3])
    small.free_state_cache([5])
    assert big.refs[3] == small.refs[5] == 1
    assert big.alloc_one_state_cache() is None
    assert small.alloc_one_state_cache() is None
    big.free_state_cache([3])
    small.free_state_cache([5])
    assert big.alloc_one_state_cache() == 3
    assert small.alloc_one_state_cache() == 5


def test_transfer_plan_does_not_allocate_or_require_checkpoint_capacity(monkeypatch):
    module = _module()
    module.dp_rank_in_node = 0
    module.backend.dp_size_in_node = 2
    module.backend.dp_world_size = 1
    module.backend.rank_in_dp = 0
    module.backend.model.req_manager = NS(get_prompt_cache_page_size=lambda: 256)
    module.backend.model.mem_manager.swa_pool = NS(page_size=128)
    req = NS(
        req_id=7, cur_kv_len=0, hold_kv_len=0, get_cur_total_len=lambda: 2500, _kv_cache_alloc_need=lambda end: end
    )

    def allocate(req, count):
        req.hold_kv_len = count
        return torch.arange(count)

    module.backend._alloc_req_kv_mem = allocate
    monkeypatch.setattr(
        transfer, "g_infer_context", NS(get_can_alloc_dsv4_swa_page_num=lambda: 2, get_can_alloc_token_num=lambda: 4096)
    )
    monkeypatch.setattr(transfer.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(
        transfer.dist, "all_gather_object", lambda output, local, group: output.__setitem__(slice(None), [local, {}])
    )
    big = module.backend.model.mem_manager.big_page_buffers
    small = module.backend.small_page_buffers
    big.alloc_state_cache(8)
    small.alloc_state_cache(8)
    tasks, plan = module.build_dsv4_trans_tasks([(7, 0, None, 0)], [req], np.array([[0, 2304]]))
    assert len(tasks) == 1 and tasks[0].terminal_small_page_buffer_id is None
    assert plan == [[(1, 7, 0, 2304)], []]
    assert big.get_free_cache_num() == small.get_free_cache_num() == 0


def test_model_warmup_free_all_does_not_reset_shared_checkpoint_owners(monkeypatch):
    manager = object.__new__(DeepseekV4MemoryManager)
    manager.big_page_buffers = SlotPool(2)
    slot = manager.big_page_buffers.alloc_one_state_cache()
    manager.big_page_buffers.retain_state_cache([slot])
    calls = []
    manager.swa_page_allocator = NS(free_all=lambda: calls.append("swa"))
    monkeypatch.setattr(MemoryManager, "free_all", lambda self: calls.append("kv"))
    manager.free_all()
    assert calls == ["kv", "swa"]
    assert manager.big_page_buffers.refs[slot] == 2
    assert manager.big_page_buffers.get_free_cache_num() == 1
