from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from lightllm.server.router.model_infer.mode_backend import generic_pre_process
from lightllm.common.cache_coordinator import HybridCacheCoordinator
from lightllm.common.linear_state import LinearStateManager
from lightllm.common.state_cache_manager.base import StateCacheManager
from lightllm.server.router.model_infer.infer_batch import InferReq
from lightllm.server.router.model_infer.mode_backend.dp_backend import dp_shared_kv_trans


class _CheckpointPool(StateCacheManager):
    def get_state_cache(self, buffer_idx):
        raise AssertionError("This test only allocates checkpoint slots")


def _advance(tokens, conv, ssm, indexer):
    for token in tokens:
        value = torch.tensor([token, token * 2], dtype=torch.float64).reshape(2, 1, 1)
        ssm.mul_(0.5).add_(value).add_(conv.sum(-1, keepdim=True) * 0.25)
        conv.copy_(torch.cat((conv[..., 1:], value), dim=-1))
        indexer.copy_(torch.cat((indexer[:, 1:], value.square()), dim=1))


def test_dp_fetch_cpu_transport_restores_and_releases_complete_request_state(monkeypatch):
    args = SimpleNamespace(page_size=4, linear_att_hash_page_size=2, linear_att_page_block_num=2)
    state = LinearStateManager.__new__(LinearStateManager)
    state.linear_config = SimpleNamespace(get_conv_state_shape=lambda: (1, 2))
    state.mtp_step = 2
    state.mtp_state_index = torch.full((4,), 2, dtype=torch.int32)
    state.conv_state = SimpleNamespace(buffer=torch.full((2, 4, 1, 4), -10.0, dtype=torch.float64))
    state.ssm_state = SimpleNamespace(buffer=torch.full((2, 12, 1, 1), -20.0, dtype=torch.float64))
    indexer = SimpleNamespace(state=SimpleNamespace(buffer=torch.full((2, 4, 2, 1), -30.0, dtype=torch.float64)))
    coordinator = HybridCacheCoordinator(state, indexer)
    coordinator.big_page_buffers = _CheckpointPool(4)
    coordinator.small_page_buffers = _CheckpointPool(2)
    tables = [torch.arange(96, dtype=torch.int32).reshape(4, 24) for _ in range(6)]
    tables[1] = torch.full((4, 24), -1, dtype=torch.int32)
    views = [SimpleNamespace(kv_buffer=torch.full((2, 128), -100.0)) for _ in range(6)]
    peers = [
        SimpleNamespace(
            mtp_step=2,
            req_to_conv_state=SimpleNamespace(buffer=torch.full_like(state.conv_state.buffer, -40)),
            req_to_ssm_state=SimpleNamespace(buffer=torch.full_like(state.ssm_state.buffer, -50)),
            req_to_indexer_tail=SimpleNamespace(buffer=torch.full_like(indexer.state.buffer, -60)),
        )
        for _ in range(6)
    ]
    next_slot = 0
    allocated = {}

    def allocate(req, count):
        nonlocal next_slot
        indexes = torch.arange(next_slot, next_slot + count, dtype=torch.int32)
        tables[1][req.req_idx, req.hold_kv_len : req.hold_kv_len + count] = indexes
        req.hold_kv_len += count
        next_slot += count
        allocated.setdefault(req.req_idx, []).extend(indexes.tolist())
        return indexes

    requests = []
    prefixes = [4, 8]
    restored_lens = [12, 16]
    source_ranks = [3, 5]
    source_rows = [2, 3]
    for req_idx, local_len, restored_len, rank, row in zip([0, 2], prefixes, restored_lens, source_ranks, source_rows):
        tokens = list(range(1 + req_idx, restored_len + 8 + req_idx))
        shared_slot = coordinator.big_page_buffers.alloc_one_state_cache()
        coordinator.big_page_buffers.checkpoint_lens[shared_slot] = local_len
        req = SimpleNamespace(
            req_idx=req_idx,
            args=args,
            cur_kv_len=local_len,
            hold_kv_len=0,
            shared_kv_node=SimpleNamespace(
                node_prefix_total_len=local_len, big_page_buffer_idx=shared_slot, parent=None
            ),
            hybrid_cache_len=restored_len + 6,
            hybrid_len_to_big_page_id={},
            tail_small_page_buffer_id=None,
            shm_req=SimpleNamespace(shm_cur_kv_len=local_len),
            get_cur_total_len=lambda tokens=tokens: len(tokens),
            tokens=tokens,
            mm_input_spans=[],
        )
        req._kv_cache_alloc_need = lambda length, req=req: InferReq._kv_cache_alloc_need(req, length)
        local_indexes = allocate(req, local_len)
        kv = torch.tensor([tokens, [token * 2 for token in tokens]], dtype=torch.float32)
        views[1].kv_buffer[:, local_indexes] = kv[:, :local_len]
        source_indexes = tables[rank][row, :restored_len]
        views[rank].kv_buffer[:, source_indexes] = kv[:, :restored_len]
        peer = peers[rank]
        conv = peer.req_to_conv_state.buffer[:, row, ..., :2].zero_()
        ssm = peer.req_to_ssm_state.buffer[:, row * 3].zero_()
        tail = peer.req_to_indexer_tail.buffer[:, row].zero_()
        _advance(tokens[:restored_len], conv, ssm, tail)
        requests.append(req)

    pool_baselines = [
        (pool.get_used_cache_num(), pool.checkpoint_lens.copy())
        for pool in (coordinator.big_page_buffers, coordinator.small_page_buffers)
    ]
    copies = []

    def copy_kv(**kwargs):
        assert [req.cur_kv_len for req in requests] == prefixes
        assert kwargs["dp_size_in_node"] == 3 and kwargs["rank_in_dp"] == 1
        copies.append(kwargs)
        for source, dp_rank, target in zip(
            kwargs["move_token_indexes"], kwargs["token_dp_indexes"], kwargs["mem_indexes"]
        ):
            views[1].kv_buffer[:, target] = kwargs["kv_views"][int(dp_rank) * 2 + 1].kv_buffer[:, source]

    backend = SimpleNamespace(
        model=SimpleNamespace(
            mem_manager=SimpleNamespace(operator=SimpleNamespace(copy_kv_from_other_dp_ranks=copy_kv)),
            cache_coordinator=coordinator,
        ),
        dp_world_size=2,
        dp_size_in_node=3,
        rank_in_dp=1,
        node_nccl_group=None,
        is_master_in_dp=True,
        req_to_token_indexes_by_rank=tables,
        kv_views=views,
        peer_states=peers,
        logger=Mock(),
        _alloc_req_kv_mem=allocate,
    )
    module = dp_shared_kv_trans.DPKVSharedMoudle.__new__(dp_shared_kv_trans.DPKVSharedMoudle)
    module.backend = backend
    module.dp_rank_in_node = 0
    module.shared_req_infos = SimpleNamespace(arr=np.array([[[4, 0], [12, 2], [8, 1]], [[8, 2], [4, 1], [16, 3]]]))
    monkeypatch.setattr(dp_shared_kv_trans, "g_infer_context", SimpleNamespace(get_can_alloc_token_num=lambda: 128))
    monkeypatch.setattr(dp_shared_kv_trans.dist, "barrier", lambda group: None)
    tasks = module.build_shared_kv_trans_tasks(requests, req_dp_ranks=[0, 0])
    assert [task.max_kv_len_mem_manager_index for task in tasks] == source_ranks
    assert [task.source_req_idx for task in tasks] == source_rows
    tensor_to = torch.Tensor.to
    tensor = torch.tensor

    def cpu_to(value, *args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return tensor_to(value, *args, **kwargs)

    with monkeypatch.context() as transport:
        transport.setattr(torch.Tensor, "to", cpu_to)
        transport.setattr(
            dp_shared_kv_trans,
            "torch",
            SimpleNamespace(
                cat=torch.cat,
                tensor=lambda data, **kwargs: tensor(data, dtype=kwargs["dtype"]),
                int32=torch.int32,
                int64=torch.int64,
            ),
        )
        module.kv_trans(tasks)

    assert len(copies) == 1
    assert state.mtp_state_index.tolist() == [0, 2, 0, 2]
    radix = SimpleNamespace(insert=Mock(), dec_node_ref_counter=Mock())
    held_nodes = [req.shared_kv_node for req in requests]
    freed = []
    for req, local_len, restored_len in zip(requests, prefixes, restored_lens):
        assert req.cur_kv_len == req.hold_kv_len == req.shm_req.shm_cur_kv_len == restored_len
        indexes = tables[1][req.req_idx, :restored_len]
        tokens = req.tokens
        torch.testing.assert_close(
            views[1].kv_buffer[:, indexes],
            tensor([tokens[:restored_len], [x * 2 for x in tokens[:restored_len]]], dtype=torch.float32),
        )
        conv = state.conv_state.buffer[:, req.req_idx, ..., :2]
        ssm = state.ssm_state.buffer[:, req.req_idx * 3]
        tail = indexer.state.buffer[:, req.req_idx]
        cold = (torch.zeros_like(conv), torch.zeros_like(ssm), torch.zeros_like(tail))
        _advance(tokens[:restored_len], *cold)
        for actual, expected in zip((conv, ssm, tail), cold):
            torch.testing.assert_close(actual, expected)
        _advance(tokens[restored_len:], *cold)
        _advance(tokens[restored_len:], conv, ssm, tail)
        for actual, expected in zip((conv, ssm, tail), cold):
            torch.testing.assert_close(actual, expected)
        allocate(req, req._kv_cache_alloc_need(len(tokens)))
        req.cur_kv_len = len(tokens)
        big_slot = coordinator.big_page_buffers.alloc_one_state_cache()
        small_slot = coordinator.small_page_buffers.alloc_one_state_cache()
        req.hybrid_len_to_big_page_id[restored_len + 4] = big_slot
        req.tail_small_page_buffer_id = small_slot
        coordinator.big_page_buffers.checkpoint_lens[big_slot] = restored_len + 4
        coordinator.small_page_buffers.checkpoint_lens[small_slot] = req.hybrid_cache_len
        assert coordinator.get_cacheable_prefix_len(req, args) == local_len
        released = coordinator.release_request_cache(
            req, SimpleNamespace(req_to_token_indexs=tables[1]), radix, args, True
        )
        freed.extend(torch.cat(released).tolist())
        assert torch.cat(released).tolist() == allocated[req.req_idx][local_len:]
        assert not req.hybrid_len_to_big_page_id
        assert req.tail_small_page_buffer_id is req.shared_kv_node is None
    assert len(freed) == len(set(freed))
    radix.insert.assert_not_called()
    assert [call.args[0] for call in radix.dec_node_ref_counter.call_args_list] == held_nodes
    for pool, (used, checkpoint_lens) in zip(
        (coordinator.big_page_buffers, coordinator.small_page_buffers), pool_baselines
    ):
        assert pool.get_used_cache_num() == used
        assert pool.checkpoint_lens == checkpoint_lens
