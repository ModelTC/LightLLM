from types import SimpleNamespace

import torch
import pytest

# 沿用生产代码的模块加载顺序，避免独立运行本测试文件时形成循环导入。
from lightllm.server.router.model_infer.mode_backend import generic_pre_process  # noqa: F401
from lightllm.server.router.model_infer.infer_batch import InferReq
from lightllm.server.router.model_infer.mode_backend.diverse_backend.impl import DiversehBackend
from lightllm.server.router.model_infer.mode_backend.diverse_backend import impl as diverse_impl
from unit_tests.common.cpu_index_helpers import make_cpu_index_manager, bind_cpu_indexes


class _FakeRadixCache:
    def __init__(self):
        self.dec_nodes = []
        self.add_nodes = []

    def dec_node_ref_counter(self, node):
        self.dec_nodes.append(node)

    def add_node_ref_counter(self, node):
        self.add_nodes.append(node)


def test_diverse_copy_skips_slaves_waiting_for_overlap_cleanup():
    active_slave = SimpleNamespace(filter_mark=False)
    filtered_slave = SimpleNamespace(filter_mark=True)
    master_req = SimpleNamespace(slave_reqs=[active_slave, filtered_slave])
    backend = DiversehBackend.__new__(DiversehBackend)

    batch_idx, run_reqs = backend._diverse_copy(
        master_reqs=[master_req],
        b_prefill_has_out=[True],
    )

    assert batch_idx == [0, 0]
    assert run_reqs == [master_req, active_slave]


def test_copy_master_kv_updates_slave_hold_kv_len():
    token_indexes_cpu = torch.full((2, 8), -1, dtype=torch.int32)
    token_indexes_cpu[0, :4] = torch.tensor([10, 11, 12, 13], dtype=torch.int32)
    old_shared_node = object()
    new_shared_node = object()
    master_req = SimpleNamespace(
        req_idx=0,
        cur_kv_len=4,
        hold_kv_len=4,
        cur_output_len=1,
        shared_kv_node=new_shared_node,
    )
    slave_req = SimpleNamespace(
        req_idx=1,
        cur_kv_len=2,
        hold_kv_len=2,
        cur_output_len=0,
        shared_kv_node=old_shared_node,
        related_master_req=master_req,
        shm_req=SimpleNamespace(input_len=4),
        args=SimpleNamespace(page_size=1),
    )
    backend = DiversehBackend.__new__(DiversehBackend)
    backend.model = SimpleNamespace(
        req_manager=make_cpu_index_manager(token_table=token_indexes_cpu, page_size=1),
    )
    backend.radix_cache = _FakeRadixCache()
    backend.is_master_in_dp = False

    bind_cpu_indexes(master_req, backend.model.req_manager)
    bind_cpu_indexes(slave_req, backend.model.req_manager, gpu_index_submitted_len=2)
    backend._copy_master_req_to_slave_req(slave_req)

    assert slave_req.token_index.get_cpu_token_indexes(0, 4).tolist() == [10, 11, 12, 13]
    assert slave_req.shared_kv_node is new_shared_node
    assert slave_req.cur_kv_len == 4
    assert slave_req.hold_kv_len == 4
    assert slave_req.token_index.token_index_len == 4
    assert slave_req.token_index.gpu_index_submitted_len == 0
    assert InferReq._kv_cache_alloc_need(slave_req, target_kv_len=5) == 1
    assert backend.radix_cache.dec_nodes == [old_shared_node]
    assert backend.radix_cache.add_nodes == [new_shared_node]


def test_master_shared_prefix_rebuild_preserves_reserved_token_indexes():
    token_indexes_cpu = torch.tensor([[10, 11, 12, 13, 20, 21, 22, 23]], dtype=torch.int32)
    shared_indexes = torch.tensor([100, 101, 102, 103], dtype=torch.int32)
    new_shared_node = SimpleNamespace(node_prefix_total_len=4)
    master_req = SimpleNamespace(
        req_idx=0,
        cur_kv_len=4,
        hold_kv_len=8,
        shared_kv_node=None,
        get_input_token_ids=lambda: [1, 2, 3, 4],
        args=SimpleNamespace(page_size=4),
    )
    freed_indexes = []

    class RadixCache(_FakeRadixCache):
        def insert(self, key, value):
            assert key.tolist() == [1, 2, 3, 4]
            assert value.tolist() == [10, 11, 12, 13]
            return 4, new_shared_node

        def match_prefix(self, key, update_refs):
            assert not update_refs
            return new_shared_node, 4, shared_indexes

    manager = make_cpu_index_manager(token_table=token_indexes_cpu, page_size=4)
    bind_cpu_indexes(master_req, manager, gpu_index_submitted_len=8)
    original_buffer = master_req.token_index._cpu_buffer
    backend = DiversehBackend.__new__(DiversehBackend)
    backend.model = SimpleNamespace(mem_manager=SimpleNamespace(free=lambda indexes: freed_indexes.append(indexes)))
    backend.radix_cache = RadixCache()

    backend._master_req_to_radix_cache(master_req)

    assert master_req.token_index.get_cpu_token_indexes(0, 8).tolist() == [100, 101, 102, 103, 20, 21, 22, 23]
    assert master_req.token_index._cpu_buffer is not original_buffer
    assert master_req.token_index.token_index_len == master_req.hold_kv_len == 8
    assert master_req.token_index.gpu_index_submitted_len == 0
    assert InferReq._kv_cache_alloc_need(master_req, target_kv_len=8) == 0
    assert InferReq._kv_cache_alloc_need(master_req, target_kv_len=9) == 4
    assert [indexes.tolist() for indexes in freed_indexes] == [[10, 11, 12, 13]]
    assert master_req.shared_kv_node is new_shared_node


@pytest.mark.parametrize("is_chuncked_mode", [False, True])
def test_diverse_waits_for_model_before_rewriting_shared_prefix(monkeypatch, is_chuncked_mode):
    order = []
    overlap_stream = object()

    class DefaultStream:
        def wait_stream(self, source):
            assert source is overlap_stream
            order.append("wait_model")

        def synchronize(self):
            order.append("synchronize")

    default_stream = DefaultStream()
    monkeypatch.setattr(torch.cuda, "default_stream", lambda: default_stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: default_stream)
    monkeypatch.setattr(diverse_impl.g_infer_context, "get_overlap_stream", lambda: overlap_stream)
    master = SimpleNamespace(
        is_master_req=lambda: True,
        get_chuncked_input_token_len=lambda: 4,
        get_cur_total_len=lambda: 4,
        cur_output_len=0,
        slave_reqs=[object()],
    )
    backend = DiversehBackend.__new__(DiversehBackend)
    backend.is_master_in_dp = False
    backend._master_req_to_radix_cache = lambda master_req: order.append("rewrite_indexes")

    packs = backend._diverse_pre_post_handle([master], is_chuncked_mode)

    assert order == ["wait_model", "rewrite_indexes", "synchronize"]
    assert packs[0].req_obj is master
    assert packs[0].output_len == 1
