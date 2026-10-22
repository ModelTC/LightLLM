from types import SimpleNamespace

import torch

# 沿用生产代码的模块加载顺序，避免独立运行本测试文件时形成循环导入。
from lightllm.server.router.model_infer.mode_backend import generic_pre_process  # noqa: F401
from lightllm.server.router.model_infer.infer_batch import InferReq
from lightllm.server.router.model_infer.mode_backend.diverse_backend.impl import DiversehBackend


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
    req_to_token_indexs = torch.full((2, 8), -1, dtype=torch.int32)
    req_to_token_indexs[0, :4] = torch.tensor([10, 11, 12, 13], dtype=torch.int32)
    old_shared_node = object()
    new_shared_node = object()
    master_req = SimpleNamespace(
        req_idx=0,
        cur_kv_len=4,
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
        req_manager=SimpleNamespace(req_to_token_indexs=req_to_token_indexs),
    )
    backend.radix_cache = _FakeRadixCache()
    backend.is_master_in_dp = False

    backend._copy_master_req_to_slave_req(slave_req)

    assert req_to_token_indexs[1, :4].tolist() == [10, 11, 12, 13]
    assert slave_req.shared_kv_node is new_shared_node
    assert slave_req.cur_kv_len == 4
    assert slave_req.hold_kv_len == 4
    assert InferReq._kv_cache_alloc_need(slave_req, target_kv_len=5) == 1
    assert backend.radix_cache.dec_nodes == [old_shared_node]
    assert backend.radix_cache.add_nodes == [new_shared_node]


def test_diverse_prefill_counts_increment_once_and_slaves_do_not_duplicate(monkeypatch):
    from unittest.mock import Mock

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: Mock())
    backend = DiversehBackend.__new__(DiversehBackend)
    backend.is_master_in_dp = True
    backend._master_req_to_radix_cache = Mock()
    backend._copy_master_req_to_slave_req = Mock()
    target = [512]
    master = SimpleNamespace(
        is_master_req=lambda: True,
        get_chuncked_input_token_len=lambda: target[0],
        get_cur_total_len=lambda: 800,
        cur_kv_len=0,
        cur_output_len=0,
        slave_reqs=[],
        shm_req=SimpleNamespace(input_len=800, group_req_id=8),
    )
    first = backend._diverse_pre_post_handle([master], True)
    assert first[0].prefill_tokens == 512 and first[0].output_len == 0
    target[0] = 800
    slave = SimpleNamespace(
        is_master_req=lambda: False, remove_master_req=Mock(), shm_req=SimpleNamespace(group_req_id=8)
    )
    master.slave_reqs = [slave]
    second = backend._diverse_pre_post_handle([master, slave], True)
    assert [pack.prefill_tokens for pack in second] == [288, 0]
    assert sum(pack.prefill_tokens for pack in first + second) == 800
    slave.remove_master_req.assert_called_once()
    backend._copy_master_req_to_slave_req.assert_called_once_with(slave_req=slave)
    target[0] = 801
    master.slave_reqs = []
    assert backend._diverse_pre_post_handle([master], True)[0].prefill_tokens == 0
