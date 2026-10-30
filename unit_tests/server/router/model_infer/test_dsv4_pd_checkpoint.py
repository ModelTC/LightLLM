"""PD continuation ownership, compact wire pages, and WRITE recovery contracts."""

import copy
import threading
import uuid
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from lightllm.common.state_cache_manager.deepseek4 import DeepseekV4StateCacheManager
from lightllm.server.pd_io_struct import PDAgentMetadata, PDChunckedTransTaskGroup
from lightllm.server.router.model_infer.mode_backend.pd.decode_node_impl import decode_impl
from lightllm.server.router.model_infer.mode_backend.pd.prefill_node_impl.prefill_impl import (
    PDChunkedPrefillForPrefillNode,
)
from lightllm.server.router.model_infer.mode_backend.pd.nixl_kv_transporter import NixlKVTransporter
from unit_tests.common.test_deepseek4_paged_cache import cache as _cache, _reserve, _checkpoint_req
from unit_tests.server.router.model_infer.mode_backend.test_pd_write_quiescence import _task


cache = _cache


@pytest.mark.parametrize("length", [1, 255, 256, 257, 511, 512, 513, 1024])
def test_pd_kv_transmits_only_compact_payload(cache, length):
    manager, requests = cache
    source, src = _reserve(manager, requests, 1024)
    destination, dst = _reserve(manager, requests, 1024)
    for pool in manager.kv_buffer:
        pool.random_(0, 256)
    expected = [pool[src.long()].clone() for pool in manager.kv_buffer]
    manager.alloc_paged_kv_move_buffer(1, 1024)
    manager.kv_move_buffer.fill_(221)
    nbytes = manager.write_mem_to_page_kv_move_buffer(src[:length].tolist(), 0, 0, [manager], 1)
    copied = ((length + 255) // 256) * 256
    assert nbytes == copied * manager.get_cell_size()
    assert manager.kv_move_buffer.flatten()[nbytes:].eq(221).all()
    # The receiver has exactly the wire bytes, not a full source staging page.
    wire = manager.kv_move_buffer.flatten()[:nbytes].clone()
    manager.kv_move_buffer.zero_()
    manager.kv_move_buffer.flatten()[:nbytes].copy_(wire)
    manager.read_page_kv_move_buffer_to_mem(dst[:length].tolist(), 0, 0, [manager], 1)
    for pool, reference in zip(manager.kv_buffer, expected):
        torch.testing.assert_close(pool[dst[:copied].long()], reference[:copied], rtol=0, atol=0)
    requests.free([source, destination], torch.cat((src, dst)))


@pytest.fixture(scope="module")
def checkpoint_mappings():
    # CUDA registration caches own these addresses for the lifetime of the process.
    # Keep attached views alive across parameter cases rather than reusing their VA.
    mappings = []
    yield mappings


@pytest.mark.parametrize("tail", [0, 1, 127, 255, 511])
def test_pd_checkpoint_survives_final_ring_overwrite(cache, tail, checkpoint_mappings):
    from lightllm.models.deepseek_v4.triton_kernel.swa_cache import gather_swa_state

    manager, requests = cache
    source, src = _reserve(manager, requests, 1024 + tail)
    destination, dst = _reserve(manager, requests, 1024 + tail)
    states = DeepseekV4StateCacheManager(
        2, manager.cache_config, shm_name=f"pd_checkpoint_{uuid.uuid4().hex}", shm_create=True
    )
    source_id, destination_id = states.alloc_state_cache(2)
    names = ("req_to_swa", "req_to_c4_state", "req_to_c4_indexer_state", "req_to_c128_state")
    for name in names:
        buffer = getattr(requests, name)
        buffer[:, source].random_(0, 16)
        setattr(manager, name, buffer)
    requests.save_state(_checkpoint_req(source, 1024), source_id, states)
    torch.cuda.synchronize()
    checkpoint = states.buffer[source_id].clone()
    # An unrelated final runtime, e.g. a tail longer than the SWA ring.
    for name in names:
        getattr(requests, name)[:, source].fill_(31)
    manager.pd_checkpoint_metas = {"small": (states.layer_cache.shm.name, states.size, states.buffer.shape[-1])}
    manager.alloc_paged_kv_move_buffer(1, 1024)
    checkpoint_mappings.append((states, manager.att_state_page_helper))
    nbytes = manager.write_mem_to_page_kv_move_buffer(
        [], 0, 0, [manager], 1, "att_state", source, checkpoint=("small", source_id)
    )
    runtime_bytes = sum(getattr(requests, name)[:, source].nbytes for name in names)
    assert nbytes == runtime_bytes + checkpoint.numel()
    manager.read_page_kv_move_buffer_to_mem(
        [], 0, 0, [manager], 1, "att_state", destination, checkpoint=("small", destination_id)
    )
    torch.cuda.synchronize()
    for name in names:
        torch.testing.assert_close(getattr(requests, name)[:, destination], getattr(requests, name)[:, source])
    torch.testing.assert_close(states.buffer[destination_id], checkpoint, rtol=0, atol=0)
    # The saved prefix restores the earlier continuation, rather than final prompt state.
    requests.restore_state(SimpleNamespace(req_idx=destination, cur_kv_len=1024), states, destination_id)
    positions = torch.arange(1024 - requests.sliding_window, 1024, device="cuda")
    actual = gather_swa_state(requests, destination, positions).cpu()
    torch.testing.assert_close(actual, states.get_state_cache(source_id)[0], rtol=0, atol=0)
    for runtime, saved in zip(
        (requests.req_to_c4_state, requests.req_to_c4_indexer_state), states.get_state_cache(source_id)[1:]
    ):
        slots = torch.arange(1020, 1024, device="cuda") % requests.c4_state_tokens
        torch.testing.assert_close(runtime[:, destination].index_select(1, slots).cpu(), saved, rtol=0, atol=0)
    requests.free([source, destination], torch.cat((src, dst)))
    states.free_state_cache([source_id, destination_id])
    states.layer_cache.shm.unlink()
    states.metadata_shm.unlink()


@pytest.mark.parametrize("received", [False, True])
@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("kind", ["small", "big"])
def test_decode_keeps_only_successfully_received_checkpoint(monkeypatch, received, failed, kind):
    node = decode_impl.PDDecodeNode.__new__(decode_impl.PDDecodeNode)
    node.is_master_in_dp = False
    node.is_deepseek_v4 = True
    node.small_page_buffers = SimpleNamespace(free_state_cache=MagicMock())
    big = SimpleNamespace(free_state_cache=MagicMock())
    node.model = SimpleNamespace(mem_manager=SimpleNamespace(big_page_buffers=big))
    req = SimpleNamespace(
        req_id=1,
        infer_aborted=False,
        pd_task_num=2,
        pd_task_success_num=2 - int(failed),
        pd_task_failed_num=int(failed),
        pd_checkpoint_received_len=512 if received else 0,
        hybrid_cache_len=512,
        tail_small_page_buffer_id=7 if kind == "small" else None,
        hybrid_len_to_big_page_id={512: 7} if kind == "big" else {},
        finish_status=SimpleNamespace(is_finished=lambda: True),
        shared_kv_node=None,
        hold_kv_len=0,
    )
    monkeypatch.setattr(decode_impl.g_infer_context, "requests_mapping", {1: req})
    assert node._filter_not_ready_reqs([1]) == [req]
    assert node._filter_not_ready_reqs([1]) == [req]
    pool = node.small_page_buffers if kind == "small" else big
    if received and not failed:
        pool.free_state_cache.assert_not_called()
        assert req.tail_small_page_buffer_id == 7 or req.hybrid_len_to_big_page_id == {512: 7}
    else:
        pool.free_state_cache.assert_called_once_with([7])
        assert req.tail_small_page_buffer_id is None and req.hybrid_len_to_big_page_id == {}


@pytest.mark.parametrize("source_kind", ["small", "big", "shared_small", "shared_big", "missing"])
def test_prefill_task_exports_the_aligned_checkpoint(source_kind):
    node = PDChunkedPrefillForPrefillNode.__new__(PDChunkedPrefillForPrefillNode)
    node.is_deepseek_v4, node.dp_rank_in_node, node.node_world_size = True, 0, 8
    req = SimpleNamespace(
        req_id=1,
        req_idx=0,
        pd_trans_device_id=-1,
        pd_task_num=0,
        hybrid_cache_len=512,
        tail_small_page_buffer_id=7 if source_kind == "small" else None,
        hybrid_len_to_big_page_id={512: 9} if source_kind == "big" else {},
        shared_kv_node=None,
        sampling_param=SimpleNamespace(
            pd_master_node_id=0,
            pd_decode_node=SimpleNamespace(agent_name="D", agent_metadata=b"D", num_pages=16, page_reg_desc=b"page"),
        ),
    )
    if source_kind.startswith("shared"):
        req.shared_kv_node = SimpleNamespace(
            node_prefix_total_len=512,
            is_big_page_node=lambda: source_kind == "shared_big",
            small_page_buffer_idx=7,
            big_page_buffer_idx=9,
        )
    task = node._create_pd_trans_task(req, 1023, 1023, "att_state")
    assert task.checkpoint_len == (0 if source_kind == "missing" else 512)
    assert task.checkpoint_buffer_id == (None if source_kind == "missing" else 9 if "big" in source_kind else 7)
    assert task.createRetObj().checkpoint_len == task.checkpoint_len


def test_nixl_short_write_fence_reuses_identical_descriptors():
    transporter = NixlKVTransporter.__new__(NixlKVTransporter)
    transporter.page_len, transporter.num_pages = 100, 3
    transporter._remote_agents_lock = threading.Lock()
    transporter._write_contexts, transporter.page_local_xfer_handles = {}, {}
    transporter.page_reg_desc = [(1000, 300, 0, "")]
    remote = PDAgentMetadata("D", b"D", 3, [(2000, 300, 1, "")], {})
    transporter.remote_agents = {"D": remote}
    transporter.nixl_agent = SimpleNamespace(
        get_xfer_descs=lambda pages, kind: pages,
        prep_xfer_dlist=lambda name, descs, kind: (name, descs),
        deserialize_descs=lambda descs: descs,
        make_prepped_xfer=MagicMock(side_effect=[7, 8, 9]),
        transfer=MagicMock(return_value="PROC"),
        query_xfer_backend=lambda handle: "UCX",
    )
    task = _task(decode_agent_name="D", src_page_index=1, dst_page_index=2, transfer_nbytes=24)
    transporter.write_blocks_paged(task)
    _, src, dst = transporter._write_contexts[7]
    assert src[1] == [(1000, 24, 0), (1100, 24, 0), (1200, 24, 0)]
    assert dst[1] == [(2000, 24, 1), (2100, 24, 1), (2200, 24, 1)]
    fence = replace(task, xfer_handle=None, write_submitted=False)
    transporter.start_write_fence(task, fence)
    transporter.nixl_agent.make_prepped_xfer.assert_called_with("WRITE", src, [1], dst, [2], b"", backends=["UCX"])
    transporter.write_blocks_paged(replace(task, xfer_handle=None, transfer_nbytes=None))
    assert set(transporter.page_local_xfer_handles) == {24, 100}
    assert set(remote.page_xfer_handles) == {24, 100}


def test_checkpoint_receipt_is_published_once_after_success(monkeypatch):
    from lightllm.server.router.model_infer.mode_backend import base_backend

    task = _task(
        page_kind="att_state",
        start_kv_index=1023,
        end_kv_index=1023,
        mem_indexes=[],
        checkpoint_len=512,
        first_gen_token_id=None,
    )
    ret = task.createRetObj()
    req = SimpleNamespace(
        pd_task_num=1,
        pd_task_failed_num=0,
        pd_task_success_num=0,
        pd_task_finished_keys=set(),
        hybrid_cache_len=512,
        pd_checkpoint_received_len=0,
    )
    monkeypatch.setattr(base_backend.g_infer_context, "requests_mapping", {task.request_id: req})
    node = PDChunkedPrefillForPrefillNode.__new__(PDChunkedPrefillForPrefillNode)
    node.is_pd_decode_mode, node.logger = True, MagicMock()
    node.shm_pd_trans_io_buffer = SimpleNamespace(read_obj=lambda: [ret, copy.copy(ret)], sub_state=lambda: None)
    node._read_pd_trans_io_buffer_and_update_req_status()
    assert req.pd_checkpoint_received_len == 512 and req.pd_task_success_num == 1


def test_decode_notify_keeps_local_checkpoint_destination(monkeypatch):
    import pickle
    import queue
    from lightllm.server.router.model_infer.mode_backend.pd.decode_node_impl.decode_trans_process import (
        _DecodeTransModule,
    )

    class StopLoop(Exception):
        pass

    local = _task(
        page_kind="att_state",
        start_kv_index=1023,
        end_kv_index=1023,
        mem_indexes=[],
        checkpoint_page_kind="small",
        checkpoint_buffer_id=7,
    )
    local.shared_time_out_mark = SimpleNamespace(update=lambda: None)
    remote = replace(local, checkpoint_buffer_id=99, checkpoint_len=512, transfer_nbytes=100, write_stage="request")
    module = _DecodeTransModule.__new__(_DecodeTransModule)
    module.device_id, module.use_write_quiescence = 0, False
    module.waiting_dict_lock, module.waiting_dict = threading.Lock(), {local.get_key(): local}
    module.request_page_task_queue = queue.Queue()
    module.transporter = SimpleNamespace(get_new_notifs=lambda: {"P": [pickle.dumps(remote)]})

    def stop():
        raise StopLoop()

    module._check_tasks_time_out = stop
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    with pytest.raises(StopLoop):
        module.accept_peer_task_loop()
    queued = module.request_page_task_queue.get_nowait()
    assert queued is local and queued.checkpoint_buffer_id == 7
    assert queued.checkpoint_len == 512 and queued.transfer_nbytes == 100
