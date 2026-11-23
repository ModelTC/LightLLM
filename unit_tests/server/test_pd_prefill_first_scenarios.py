"""Twenty checks for PD 分离的 Prefill 先推理模式.

These drive the real prefill, decode, router, and infer-command functions with
small stand-ins, so a logic mistake fails without starting a model service.
"""

import argparse
import asyncio
import gc
import pickle
import weakref
from types import SimpleNamespace

import pytest
import torch

from lightllm.server.core.objs.io_objs import AbortedReqCmd, PdPfDecodeKvReadyCmd
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.server.pd_io_struct import ObjType, PDDecodeNodeInfo
from lightllm.server.router.manager import RouterManager
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.server.router.model_infer.mode_backend.pd.decode_node_impl.decode_impl import PDDecodeNode
from lightllm.server.router.model_infer.mode_backend.pd.helper import iter_kv_spans, submit_prefill_first_tasks_if_ready
from lightllm.server.router.model_infer.mode_backend.pd.prefill_node_impl.prefill_impl import (
    PDChunkedPrefillForPrefillNode,
)


class _ShmReq:
    def __init__(self, input_len: int) -> None:
        self.input_len = input_len


class _TokenIndex:
    def get_cpu_token_indexes(self, start: int, end: int):
        return torch.arange(start, end)


class _Req:
    def __init__(
        self, req_id: int, input_len: int, cur_kv_len: int, prefill_first: bool = True, payload: bytes = b""
    ) -> None:
        self.req_id = req_id
        self.req_idx = 0
        self.cur_kv_len = cur_kv_len
        self.shm_req = _ShmReq(input_len=input_len)
        self.sampling_param = SimpleNamespace(
            shm_param=SimpleNamespace(
                pd_prefill_first=prefill_first,
                pd_kv_trans_params=SimpleNamespace(get=lambda: payload, data_len=len(payload)),
            ),
            pd_decode_node=_decode_info(ready_kv_len=0),
            pd_master_node_id=2,
        )
        self.infer_aborted = False
        self.pd_task_num = 0
        self.pd_task_failed_num = 0
        self.pd_task_success_num = 0
        self.pd_trans_device_id = -1
        self.pd_trans_kv_start_index = 0
        self.token_index = _TokenIndex()


def _req(
    req_id: int,
    input_len: int,
    cur_kv_len: int,
    prefill_first: bool = True,
    payload: bytes = b"",
) -> _Req:
    return _Req(
        req_id=req_id,
        input_len=input_len,
        cur_kv_len=cur_kv_len,
        prefill_first=prefill_first,
        payload=payload,
    )


def _prefill_backend(queue):
    backend = PDChunkedPrefillForPrefillNode.__new__(PDChunkedPrefillForPrefillNode)
    backend.args = SimpleNamespace(enable_cpu_cache=False, pd_kv_page_size=4, page_size=4)
    backend.is_master_in_dp = True
    backend.info_queue = queue
    backend.node_world_size = 1
    backend.dp_rank_in_node = 0
    return backend


def _install_mapping(monkeypatch, *reqs):
    monkeypatch.setattr(g_infer_context, "requests_mapping", {req.req_id: req for req in reqs})


def _decode_info(ready_kv_len: int) -> PDDecodeNodeInfo:
    return PDDecodeNodeInfo(
        decode_node_id=1,
        pd_master_node_id=2,
        agent_name="decode",
        agent_metadata=b"meta",
        num_pages=1,
        page_reg_desc=b"pages",
        request_id=9,
        ready_kv_len=ready_kv_len,
    )


class _Buffer:
    def __init__(self) -> None:
        self.obj = None

    def is_empty(self) -> bool:
        return True

    def write_obj(self, obj) -> None:
        self.obj = obj

    def set_ready(self) -> None:
        return


def test_01_full_match_span_keeps_the_reserved_token():
    assert iter_kv_spans(start=7, end=8, page_size=4) == [(7, 8)]


def test_02_spans_follow_page_size_and_keep_the_short_tail():
    assert iter_kv_spans(start=1, end=6, page_size=4) == [(1, 5), (5, 6)]


def test_03_empty_span_and_bad_page_size_are_rejected():
    with pytest.raises(ValueError):
        iter_kv_spans(start=4, end=4, page_size=2)
    with pytest.raises(ValueError):
        iter_kv_spans(start=0, end=2, page_size=0)


def test_04_pf_messages_do_not_reuse_old_handshake_values():
    assert ObjType.PD_PF_PREFILL_REQ.value == 10
    assert ObjType.PD_PF_PREFILL_KV_READY.value == 11
    assert ObjType.PD_PF_DECODE_KV_READY.value == 12
    assert ObjType.PD_PF_PREFILL_REQ != ObjType.REQ
    assert ObjType.PD_PF_DECODE_KV_READY != ObjType.PD_REQ_DECODE_NODE_INFO


def test_05_schedule_interval_default_is_5ms_and_cli_help_is_english():
    from lightllm.server.api_cli import add_cli_args

    assert StartArgs().schedule_time_interval == 0.005
    assert StartArgs().pd_prefill_first is False
    parser = argparse.ArgumentParser()
    add_cli_args(parser)
    help_text = parser.format_help()
    assert "default is 5ms" in help_text
    assert "PD disaggregation prefill-first mode" in help_text
    assert "PD 分离" not in help_text


def test_06_wait_closure_does_not_capture_the_request_or_backend():
    gc.disable()
    try:
        req = _req(req_id=1, input_len=4, cur_kv_len=4)
        backend = _prefill_backend(queue=SimpleNamespace(put=lambda _task: None))
        req_ref = weakref.ref(req)
        backend_ref = weakref.ref(backend)
        backend._pf_prefill_chuncked_handle_func(req_obj=req, next_token_id=7, next_token_prob=-0.2, output_len=1)
        closure = req.pd_wait_decode_kv_info_ready
        assert closure.__code__.co_freevars == ("next_token_id", "next_token_prob")
        del closure
        del req
        del backend
        assert req_ref() is None
        assert backend_ref() is None
    finally:
        gc.enable()


def test_07_unfinished_prefill_first_chunk_does_not_wait_or_emit_tasks(monkeypatch):
    monkeypatch.setattr(g_infer_context, "is_hybrid_att_model", False)
    queued = []
    backend = _prefill_backend(queue=SimpleNamespace(put=queued.append))
    req = _req(req_id=1, input_len=10, cur_kv_len=3)
    backend._prefill_chuncked_handle_func(req_obj=req, next_token_id=1, next_token_prob=0.0, output_len=0)
    assert not hasattr(req, "pd_wait_decode_kv_info_ready")
    assert queued == []
    assert req.pd_task_num == 0


def test_08_finished_prefill_first_chunk_installs_one_closure_and_keeps_the_first_token():
    backend = _prefill_backend(queue=SimpleNamespace(put=lambda _task: None))
    req = _req(req_id=1, input_len=4, cur_kv_len=4)
    backend._prefill_chuncked_handle_func(req_obj=req, next_token_id=11, next_token_prob=-0.1, output_len=1)
    first = req.pd_wait_decode_kv_info_ready
    backend._prefill_chuncked_handle_func(req_obj=req, next_token_id=99, next_token_prob=-9.0, output_len=1)
    assert req.pd_wait_decode_kv_info_ready is first
    assert first.__closure__[0].cell_contents == 11


def test_09_old_prefill_path_emits_a_chunk_and_does_not_install_the_wait_closure(monkeypatch):
    monkeypatch.setattr(g_infer_context, "is_hybrid_att_model", False)
    queued = []
    backend = _prefill_backend(queue=SimpleNamespace(put=queued.append))
    req = _req(req_id=1, input_len=4, cur_kv_len=4, prefill_first=False)
    backend._prefill_chuncked_handle_func(req_obj=req, next_token_id=8, next_token_prob=-0.3, output_len=1)
    assert not hasattr(req, "pd_wait_decode_kv_info_ready")
    assert len(queued) == 1
    assert queued[0].first_gen_token_id == 8
    assert queued[0].start_kv_index == 0
    assert queued[0].end_kv_index == 4


def test_10_filter_holds_a_finished_request_that_is_still_waiting(monkeypatch):
    req = _req(req_id=3, input_len=4, cur_kv_len=4)
    req.pd_wait_decode_kv_info_ready = lambda **_kwargs: None
    _install_mapping(monkeypatch, req)
    ready = _prefill_backend(queue=SimpleNamespace(put=lambda _task: None))._filter_not_ready_reqs([3])
    assert ready == []


def test_11_filter_lets_an_aborted_waiting_request_leave(monkeypatch):
    req = _req(req_id=3, input_len=4, cur_kv_len=4)
    req.infer_aborted = True
    req.pd_wait_decode_kv_info_ready = lambda **_kwargs: None
    _install_mapping(monkeypatch, req)
    ready = _prefill_backend(queue=SimpleNamespace(put=lambda _task: None))._filter_not_ready_reqs([3])
    assert ready == [req]


def test_12_filter_does_not_hold_an_old_mode_finished_request(monkeypatch):
    req = _req(req_id=3, input_len=4, cur_kv_len=4, prefill_first=False)
    _install_mapping(monkeypatch, req)
    ready = _prefill_backend(queue=SimpleNamespace(put=lambda _task: None))._filter_not_ready_reqs([3])
    assert ready == [req]


def test_13_filter_keeps_an_in_progress_prefill_first_request_schedulable(monkeypatch):
    req = _req(req_id=3, input_len=10, cur_kv_len=4)
    _install_mapping(monkeypatch, req)
    ready = _prefill_backend(queue=SimpleNamespace(put=lambda _task: None))._filter_not_ready_reqs([3])
    assert ready == [req]


def test_14_filter_waits_until_real_transfer_tasks_finish(monkeypatch):
    req = _req(req_id=3, input_len=4, cur_kv_len=4)
    req.pd_task_num = 2
    req.pd_task_success_num = 1
    _install_mapping(monkeypatch, req)
    ready = _prefill_backend(queue=SimpleNamespace(put=lambda _task: None))._filter_not_ready_reqs([3])
    assert ready == []
    req.pd_task_success_num = 2
    ready = _prefill_backend(queue=SimpleNamespace(put=lambda _task: None))._filter_not_ready_reqs([3])
    assert ready == [req]


def test_15_ready_command_for_a_missing_request_is_ignored(monkeypatch):
    _install_mapping(monkeypatch)
    backend = _prefill_backend(queue=SimpleNamespace(put=lambda _task: None))
    backend._on_pf_decode_kv_ready(req_id=404)


def test_16_ready_command_without_the_wait_attribute_is_ignored(monkeypatch):
    req = _req(req_id=5, input_len=4, cur_kv_len=4)
    _install_mapping(monkeypatch, req)
    backend = _prefill_backend(queue=SimpleNamespace(put=lambda _task: None))
    backend._on_pf_decode_kv_ready(req_id=5)
    assert req.pd_task_num == 0


def test_17_ready_command_on_an_aborted_request_drops_the_attribute_without_transfer(monkeypatch):
    called = []
    req = _req(req_id=5, input_len=4, cur_kv_len=4)
    req.infer_aborted = True
    req.pd_wait_decode_kv_info_ready = lambda **_kwargs: called.append(1)
    _install_mapping(monkeypatch, req)
    backend = _prefill_backend(queue=SimpleNamespace(put=called.append))
    backend._on_pf_decode_kv_ready(req_id=5)
    assert called == []
    assert not hasattr(req, "pd_wait_decode_kv_info_ready")


def test_18_ready_command_puts_the_first_token_on_the_last_task_including_att_state(monkeypatch):
    info = _decode_info(ready_kv_len=1)
    payload = pickle.dumps(info)
    req = _req(req_id=9, input_len=6, cur_kv_len=6, payload=payload)
    backend = _prefill_backend(queue=SimpleNamespace(put=lambda task: queued.append(task)))
    queued = []
    monkeypatch.setattr(g_infer_context, "is_hybrid_att_model", True)
    backend._pf_prefill_chuncked_handle_func(req_obj=req, next_token_id=21, next_token_prob=-0.4, output_len=1)
    _install_mapping(monkeypatch, req)
    backend._on_pf_decode_kv_ready(req_id=9)
    assert [(task.start_kv_index, task.end_kv_index, task.page_kind) for task in queued] == [
        (1, 5, "kv"),
        (5, 6, "kv"),
        (6, 6, "att_state"),
    ]
    assert queued[-1].first_gen_token_id == 21
    assert queued[-1].first_gen_token_logprob == -0.4
    assert queued[0].first_gen_token_id is None
    assert not hasattr(req, "pd_wait_decode_kv_info_ready")
    assert req.pd_task_num == 3


def test_19_no_token_to_send_is_an_assertion(monkeypatch):
    info = _decode_info(ready_kv_len=6)
    req = _req(req_id=9, input_len=6, cur_kv_len=6, payload=pickle.dumps(info))
    backend = _prefill_backend(queue=SimpleNamespace(put=lambda _task: None))
    monkeypatch.setattr(g_infer_context, "is_hybrid_att_model", False)
    with pytest.raises(AssertionError):
        submit_prefill_first_tasks_if_ready(backend=backend, req_obj=req, next_token_id=1, next_token_logprob=0.0)


def test_20_decode_prefill_first_full_match_sends_a_real_page_and_the_old_path_does_not():
    def run(method_name: str, cur_kv_len: int):
        backend = PDDecodeNode.__new__(PDDecodeNode)
        backend.args = SimpleNamespace(pd_kv_page_size=4, page_size=4)
        backend.is_master_in_dp = False
        backend.node_world_size = 1
        recorded = []

        def create(**kwargs):
            recorded.append((kwargs["kv_start_index"], kwargs["kv_end_index"], kwargs.get("page_kind", "kv")))
            kwargs["group"].task_list.append(object())

        backend._create_pd_trans_task = create
        req = SimpleNamespace(
            req_id=1,
            req_idx=0,
            cur_kv_len=cur_kv_len,
            hold_kv_len=cur_kv_len,
            pd_trans_kv_start_index=0,
            pd_trans_device_id=-1,
            pd_task_num=0,
            shm_req=SimpleNamespace(input_len=8),
            kv_cache_alloc_need=lambda target: target - cur_kv_len,
            alloc_kv_mem=lambda count: torch.arange(count),
        )
        getattr(backend, method_name)(req_obj=req)
        return recorded

    assert run("_pf_decode_node_gen_trans_tasks", cur_kv_len=7) == [(7, 8, "kv")]
    assert run("_normal_decode_node_gen_trans_tasks", cur_kv_len=7) == [(7, 7, "kv")]


def _router_req(req_id: int, prefill_first: bool, data_len: int, aborted: bool = False):
    return SimpleNamespace(
        request_id=req_id,
        is_aborted=aborted,
        _router_aborted=False,
        _router_pf_decode_kv_notified=False,
        sample_params=SimpleNamespace(
            pd_prefill_first=prefill_first,
            pd_kv_trans_params=SimpleNamespace(data_len=data_len),
        ),
    )


def test_21_router_selects_ready_prefill_first_once_and_skips_the_others():
    ready = _router_req(req_id=1, prefill_first=True, data_len=8)
    aborted = _router_req(req_id=2, prefill_first=True, data_len=8, aborted=True)
    old = _router_req(req_id=3, prefill_first=False, data_len=8)
    empty = _router_req(req_id=4, prefill_first=True, data_len=0)
    router = SimpleNamespace(running_batch=SimpleNamespace(reqs=[ready, aborted, old, empty]))
    selected = RouterManager._get_pf_decode_kv_ready_reqs(router)
    assert [req.request_id for req in selected] == [1]
    assert ready._router_pf_decode_kv_notified is True
    selected_again = RouterManager._get_pf_decode_kv_ready_reqs(router)
    assert selected_again == []


def test_22_selected_request_is_written_once_and_not_selected_again():
    ready = _router_req(req_id=1, prefill_first=True, data_len=8)
    router = SimpleNamespace(
        running_batch=SimpleNamespace(reqs=[ready]),
        shm_reqs_io_buffer=_Buffer(),
    )
    selected = RouterManager._get_pf_decode_kv_ready_reqs(router)
    asyncio.run(RouterManager._write_aborted_and_pf_decode_kv_ready(router, aborted_reqs=[], ready_reqs=selected))
    assert [type(cmd) for cmd in router.shm_reqs_io_buffer.obj] == [PdPfDecodeKvReadyCmd]
    assert ready._router_pf_decode_kv_notified is True
    assert RouterManager._get_pf_decode_kv_ready_reqs(router) == []


def test_23_one_queue_carries_abort_before_prefill_first_ready(monkeypatch):
    abort_req = _req(req_id=1, input_len=4, cur_kv_len=4)
    ready_req = _req(req_id=2, input_len=4, cur_kv_len=4)
    calls = []
    ready_req.pd_wait_decode_kv_info_ready = lambda **_kwargs: calls.append(ready_req.infer_aborted)
    _install_mapping(monkeypatch, abort_req, ready_req)
    buffer = _Buffer()
    router = SimpleNamespace(shm_reqs_io_buffer=buffer)
    abort_router_req = SimpleNamespace(request_id=1)
    ready_router_req = SimpleNamespace(request_id=2, _router_pf_decode_kv_notified=False)
    asyncio.run(
        RouterManager._write_aborted_and_pf_decode_kv_ready(
            router, aborted_reqs=[abort_router_req], ready_reqs=[ready_router_req]
        )
    )
    assert [type(cmd) for cmd in buffer.obj] == [AbortedReqCmd, PdPfDecodeKvReadyCmd]
    reader = SimpleNamespace(
        shm_reqs_io_buffer=SimpleNamespace(read_obj=lambda: buffer.obj, sub_state=lambda: None),
        _on_pf_decode_kv_ready=lambda req_id: PDChunkedPrefillForPrefillNode._on_pf_decode_kv_ready(
            SimpleNamespace(), req_id=req_id
        ),
    )
    # The same request appearing in both commands must see abort before the closure.
    both = _req(req_id=7, input_len=4, cur_kv_len=4)
    both.pd_wait_decode_kv_info_ready = lambda **_kwargs: calls.append("built")
    monkeypatch.setattr(g_infer_context, "requests_mapping", {1: abort_req, 2: ready_req, 7: both})
    reader.shm_reqs_io_buffer = SimpleNamespace(
        read_obj=lambda: [AbortedReqCmd(req_id=7), PdPfDecodeKvReadyCmd(req_id=7)],
        sub_state=lambda: None,
    )
    ModeBackend._read_reqs_buffer_and_init_reqs(reader)
    assert both.infer_aborted is True
    assert calls == []
    assert not hasattr(both, "pd_wait_decode_kv_info_ready")
