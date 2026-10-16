import threading
import weakref
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.server.router.model_infer.mode_backend import ep_runtime, eplb_manager
from lightllm.server.router.model_infer.mode_backend.chunked_prefill.impl import ChunkedPrefillBackend
from lightllm.server.router.model_infer.mode_backend.dp_backend.impl import DPChunkedPrefillBackend
from lightllm.server.router.model_infer.mode_backend.overlap_events import OverlapEventManager


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", "0")
    events = []
    model = TpPartBaseModel.__new__(TpPartBaseModel)
    output = object()
    model._prefill = lambda model_input: events.append("prefill") or output
    model._decode = lambda model_input: events.append("decode") or output
    model._microbatch_overlap_prefill_cuda = lambda *inputs: events.append("overlap") or output
    model.microbatch_overlap_decode = lambda *inputs: output
    monitor = SimpleNamespace(enabled=True, record_prefill_round=lambda: events.append("record"))
    manager = SimpleNamespace(
        before_prefill=lambda: events.append("prepare"),
        step=lambda: events.append("step"),
        poll=lambda: events.append("poll"),
        _eplb_states=[SimpleNamespace(migration_poll=None)],
        poll_migration_at_layer_boundary=lambda: events.append("layer_poll"),
    )
    backend = SimpleNamespace(
        args=SimpleNamespace(
            enable_prefill_eplb=True,
            disable_ep_balance_monitor=False,
            enable_prefill_microbatch_overlap=False,
            enable_tpsp_mix_mode=False,
        ),
        model=model,
        overlap_event_manager=OverlapEventManager(),
    )
    monkeypatch.setattr(ep_runtime, "EPBalanceMonitor", lambda model: monitor)
    monkeypatch.setattr(
        ep_runtime, "should_enable_ep_balance_monitor", lambda args: not args.disable_ep_balance_monitor
    )
    monkeypatch.setattr(eplb_manager, "EPLBManager", lambda model: manager)
    monkeypatch.setattr(ep_runtime.dist, "barrier", lambda: events.append("barrier"))
    return backend, events, output, monitor, manager


@pytest.mark.parametrize("eplb,monitor_enabled", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("batch_size", [0, 3])
def test_completed_rounds_and_decode_preserve_behavior(runtime, eplb, monitor_enabled, batch_size):
    backend, events, output, monitor, _ = runtime
    backend.args.enable_prefill_eplb = eplb
    backend.args.disable_ep_balance_monitor = not monitor_enabled
    model = backend.model
    original_methods = vars(model).copy()
    original_event_methods = vars(backend.overlap_event_manager).copy()
    ep_runtime.install_ep_runtime(backend)
    assert events == (["barrier"] if eplb else [])
    events.clear()

    model_input = SimpleNamespace(is_prefill=True, batch_size=batch_size, to_cuda=lambda: None)
    expected = (["record"] if monitor_enabled else []) + (["step"] if eplb else [])
    assert model.forward(model_input) is output
    assert events == [*(["prepare"] if eplb else []), "prefill", *expected]
    events.clear()
    # Empty DP ranks and both empty microbatches still participate in the
    # globally scheduled round. Count one full round, not one per microbatch.
    assert model._microbatch_overlap_prefill_cuda(model_input, model_input) is output
    assert events == [*(["prepare"] if eplb else []), "overlap", *expected]
    events.clear()
    model_input.is_prefill = False
    assert model.forward(model_input) is output
    assert events == ["decode"]
    assert model._decode is original_methods["_decode"]
    assert model.microbatch_overlap_decode is original_methods["microbatch_overlap_decode"]
    if not eplb:
        assert vars(backend.overlap_event_manager) == original_event_methods
    if not eplb and not monitor_enabled:
        assert vars(model) == original_methods


def test_disabled_monitor_and_draft_methods_are_untouched(runtime):
    backend, events, _, monitor, _ = runtime
    backend.args.enable_prefill_eplb = False
    monitor.enabled = False
    original_methods = vars(backend.model).copy()
    ep_runtime.install_ep_runtime(backend)
    assert vars(backend.model) == original_methods
    assert not events
    # The installer binds only the supplied target model, never its class.
    draft = TpPartBaseModel.__new__(TpPartBaseModel)
    backend.args.enable_prefill_eplb = True
    ep_runtime.install_ep_runtime(backend)
    assert "_prefill" not in vars(draft)
    assert draft._prefill.__func__ is TpPartBaseModel._prefill


@pytest.mark.parametrize("method", ["_prefill", "_microbatch_overlap_prefill_cuda"])
def test_failed_prefill_is_not_published(runtime, method):
    backend, events, _, _, _ = runtime

    def fail(*args, **kwargs):
        raise ValueError("forward failed")

    setattr(backend.model, method, fail)
    ep_runtime.install_ep_runtime(backend)
    events.clear()
    with pytest.raises(ValueError, match="forward failed"):
        getattr(backend.model, method)(object())
    assert events == ["prepare"]


def test_poll_waits_for_execution_ownership_and_propagates_failures(runtime):
    backend, events, _, _, manager = runtime
    ep_runtime.install_ep_runtime(backend)
    first = backend.overlap_event_manager.get_overlap_event_pack()
    second = backend.overlap_event_manager.get_overlap_event_pack()
    first._close_overlap()
    second._close_overlap()
    events.clear()
    started = threading.Event()
    finished = threading.Event()

    def run_second():
        started.set()
        second.wait_to_forward()
        second.notify_post_handle_event.set()
        finished.set()

    thread = threading.Thread(target=run_second, daemon=True)
    thread.start()
    assert started.wait(2)
    assert not finished.wait(0.02)
    assert not events
    first.wait_to_forward()
    assert events == ["poll"]
    first.notify_pre_post_handle()
    assert finished.wait(2)
    thread.join(2)
    assert not thread.is_alive()
    assert events == ["poll", "poll"]

    def fail():
        raise RuntimeError("poll failed")

    manager.poll = fail
    with pytest.raises(RuntimeError, match="poll failed"):
        second.wait_to_forward()


def test_poll_binding_does_not_retain_completed_event_packs(runtime):
    backend, _, _, _, _ = runtime
    ep_runtime.install_ep_runtime(backend)
    event_pack = backend.overlap_event_manager.get_overlap_event_pack()
    reference = weakref.ref(event_pack)
    del event_pack
    assert reference() is None


@pytest.mark.parametrize("backend_class", [ChunkedPrefillBackend, DPChunkedPrefillBackend])
def test_real_infer_loops_poll_before_requests_even_when_idle(runtime, monkeypatch, backend_class):
    template, events, _, _, _ = runtime
    backend = backend_class.__new__(backend_class)
    vars(backend).update(vars(template))
    backend.support_overlap = False
    backend.classed_req_no_decode = False
    backend.classed_req_strict_prefill = False
    backend.logger = SimpleNamespace(exception=lambda message: None)
    actions = iter(("prefill", "decode", "pass"))
    backend.control_state_machine = SimpleNamespace(
        try_recover_paused_reqs=lambda: False,
        select_run_way=lambda **kwargs: _run_way(next(actions)),
    )
    backend._get_classed_reqs = lambda **kwargs: ([], [])
    get_event_pack = backend.overlap_event_manager.get_overlap_event_pack

    def get_ready_event_pack():
        event_pack = get_event_pack()
        # Simulate the other thread's completed post-handle stage; this test
        # drives one real infer_loop. The test above checks actual handoff.
        event_pack.wait_post_handle_event.set()
        return event_pack

    backend.overlap_event_manager.get_overlap_event_pack = get_ready_event_pack

    class StopLoop(BaseException):
        pass

    def read_requests():
        events.append("requests")
        if events.count("requests") == 4:
            raise StopLoop()

    def gather(**kwargs):
        events.append("gather")
        return False, False

    def prefill(event_pack, prefill_reqs):
        backend.model.forward(SimpleNamespace(is_prefill=True, to_cuda=lambda: None))
        event_pack.notify_pre_post_handle()

    def decode(event_pack, decode_reqs):
        backend.model.forward(SimpleNamespace(is_prefill=False, to_cuda=lambda: None))
        event_pack.notify_pre_post_handle()

    backend._try_read_new_reqs = read_requests
    backend._dp_all_reduce_req_presence = gather
    backend.prefill = prefill
    backend.decode = decode
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)
    from lightllm.server.router.model_infer.infer_batch import g_infer_context

    monkeypatch.setattr(g_infer_context, "get_overlap_stream", lambda: SimpleNamespace(wait_stream=lambda stream: None))
    ep_runtime.install_ep_runtime(backend)
    events.clear()
    with pytest.raises(StopLoop):
        backend.infer_loop()
    gathered = ["gather"] if backend_class is DPChunkedPrefillBackend else []
    assert events == [
        "poll",
        "requests",
        *gathered,
        "prepare",
        "prefill",
        "record",
        "step",
        "poll",
        "requests",
        *gathered,
        "decode",
        "poll",
        "requests",
        *gathered,
        "poll",
        "requests",
    ]


def _run_way(action):
    return SimpleNamespace(
        is_prefill=lambda: action == "prefill", is_decode=lambda: action == "decode", is_pass=lambda: action == "pass"
    )


@pytest.mark.parametrize(
    "microbatch,tpsp,installed", [(False, False, True), (True, False, False), (False, True, False)]
)
def test_layer_migration_hooks_only_use_ordered_prefill_paths(runtime, microbatch, tpsp, installed):
    backend, events, _, _, manager = runtime
    backend.args.enable_prefill_microbatch_overlap = microbatch
    backend.args.enable_tpsp_mix_mode = tpsp
    ep_runtime.install_ep_runtime(backend)
    hook = manager._eplb_states[0].migration_poll
    assert (hook is not None) == installed
    if installed:
        hook()
        assert events[-1] == "layer_poll"
