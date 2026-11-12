from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from lightllm.server.router.model_infer.mode_backend.chunked_prefill import impl
from lightllm.server.router.model_infer.mode_backend.chunked_prefill.control_state import ControlState


class StopLoop(Exception):
    pass


@pytest.mark.parametrize("mode", ["decode", "prefill", "normal"])
@pytest.mark.parametrize("overlap", [False, True])
def test_pending_wait_releases_to_idle(monkeypatch, mode, overlap):
    context = SimpleNamespace(infer_req_ids=[1])
    monkeypatch.setattr(impl, "g_infer_context", context)
    monkeypatch.setattr(impl, "get_current_device_id", lambda: 0)
    monkeypatch.setattr(impl.torch.cuda, "set_device", lambda _: None)
    events = Mock()
    events.notify_forward_and_wait_post_handle.side_effect = lambda: context.infer_req_ids.clear()
    waits = []

    def sleep(seconds):
        waits.append(seconds)
        if len(waits) == 2:
            raise StopLoop()

    monkeypatch.setattr(impl, "time", SimpleNamespace(sleep=sleep))
    backend = impl.ChunkedPrefillBackend.__new__(impl.ChunkedPrefillBackend)
    backend.support_overlap = overlap
    backend.is_pd_decode_mode = mode == "decode"
    backend.classed_req_no_decode = mode == "prefill"
    backend.classed_req_strict_prefill = False
    backend.overlap_event_manager = SimpleNamespace(get_overlap_event_pack=lambda: events)
    backend._try_read_new_reqs = Mock()
    backend._get_classed_reqs = Mock(return_value=([], [], 0))
    backend.control_state_machine = ControlState.__new__(ControlState)
    backend.control_state_machine.is_aggressive_schedule = True
    backend.control_state_machine.step_count = 0
    backend.logger = Mock()

    with pytest.raises(StopLoop):
        backend.infer_loop()

    assert waits == ([0.001, 0.02] if mode == "decode" else [0.02, 0.02])
    assert backend._try_read_new_reqs.call_count == 2
    cycle = [
        call.wait_to_forward(),
        call.notify_post_handle_and_wait_pre_post_handle(),
        call.notify_forward_and_wait_post_handle(),
        call.notify_pre_post_handle(),
    ]
    if not overlap:
        cycle.insert(0, call._close_overlap())
    assert events.mock_calls == cycle * 2
