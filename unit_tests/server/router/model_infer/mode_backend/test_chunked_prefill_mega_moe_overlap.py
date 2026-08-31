from lightllm.server.router.model_infer.mode_backend.chunked_prefill import (
    impl as chunked_prefill_impl,
)
from lightllm.server.router.model_infer.mode_backend.chunked_prefill.impl import (
    ChunkedPrefillBackend,
)


class _FakeEvent:
    def __init__(self, calls=None):
        self.recorded = False
        self.synchronized = False
        self.calls = calls

    def record(self):
        self.recorded = True

    def synchronize(self):
        self.synchronized = True
        if self.calls is not None:
            self.calls.append("synchronize")


class _FakeEventPack:
    def __init__(self, calls):
        self.calls = calls

    def notify_forward_and_wait_post_handle(self):
        self.calls.append("notify_forward")


def test_record_forward_completion_keeps_cpu_bookkeeping_async(monkeypatch):
    event = _FakeEvent()
    monkeypatch.setattr(chunked_prefill_impl.torch.cuda, "Event", lambda: event)
    backend = ChunkedPrefillBackend.__new__(ChunkedPrefillBackend)
    backend._serialize_sm90_mega_moe_forwards = True

    assert backend._record_forward_completion() is event
    assert event.recorded
    assert not event.synchronized


def test_mega_moe_waits_before_notifying_next_forward():
    calls = []
    event = _FakeEvent(calls)
    event_pack = _FakeEventPack(calls)
    backend = ChunkedPrefillBackend.__new__(ChunkedPrefillBackend)
    backend._serialize_sm90_mega_moe_forwards = True

    backend._notify_next_forward_when_safe(event_pack, event)

    assert calls == ["synchronize", "notify_forward"]


def test_regular_path_notifies_next_forward_before_waiting():
    calls = []
    event = _FakeEvent(calls)
    event_pack = _FakeEventPack(calls)
    backend = ChunkedPrefillBackend.__new__(ChunkedPrefillBackend)
    backend._serialize_sm90_mega_moe_forwards = False

    backend._notify_next_forward_when_safe(event_pack, event)

    assert calls == ["notify_forward", "synchronize"]
