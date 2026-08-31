from types import SimpleNamespace

from lightllm.server.router import manager as router_manager
from lightllm.server.router.manager import RouterManager


class _ReqQueue:
    def __init__(self, waiting_req_num):
        self.waiting_req_num = waiting_req_num

    def get_wait_req_num(self):
        return self.waiting_req_num


def _make_router(waiting_req_num, interval=0.5, running_req_num=0, pending_req_num=0):
    router = RouterManager.__new__(RouterManager)
    router.args = SimpleNamespace(running_max_req_size=64)
    router.prefill_coalesce_interval = interval
    router._prefill_coalesce_deadline = None
    router.req_queue = _ReqQueue(waiting_req_num)
    router.running_batch = None if running_req_num == 0 else SimpleNamespace(reqs=[None] * running_req_num)
    router.schedule_new_batch = None if pending_req_num == 0 else SimpleNamespace(reqs=[None] * pending_req_num)
    return router


def test_prefill_coalescing_is_disabled_by_default():
    router = _make_router(waiting_req_num=1, interval=0.0)

    assert not router._should_defer_prefill_batch()


def test_partial_burst_waits_only_until_original_deadline(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(router_manager.time, "monotonic", lambda: now[0])
    router = _make_router(waiting_req_num=12)

    assert router._should_defer_prefill_batch()
    assert router._prefill_coalesce_deadline == 10.5

    now[0] = 10.49
    router.req_queue.waiting_req_num = 48
    assert router._should_defer_prefill_batch()
    assert router._prefill_coalesce_deadline == 10.5

    now[0] = 10.5
    assert not router._should_defer_prefill_batch()
    assert router._prefill_coalesce_deadline is None


def test_full_runnable_batch_skips_coalescing(monkeypatch):
    monkeypatch.setattr(router_manager.time, "monotonic", lambda: 10.0)
    router = _make_router(waiting_req_num=48, running_req_num=16)

    assert not router._should_defer_prefill_batch()
    assert router._prefill_coalesce_deadline is None


def test_expired_burst_launches_as_soon_as_running_batch_clears(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(router_manager.time, "monotonic", lambda: now[0])
    router = _make_router(waiting_req_num=40, running_req_num=64)

    assert router._should_defer_prefill_batch()
    now[0] = 11.0
    assert router._should_defer_prefill_batch()

    router.running_batch = None
    assert not router._should_defer_prefill_batch()
    assert router._prefill_coalesce_deadline is None
