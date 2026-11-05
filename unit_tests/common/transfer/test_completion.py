import gc
import weakref
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention_state.manager import AttentionStateManager
from lightllm.common.execution.storage.lifetime import DeferredReclaimer
from lightllm.common.execution.transfer import CopyOperation, CopyPlan, Transfer, TransferFailure
from unit_tests.common.transfer.helpers import Event, Runtime


def test_source_can_be_reused_before_destination_is_published():
    runtime = Runtime()
    copied = []
    plan = CopyPlan("pack", (CopyOperation("kv", None, lambda: copied.append("kv")),))

    def publish_all_shards():
        assert copied == ["kv"]
        assert len(runtime.events) == 1
        runtime.events[0].ready = True

    completion = Transfer(runtime).submit(plan, destination_barrier=publish_all_shards)
    assert completion.source_reusable.query()
    assert not completion.destination_ready.query()
    completion.destination_ready.synchronize()
    assert completion.destination_ready.query()
    assert completion.keepalive is plan


@pytest.mark.parametrize("fail_in_barrier", [False, True])
def test_partial_copy_failure_has_quiescence_without_ready_destination(fail_in_barrier):
    runtime = Runtime()
    copied = []

    def fail():
        raise ValueError("injected submission failure")

    operations = [CopyOperation("kv", None, lambda: copied.append("kv"))]
    if not fail_in_barrier:
        operations.append(CopyOperation("state", None, fail))
    plan = CopyPlan("unpack", tuple(operations))
    with pytest.raises(TransferFailure) as caught:
        Transfer(runtime).submit(plan, destination_barrier=fail if fail_in_barrier else None)
    failure = caught.value
    assert copied == ["kv"]
    assert isinstance(failure.__cause__, ValueError)
    assert not hasattr(failure, "destination_ready")
    assert not failure.quiesced.query()
    assert failure.keepalive is plan


def test_retirement_waits_and_snapshots_reused_allocator_index_buffer():
    manager = AttentionStateManager(1)
    released = []
    manager.full_att_allocator = SimpleNamespace(free=lambda tokens: released.append(tokens.tolist()))
    manager.bounded_att_big_page_allocator = SimpleNamespace(free=lambda pages: released.append(pages))
    tokens = torch.tensor([3, 4])
    pages = [8]
    event = Event()
    manager.retire(event, token_indices=tokens, big_pages=pages)
    tokens.fill_(99)
    pages.clear()
    assert manager.reclaimer.collect() == 0
    assert released == []
    event.ready = True
    assert manager.reclaimer.collect() == 1
    assert released == [[3, 4], [8]]
    assert manager.reclaimer.collect() == 0


def test_independent_releases_are_not_blocked_and_keep_views_alive():
    class View:
        pass

    view = View()
    ref = weakref.ref(view)
    reclaimer = DeferredReclaimer()
    slow, fast = Event(), Event(True)
    freed = []
    reclaimer.defer(slow, lambda: freed.append("slow"), view)
    reclaimer.defer(fast, lambda: freed.append("fast"))
    del view
    gc.collect()
    assert ref() is not None
    assert reclaimer.collect() == 1
    assert freed == ["fast"]
    reclaimer.synchronize()
    gc.collect()
    assert ref() is None
    assert freed == ["fast", "slow"]


def test_failed_completion_query_does_not_release_resources():
    def query():
        raise RuntimeError("device lost")

    freed = []
    reclaimer = DeferredReclaimer()
    reclaimer.defer(SimpleNamespace(query=query), lambda: freed.append(True))
    with pytest.raises(RuntimeError, match="device lost"):
        reclaimer.collect()
    assert not freed
    assert len(reclaimer.pending) == 1
