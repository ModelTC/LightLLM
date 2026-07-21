import threading

import pytest

from lightllm.server.inference_runtime.prefix_context import (
    DuplicatePrefixTask,
    InvalidPrefixContextTransition,
    PrefixContextRegistry,
    PrefixContextReleaseError,
    PrefixContextState,
    StalePrefixContextVersion,
    UnknownPrefixContext,
)


def _registry():
    released = []
    registry = PrefixContextRegistry(
        lambda identity, resource: released.append((identity, resource)),
    )
    return registry, released


def test_context_build_activate_serial_tasks_and_close():
    registry, released = _registry()
    identity = registry.begin_build("camera-prefix", resource="kv-v1")

    assert identity.version == 1
    assert registry.snapshot(identity).state is PrefixContextState.BUILDING
    assert registry.current("camera-prefix") is None

    registry.activate(identity)
    assert registry.current("camera-prefix") == identity
    assert registry.snapshot(identity).state is PrefixContextState.ACTIVE

    registry.enqueue_task(identity, 1, payload={"state": [1.0]})
    registry.enqueue_task(identity, 2, payload={"state": [2.0]})
    with pytest.raises(DuplicatePrefixTask):
        registry.enqueue_task(identity, 2)

    first = registry.pin_next_task(identity)
    assert first.task_id == 1
    assert first.payload == {"state": [1.0]}
    assert registry.pin_next_task(identity) is None
    assert registry.acknowledge_task(first, safe=True)
    assert not registry.acknowledge_task(first, safe=True)

    second = registry.pin_next_task(identity)
    assert second.task_id == 2
    assert registry.acknowledge_task(second, safe=True)

    assert registry.close(identity)
    assert not registry.close(identity)
    assert registry.snapshot(identity).state is PrefixContextState.RETIRED
    assert released == [(identity, "kv-v1")]


def test_close_drains_already_accepted_tasks_but_rejects_new_tasks():
    registry, released = _registry()
    identity = registry.begin("session", "kv")
    registry.activate(identity)
    registry.enqueue(identity, "step-1")
    registry.enqueue(identity, "step-2")

    assert registry.close(identity, reason="client_close")
    snapshot = registry.snapshot(identity)
    assert snapshot.state is PrefixContextState.DRAINING
    assert snapshot.close_reason == "client_close"
    assert snapshot.queued_task_ids == ("step-1", "step-2")
    assert released == []

    with pytest.raises(InvalidPrefixContextTransition, match="does not accept"):
        registry.enqueue(identity, "too-late")

    first = registry.pin_next(identity)
    registry.ack(first, safe=True)
    assert registry.snapshot(identity).state is PrefixContextState.DRAINING
    second = registry.pin_next(identity)
    registry.ack(second, safe=True)

    assert registry.snapshot(identity).state is PrefixContextState.RETIRED
    assert released == [(identity, "kv")]


def test_close_waits_for_safe_ack_before_releasing_resource():
    registry, released = _registry()
    identity = registry.begin("session", "prefix-and-scratch")
    registry.activate(identity)
    registry.enqueue(identity, 7)
    pin = registry.pin_next(identity)

    registry.close(identity)
    assert registry.snapshot(identity).state is PrefixContextState.DRAINING
    assert released == []

    registry.ack(pin, safe=True)
    assert registry.snapshot(identity).state is PrefixContextState.RETIRED
    assert released == [(identity, "prefix-and-scratch")]


def test_unsafe_ack_poison_context_and_never_calls_release_callback():
    registry, released = _registry()
    identity = registry.begin("session", "unsafe-kv")
    registry.activate(identity)
    registry.enqueue(identity, 1)
    pin = registry.pin_next(identity)
    registry.close(identity)

    assert registry.ack(pin, safe=False)
    snapshot = registry.snapshot(identity)
    assert snapshot.state is PrefixContextState.POISONED
    assert snapshot.has_pin
    assert "safe acknowledgement" in snapshot.poison_reason
    assert released == []
    assert not registry.close(identity)


def test_poisoned_context_returns_every_queued_task_for_terminal_errors():
    registry, _ = _registry()
    identity = registry.begin("session", "unsafe-kv")
    registry.activate(identity)
    registry.enqueue(identity, 1, payload="active")
    registry.enqueue(identity, 2, payload="queued-2")
    registry.enqueue(identity, 3, payload="queued-3")
    pin = registry.pin_next(identity)

    registry.ack(pin, safe=False)

    tasks = registry.drain_queued_tasks(identity)
    assert [(task.task_id, task.payload) for task in tasks] == [
        (2, "queued-2"),
        (3, "queued-3"),
    ]
    assert registry.snapshot(identity).queued_task_ids == ()


def test_high_frequency_task_history_is_bounded_and_retired_record_is_forgotten():
    registry, _ = _registry()
    identity = registry.begin("session", "kv")
    registry.activate(identity)

    for task_id in range(256):
        registry.enqueue(identity, task_id)
        registry.ack(registry.pin_next(identity), safe=True)

    record = registry._records[identity]
    assert record.known_task_ids == set()
    assert len(record.acked_pin_ids) == 64

    registry.close(identity)
    assert registry.forget_retired(identity)
    with pytest.raises(UnknownPrefixContext):
        registry.snapshot(identity)


def test_replace_is_copy_on_write_and_old_pin_drains_after_publish():
    registry, released = _registry()
    old = registry.begin("session", "kv-v1")
    registry.activate(old)
    registry.enqueue(old, "old-task")
    old_pin = registry.pin_next(old)

    new = registry.begin_replace(old, "kv-v2")
    assert new.version == 2
    assert registry.current("session") == old
    assert registry.snapshot(old).state is PrefixContextState.ACTIVE
    assert registry.snapshot(new).state is PrefixContextState.BUILDING

    registry.activate(new)
    assert registry.current("session") == new
    assert registry.snapshot(new).state is PrefixContextState.ACTIVE
    assert registry.snapshot(old).state is PrefixContextState.DRAINING
    assert released == []

    registry.ack(old_pin, safe=True)
    assert registry.snapshot(old).state is PrefixContextState.RETIRED
    assert released == [(old, "kv-v1")]
    assert registry.snapshot(new).state is PrefixContextState.ACTIVE


def test_provisional_replace_executes_new_version_while_holding_old_kv_until_commit():
    registry, released = _registry()
    old = registry.begin("session", "kv-v1")
    registry.activate(old)
    registry.enqueue(old, "accepted-before-replace")

    new = registry.begin_replace(old, "kv-v2")
    registry.activate_provisional_replace(new)

    assert registry.is_provisional_replace(new)
    assert not registry.is_provisional_replace(old)
    assert registry.current("session") == old
    assert registry.snapshot(old).state is PrefixContextState.DRAINING
    assert registry.snapshot(new).state is PrefixContextState.ACTIVE

    # Both the predecessor's accepted work and the replacement's first action
    # may finish before HTTP resolves ownership.  The hold keeps old KV alive.
    registry.ack(registry.pin_next(old), safe=True)
    registry.enqueue(new, "first-new-action")
    registry.ack(registry.pin_next(new), safe=True)
    assert registry.snapshot(old).state is PrefixContextState.DRAINING
    assert released == []

    registry.commit_provisional_replace(new)

    assert not registry.is_provisional_replace(new)
    assert registry.current("session") == new
    assert registry.snapshot(new).state is PrefixContextState.ACTIVE
    assert registry.snapshot(old).state is PrefixContextState.RETIRED
    assert released == [(old, "kv-v1")]


def test_provisional_replace_rollback_restores_old_and_drains_new_safely():
    registry, released = _registry()
    old = registry.begin("session", "kv-v1")
    registry.activate(old)
    new = registry.begin_replace(old, "kv-v2")
    registry.activate_provisional_replace(new)
    registry.enqueue(new, "in-flight-new-action")
    new_pin = registry.pin_next(new)

    restored = registry.rollback_provisional_replace(new)

    assert restored.identity == old
    assert not registry.is_provisional_replace(new)
    assert registry.current("session") == old
    assert registry.snapshot(old).state is PrefixContextState.ACTIVE
    assert registry.snapshot(new).state is PrefixContextState.DRAINING
    assert released == []
    registry.enqueue(old, "old-version-continues")

    with pytest.raises(InvalidPrefixContextTransition, match="does not accept"):
        registry.enqueue(new, "too-late")

    registry.ack(new_pin, safe=True)
    assert registry.snapshot(new).state is PrefixContextState.RETIRED
    assert registry.snapshot(old).state is PrefixContextState.ACTIVE
    assert released == [(new, "kv-v2")]


def test_provisional_activation_requires_a_replacement():
    registry, released = _registry()
    identity = registry.begin("session", "kv")

    with pytest.raises(
        InvalidPrefixContextTransition,
        match="only a replacement",
    ):
        registry.activate_provisional_replace(identity)

    assert registry.snapshot(identity).state is PrefixContextState.BUILDING
    assert released == []


def test_stale_replacement_cannot_overwrite_a_closed_base():
    registry, released = _registry()
    old = registry.begin("session", "kv-v1")
    registry.activate(old)
    replacement = registry.begin_replace(old, "kv-v2")

    registry.close(old)
    with pytest.raises(StalePrefixContextVersion, match="base"):
        registry.activate(replacement)

    assert registry.close(replacement, reason="abandoned_build")
    assert registry.snapshot(replacement).state is PrefixContextState.RETIRED
    assert released == [(old, "kv-v1"), (replacement, "kv-v2")]


def test_release_callback_failure_poison_context_and_surfaces_error():
    def fail_release(identity, resource):
        raise RuntimeError(f"cannot release {resource}")

    registry = PrefixContextRegistry(fail_release)
    identity = registry.begin("session", "kv")
    registry.activate(identity)

    with pytest.raises(PrefixContextReleaseError) as exc_info:
        registry.close(identity)

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    snapshot = registry.snapshot(identity)
    assert snapshot.state is PrefixContextState.POISONED
    assert "cannot release kv" in snapshot.release_error


def test_only_one_thread_can_acquire_the_serial_task_pin():
    registry, _ = _registry()
    identity = registry.begin("session", "kv")
    registry.activate(identity)
    registry.enqueue(identity, 1)
    results = []

    def pin():
        results.append(registry.pin_next(identity))

    threads = [threading.Thread(target=pin) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    pins = [result for result in results if result is not None]
    assert len(pins) == 1
    assert results.count(None) == 15
    registry.ack(pins[0], safe=True)


def test_poisoned_build_is_not_published_or_released():
    registry, released = _registry()
    identity = registry.begin("session", "partially-built-kv")

    assert registry.poison(identity, "builder lost ownership")
    assert not registry.poison(identity, "duplicate")
    assert registry.current("session") is None
    assert registry.snapshot(identity).state is PrefixContextState.POISONED
    assert released == []
    with pytest.raises(InvalidPrefixContextTransition, match="cannot be reused"):
        registry.begin("session", "unsafe-replacement")
