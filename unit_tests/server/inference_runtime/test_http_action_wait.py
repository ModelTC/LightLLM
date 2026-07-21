import asyncio
from types import SimpleNamespace

import torch

from lightllm.server.actionserver.objs import (
    ActionContextOwnerDisposition,
    ActionOutcome,
    ActionReleaseDecision,
    ActionResponse,
    ActionStatus,
    ActionTaskIdentity,
)
from lightllm.server.httpserver.manager import HttpServerManager


class _RacingRestartStore:
    def __init__(self, identity):
        self.identity = identity

    def get_status(self, slot_index):
        return ActionStatus.HAS_OUTPUT

    def get_identity(self, slot_index):
        return self.identity

    def get_release_decision(self, slot_index, *, identity):
        assert identity == self.identity
        return ActionReleaseDecision.RESTART_REQUIRED

    def read_terminal_response(self, identity, request_id):
        assert identity == self.identity
        return ActionResponse(
            request_id=request_id,
            actions=torch.tensor([[1.0]]),
            action_horizon=1,
            action_dim=1,
            outcome=ActionOutcome.SUCCESS,
        )


class _DelayedRetirementStore:
    def __init__(self, identity):
        self.identity = identity
        self.retired = False
        self.read_calls = 0
        self.release_calls = 0

    def get_status(self, slot_index):
        return ActionStatus.DONE

    def get_identity(self, slot_index):
        return self.identity

    def all_ranks_retired(self, slot_index):
        return self.retired

    def get_release_decision(self, slot_index, *, identity):
        assert identity == self.identity
        return ActionReleaseDecision.RELEASE

    def read_response(self, slot_index, request_id, *, identity):
        assert identity == self.identity
        assert self.retired
        self.read_calls += 1
        return ActionResponse(
            request_id=request_id,
            actions=None,
            action_horizon=0,
            action_dim=0,
            outcome=ActionOutcome.SUCCESS,
        )

    def release_slot(self, identity):
        assert identity == self.identity
        assert self.retired
        self.release_calls += 1
        return True


class _RunningStore:
    def __init__(self, identity):
        self.identity = identity

    def get_status(self, slot_index):
        return ActionStatus.RUNNING

    def get_identity(self, slot_index):
        return self.identity

    def get_release_decision(self, slot_index, *, identity):
        assert identity == self.identity
        return ActionReleaseDecision.WAIT


class _OwnerDispositionStore:
    def __init__(self):
        self.calls = []

    def mark_context_owner_disposition(self, identity, disposition):
        self.calls.append((identity, disposition))
        return True


def test_http_surfaces_restart_when_success_payload_races_safe_ack_deadline():
    slot_index = 3
    request_id = 41
    identity = ActionTaskIdentity(slot_index, request_id, request_id)
    manager = object.__new__(HttpServerManager)
    manager.action_output_store = _RacingRestartStore(identity)
    manager.disable_abort = True
    req = SimpleNamespace(
        index_in_shm_mem=slot_index,
        request_id=request_id,
    )
    req_status = SimpleNamespace(
        group_req_objs=SimpleNamespace(shm_req_objs=[req]),
        aborted=False,
        mark_action_output_consumed=lambda: None,
    )
    action_request = SimpleNamespace(
        timeout=1.0,
        request_id="external-41",
        raw_state=None,
    )

    response = asyncio.run(
        manager._wait_to_action_output(
            request_id,
            req_status,
            action_request,
            None,
        )
    )

    assert response.actions is None
    assert response.outcome is ActionOutcome.RESTART_REQUIRED
    assert response.restart_required
    assert response.error_info is not None


def test_http_does_not_release_done_output_until_every_rank_retires():
    async def scenario():
        slot_index = 5
        request_id = 52
        identity = ActionTaskIdentity(slot_index, request_id, request_id)
        store = _DelayedRetirementStore(identity)
        consumed = []
        manager = object.__new__(HttpServerManager)
        manager.action_output_store = store
        manager.disable_abort = True
        manager.latest_success_infer_time_mark = SimpleNamespace(set_value=lambda value: None)
        req = SimpleNamespace(
            index_in_shm_mem=slot_index,
            request_id=request_id,
        )
        req_status = SimpleNamespace(
            group_req_objs=SimpleNamespace(shm_req_objs=[req]),
            aborted=False,
            mark_action_output_consumed=lambda: consumed.append(True),
        )
        action_request = SimpleNamespace(
            timeout=1.0,
            request_id="external-52",
            raw_state=None,
        )

        task = asyncio.create_task(
            manager._wait_to_action_output(
                request_id,
                req_status,
                action_request,
                None,
            )
        )
        await asyncio.sleep(0.01)
        assert not task.done()
        assert store.read_calls == 0
        assert store.release_calls == 0

        store.retired = True
        response = await asyncio.wait_for(task, timeout=0.2)

        assert response.outcome is ActionOutcome.SUCCESS
        assert store.read_calls == 1
        assert store.release_calls == 1
        assert consumed == [True]

    asyncio.run(scenario())


def test_action_waiter_observes_the_normal_shm_abort_flag():
    slot_index = 7
    request_id = 61
    identity = ActionTaskIdentity(slot_index, request_id, request_id)
    manager = object.__new__(HttpServerManager)
    manager.action_output_store = _RunningStore(identity)
    manager.disable_abort = True
    req = SimpleNamespace(
        index_in_shm_mem=slot_index,
        request_id=request_id,
        is_aborted=True,
    )
    req_status = SimpleNamespace(
        group_req_objs=SimpleNamespace(shm_req_objs=[req]),
        aborted=False,
    )
    action_request = SimpleNamespace(
        timeout=1.0,
        request_id="external-61",
        raw_state=None,
    )

    try:
        asyncio.run(
            manager._wait_to_action_output(
                request_id,
                req_status,
                action_request,
                None,
            )
        )
    except RuntimeError as exc:
        assert "was aborted" in str(exc)
    else:
        raise AssertionError("action waiter ignored the ShmReq abort flag")


def test_normal_abort_keeps_the_existing_req_status_wakeup_semantics():
    async def scenario():
        group_request_id = 73
        req = SimpleNamespace(is_aborted=False)
        event = asyncio.Event()
        req_status = SimpleNamespace(
            group_req_objs=SimpleNamespace(
                group_req_id=group_request_id,
                shm_req_objs=[req],
            ),
            aborted=False,
            event=event,
            action_output_pending=False,
        )
        manager = object.__new__(HttpServerManager)
        manager.req_id_to_out_inf = {group_request_id: req_status}

        assert await manager.abort(group_request_id)
        assert req.is_aborted
        assert not req_status.aborted
        assert not event.is_set()

    asyncio.run(scenario())


def test_public_api_resolves_persistent_context_owner_before_slot_release():
    identity = ActionTaskIdentity(8, 81, 81)
    req_status = SimpleNamespace(
        action_context_owner_identity=identity,
        action_context_owner_disposition=None,
        action_output_ready_to_release=False,
        action_output_discarded=False,
    )
    store = _OwnerDispositionStore()
    manager = object.__new__(HttpServerManager)
    manager._context_owner_waiters = {identity: req_status}
    manager.action_output_store = store

    assert manager.resolve_action_context_owner(identity, delivered=True)
    assert (
        req_status.action_context_owner_disposition
        is ActionContextOwnerDisposition.DELIVERED
    )
    assert req_status.action_output_ready_to_release
    assert not req_status.action_output_discarded
    assert store.calls == [
        (identity, ActionContextOwnerDisposition.DELIVERED)
    ]

    manager._mark_action_output_discarded(req_status)
    assert not req_status.action_output_discarded
    assert store.calls == [
        (identity, ActionContextOwnerDisposition.DELIVERED)
    ]
