from types import SimpleNamespace

import torch

from lightllm.server.actionserver.objs import (
    ActionOutcome,
    ActionReleaseDecision,
    ActionStatus,
    ActionTaskIdentity,
)
from lightllm.server.inference_runtime.action_branch import (
    ActionBranchRuntime,
    _PendingAction,
)
from lightllm.server.router.model_infer.infer_batch import g_infer_context


class _MemoryManager:
    def __init__(self):
        self.freed = []

    def free(self, indexes):
        self.freed.append(indexes.clone())


class _Store:
    def __init__(self, identity, *, decision=ActionReleaseDecision.RELEASE):
        self.identity = identity
        self.decision = decision
        self.status = ActionStatus.HAS_OUTPUT
        self.rank_consumed = False
        self.done_observed = False
        self.rank_retired = False

    def get_status(self, _slot):
        return self.status

    def get_release_decision(self, _slot, *, identity):
        assert identity == self.identity
        return self.decision

    def read_terminal_response(self, identity, _request_id):
        assert identity == self.identity
        outcome = (
            ActionOutcome.RESTART_REQUIRED
            if self.decision is ActionReleaseDecision.RESTART_REQUIRED
            else ActionOutcome.SUCCESS
        )
        return SimpleNamespace(outcome=outcome)

    def matches(self, identity):
        return identity == self.identity

    def mark_rank_consumed(self, _slot, _rank, *, identity):
        assert identity == self.identity
        self.rank_consumed = True
        return True

    def all_ranks_consumed(self, _slot):
        return self.rank_consumed

    def acknowledge(self, _slot, *, identity):
        assert identity == self.identity
        self.status = ActionStatus.DONE
        return True

    def mark_rank_done_observed(self, _slot, _rank, *, identity):
        assert identity == self.identity
        if self.status is not ActionStatus.DONE:
            return False
        self.done_observed = True
        return True

    def all_ranks_done_observed(self, _slot):
        return self.done_observed

    def mark_rank_retired(self, _slot, _rank, *, identity):
        assert identity == self.identity
        if not self.done_observed:
            return False
        self.rank_retired = True
        return True

    def all_ranks_retired(self, _slot):
        return self.rank_retired


def _runtime(monkeypatch, *, release_scratch=True, decision=None, callback=None):
    identity = ActionTaskIdentity(3, 41, 41)
    store = _Store(
        identity,
        decision=decision or ActionReleaseDecision.RELEASE,
    )
    memory = _MemoryManager()
    backend = SimpleNamespace(
        is_master_in_node=True,
        rank_in_node=0,
        model=SimpleNamespace(mem_manager=memory),
    )
    runtime = ActionBranchRuntime(backend)
    runtime.output_store = store
    req = SimpleNamespace(req_id=41, infer_aborted=False, shm_req=SimpleNamespace(is_aborted=False))
    monkeypatch.setattr(g_infer_context, "requests_mapping", {41: req})
    runtime._pending[41] = _PendingAction(
        req_id=41,
        slot_index=3,
        identity=identity,
        scratch_mem_indexes=torch.tensor([7, 8], dtype=torch.int32),
        deadline_at=float("inf"),
        ack_grace_seconds=1.0,
        request_id="external-41",
        release_scratch=release_scratch,
        completion_callback=callback,
    )
    return runtime, store, memory


def test_safe_ack_completes_task_and_releases_task_owned_scratch(monkeypatch):
    runtime, store, memory = _runtime(monkeypatch)

    runtime.poll()

    assert store.status is ActionStatus.DONE
    assert store.done_observed
    assert store.rank_retired
    assert runtime.completion(41) is ActionOutcome.SUCCESS
    torch.testing.assert_close(
        memory.freed[0],
        torch.tensor([7, 8], dtype=torch.int32),
    )


def test_context_owned_scratch_is_reused_only_after_safe_callback(monkeypatch):
    callbacks = []
    runtime, store, memory = _runtime(
        monkeypatch,
        release_scratch=False,
        callback=lambda outcome, safe: callbacks.append((outcome, safe)),
    )

    runtime.poll()

    assert store.status is ActionStatus.DONE
    assert store.done_observed
    assert store.rank_retired
    assert memory.freed == []
    assert callbacks == [(ActionOutcome.SUCCESS, True)]


def test_restart_required_poison_callback_retains_all_resources(monkeypatch):
    callbacks = []
    runtime, _, memory = _runtime(
        monkeypatch,
        release_scratch=False,
        decision=ActionReleaseDecision.RESTART_REQUIRED,
        callback=lambda outcome, safe: callbacks.append((outcome, safe)),
    )

    runtime.poll()
    runtime.poll()

    assert runtime.has_pending(41)
    assert not runtime.is_completed(41)
    assert memory.freed == []
    assert callbacks == [(ActionOutcome.RESTART_REQUIRED, False)]
