from types import SimpleNamespace

import torch

from lightllm.server.actionserver.objs import (
    ActionContextOwnerDisposition,
    ActionOutcome,
    ActionRequest,
    ActionTaskIdentity,
    PrefixContextOp,
    PrefixContextRef,
)
from lightllm.server.core.objs import FinishStatus
from lightllm.server.core.objs.req import Req
from lightllm.server.detokenization.decode_req import DecodeReq
from lightllm.server.inference_runtime.action_branch import PrefixKVResource
from lightllm.server.inference_runtime.prefix_context import (
    PrefixContextRegistry,
    PrefixContextState,
)
from lightllm.server.inference_runtime.runtime import VLARequestLifecycle
from lightllm.server.router.model_infer.infer_batch import g_infer_context


class _ImmediateActionBranch:
    def __init__(self):
        self.submissions = []

    @staticmethod
    def has_context_retirement_in_progress():
        return False

    def submit(self, req, **kwargs):
        self.submissions.append((req, kwargs))
        kwargs["completion_callback"](ActionOutcome.SUCCESS, True)
        return True

    @staticmethod
    def publish_request_error(*_args, **_kwargs):
        raise AssertionError("the successful hot-path test published an error")


class _DeferredActionBranch:
    def __init__(self):
        self.submissions = []
        self.errors = []

    @staticmethod
    def has_context_retirement_in_progress():
        return False

    def submit(self, req, **kwargs):
        self.submissions.append((req, kwargs))
        return True

    def publish_request_error(self, req, error):
        self.errors.append((req.req_id, error))


class _OwnerStore:
    def __init__(self, disposition):
        self.disposition = disposition
        self.acks = []

    def get_context_owner_disposition(self, identity):
        return self.disposition

    def mark_context_owner_rank_acked(self, identity, rank):
        self.acks.append((identity, rank))
        return True


class _ControlActionBranch:
    def __init__(self):
        self.successes = []
        self.errors = []

    def publish_control_success(self, req, *, context_version):
        self.successes.append((req.req_id, context_version))

    def publish_request_error(self, req, error):
        self.errors.append((req.req_id, error))


def test_first_context_task_registers_mapping_and_later_ticks_omit_it(monkeypatch):
    resource = PrefixKVResource(
        prefix_len=3,
        local_prefix_mem_indexes=torch.tensor([11, 12, 13], dtype=torch.int32),
        prefix_rank_major=torch.tensor([[11, 12, 13]], dtype=torch.int32),
        local_scratch_mem_indexes=torch.tensor([21, 22, 23], dtype=torch.int32),
        scratch_rank_major=torch.tensor([[21, 22, 23]], dtype=torch.int32),
    )
    contexts = PrefixContextRegistry(lambda *_: None, server_epoch="epoch-1")
    identity = contexts.begin("robot-7", resource)
    contexts.activate(identity)
    contexts.enqueue(identity, 101, payload=101)
    contexts.enqueue(identity, 102, payload=102)

    lifecycle = VLARequestLifecycle.__new__(VLARequestLifecycle)
    lifecycle.backend = SimpleNamespace(logger=SimpleNamespace(exception=lambda *_args, **_kwargs: None))
    lifecycle.contexts = contexts
    lifecycle.action_branch = _ImmediateActionBranch()
    lifecycle._context_identities = {identity}
    lifecycle._routed_context_reqs = {101, 102}
    lifecycle._close_reqs = {}
    lifecycle._new_context_reqs = {}
    requests = {
        101: SimpleNamespace(req_id=101),
        102: SimpleNamespace(req_id=102),
    }
    monkeypatch.setattr(g_infer_context, "requests_mapping", requests)

    lifecycle._dispatch_context_tasks()
    lifecycle._dispatch_context_tasks()

    assert len(lifecycle.action_branch.submissions) == 2
    first = lifecycle.action_branch.submissions[0][1]
    hot = lifecycle.action_branch.submissions[1][1]
    torch.testing.assert_close(first["prefix_rank_major"], resource.prefix_rank_major)
    torch.testing.assert_close(first["scratch_rank_major"], resource.scratch_rank_major)
    assert first["prefix_context_identity"] == identity
    assert hot["prefix_rank_major"] is None
    assert hot["scratch_rank_major"] is None
    assert hot["prefix_context_identity"] == identity


def test_rollback_waits_for_newer_inflight_pin_before_dispatching_old(monkeypatch):
    def resource(offset):
        return PrefixKVResource(
            prefix_len=2,
            local_prefix_mem_indexes=torch.tensor(
                [offset, offset + 1], dtype=torch.int32
            ),
            prefix_rank_major=torch.tensor(
                [[offset, offset + 1]], dtype=torch.int32
            ),
            local_scratch_mem_indexes=torch.tensor(
                [offset + 2, offset + 3], dtype=torch.int32
            ),
            scratch_rank_major=torch.tensor(
                [[offset + 2, offset + 3]], dtype=torch.int32
            ),
        )

    contexts = PrefixContextRegistry(lambda *_: None, server_epoch="epoch-1")
    old = contexts.begin("robot-7", resource(10))
    contexts.activate(old)
    replacement = contexts.begin_replace(old, resource(20))
    contexts.activate_provisional_replace(replacement)
    contexts.enqueue(replacement, 201, payload=201)
    replacement_pin = contexts.pin_next(replacement)

    # HTTP can discard the replacement while its first action is still
    # aborting.  Rollback restores the old public handle, but the newer pin
    # must remain the logical-context barrier until its worker lease is safe.
    contexts.rollback_provisional_replace(replacement)
    contexts.enqueue(old, 101, payload=101)

    lifecycle = VLARequestLifecycle.__new__(VLARequestLifecycle)
    lifecycle.backend = SimpleNamespace(
        logger=SimpleNamespace(exception=lambda *_args, **_kwargs: None)
    )
    lifecycle.contexts = contexts
    lifecycle.action_branch = _DeferredActionBranch()
    lifecycle._context_identities = {old, replacement}
    lifecycle._new_context_reqs = {}
    monkeypatch.setattr(
        g_infer_context,
        "requests_mapping",
        {101: SimpleNamespace(req_id=101)},
    )

    lifecycle._dispatch_context_tasks()

    assert lifecycle.action_branch.submissions == []
    assert contexts.snapshot(old).queued_task_ids == (101,)
    assert contexts.snapshot(replacement).has_pin

    contexts.ack(replacement_pin, safe=True)
    lifecycle._dispatch_context_tasks()

    assert [req.req_id for req, _ in lifecycle.action_branch.submissions] == [101]
    assert replacement not in lifecycle._context_identities


def test_unsafe_predecessor_fails_replace_successor_without_releasing_kv(monkeypatch):
    def resource(offset):
        return PrefixKVResource(
            prefix_len=2,
            local_prefix_mem_indexes=torch.tensor([offset, offset + 1], dtype=torch.int32),
            prefix_rank_major=torch.tensor([[offset, offset + 1]], dtype=torch.int32),
            local_scratch_mem_indexes=torch.tensor([offset + 2, offset + 3], dtype=torch.int32),
            scratch_rank_major=torch.tensor([[offset + 2, offset + 3]], dtype=torch.int32),
        )

    released = []
    contexts = PrefixContextRegistry(
        lambda identity, value: released.append((identity, value)),
        server_epoch="epoch-1",
    )
    old = contexts.begin("robot-7", resource(10))
    contexts.activate(old)
    contexts.enqueue(old, 101, payload=101)
    contexts.enqueue(old, 102, payload=102)

    lifecycle = VLARequestLifecycle.__new__(VLARequestLifecycle)
    lifecycle.backend = SimpleNamespace(logger=SimpleNamespace(exception=lambda *_args, **_kwargs: None))
    lifecycle.contexts = contexts
    lifecycle.action_branch = _DeferredActionBranch()
    lifecycle._context_identities = {old}
    lifecycle._routed_context_reqs = {101, 102}
    lifecycle._close_reqs = {}
    lifecycle._new_context_reqs = {}
    requests = {
        request_id: SimpleNamespace(req_id=request_id)
        for request_id in (101, 102, 201, 202)
    }
    monkeypatch.setattr(g_infer_context, "requests_mapping", requests)

    # Pin the old version before publishing its copy-on-write replacement.
    lifecycle._dispatch_context_tasks()
    replacement = contexts.begin_replace(old, resource(20))
    contexts.activate(replacement)
    contexts.enqueue(replacement, 201, payload=201)
    contexts.enqueue(replacement, 202, payload=202)
    lifecycle._context_identities.add(replacement)
    lifecycle._routed_context_reqs.update({201, 202})

    old_callback = lifecycle.action_branch.submissions[0][1]["completion_callback"]
    old_callback(ActionOutcome.RESTART_REQUIRED, False)

    assert contexts.snapshot(old).state is PrefixContextState.POISONED
    assert contexts.snapshot(replacement).state is PrefixContextState.POISONED
    assert contexts.snapshot(old).queued_task_ids == ()
    assert contexts.snapshot(replacement).queued_task_ids == ()
    assert contexts.current("robot-7") is None
    assert [request_id for request_id, _ in lifecycle.action_branch.errors] == [102, 201, 202]
    assert all(
        "process restart is required" in str(error)
        for _, error in lifecycle.action_branch.errors
    )
    assert released == []

    # Neither the replacement owner nor later REUSE work may cross the
    # poisoned predecessor barrier on a subsequent classifier pass.
    lifecycle._dispatch_context_tasks()
    assert [req.req_id for req, _ in lifecycle.action_branch.submissions] == [101]
    assert released == []


def test_filtered_create_owner_is_closed_when_http_discards_handle(monkeypatch):
    resource = object()
    released = []
    contexts = PrefixContextRegistry(
        lambda identity, value: released.append((identity, value)),
        server_epoch="epoch-1",
    )
    context_identity = contexts.begin("robot-7", resource)
    contexts.activate(context_identity)
    task_identity = ActionTaskIdentity(3, 101, 101)
    owner_store = _OwnerStore(ActionContextOwnerDisposition.DISCARDED)

    lifecycle = VLARequestLifecycle.__new__(VLARequestLifecycle)
    lifecycle.backend = SimpleNamespace(
        rank_in_node=0,
        logger=SimpleNamespace(exception=lambda *_args, **_kwargs: None),
    )
    lifecycle.contexts = contexts
    lifecycle.action_branch = SimpleNamespace(output_store=owner_store)
    lifecycle._new_context_reqs = {
        101: (context_identity, task_identity),
    }
    monkeypatch.setattr(g_infer_context, "requests_mapping", {})

    lifecycle._poll_context_owners()

    assert contexts.snapshot(context_identity).state is PrefixContextState.RETIRED
    assert released == [(context_identity, resource)]
    assert lifecycle._new_context_reqs == {}
    assert owner_store.acks == [(task_identity, 0)]


def test_filtered_create_owner_is_retained_after_public_delivery(monkeypatch):
    released = []
    contexts = PrefixContextRegistry(
        lambda identity, value: released.append((identity, value)),
        server_epoch="epoch-1",
    )
    context_identity = contexts.begin("robot-7", object())
    contexts.activate(context_identity)
    task_identity = ActionTaskIdentity(4, 102, 102)
    owner_store = _OwnerStore(ActionContextOwnerDisposition.DELIVERED)

    lifecycle = VLARequestLifecycle.__new__(VLARequestLifecycle)
    lifecycle.backend = SimpleNamespace(
        rank_in_node=0,
        logger=SimpleNamespace(exception=lambda *_args, **_kwargs: None),
    )
    lifecycle.contexts = contexts
    lifecycle.action_branch = SimpleNamespace(output_store=owner_store)
    lifecycle._new_context_reqs = {
        102: (context_identity, task_identity),
    }
    monkeypatch.setattr(g_infer_context, "requests_mapping", {})

    lifecycle._poll_context_owners()

    assert contexts.snapshot(context_identity).state is PrefixContextState.ACTIVE
    assert released == []
    assert lifecycle._new_context_reqs == {}
    assert owner_store.acks == [(task_identity, 0)]


def test_delivered_replace_commits_new_context_and_retires_old(monkeypatch):
    old_resource = object()
    new_resource = object()
    released = []
    contexts = PrefixContextRegistry(
        lambda identity, value: released.append((identity, value)),
        server_epoch="epoch-1",
    )
    old_identity = contexts.begin("robot-7", old_resource)
    contexts.activate(old_identity)
    new_identity = contexts.begin_replace(old_identity, new_resource)
    contexts.activate_provisional_replace(new_identity)
    task_identity = ActionTaskIdentity(5, 103, 103)
    owner_store = _OwnerStore(ActionContextOwnerDisposition.DELIVERED)

    lifecycle = VLARequestLifecycle.__new__(VLARequestLifecycle)
    lifecycle.backend = SimpleNamespace(
        rank_in_node=0,
        logger=SimpleNamespace(exception=lambda *_args, **_kwargs: None),
    )
    lifecycle.contexts = contexts
    lifecycle.action_branch = SimpleNamespace(output_store=owner_store)
    lifecycle._new_context_reqs = {
        103: (new_identity, task_identity),
    }
    monkeypatch.setattr(g_infer_context, "requests_mapping", {})

    lifecycle._poll_context_owners()

    assert contexts.current("robot-7") == new_identity
    assert contexts.snapshot(old_identity).state is PrefixContextState.RETIRED
    assert contexts.snapshot(new_identity).state is PrefixContextState.ACTIVE
    assert not contexts.is_provisional_replace(new_identity)
    assert released == [(old_identity, old_resource)]
    assert lifecycle._new_context_reqs == {}
    assert owner_store.acks == [(task_identity, 0)]


def test_discarded_replace_restores_old_context_and_retires_new(monkeypatch):
    old_resource = object()
    new_resource = object()
    released = []
    contexts = PrefixContextRegistry(
        lambda identity, value: released.append((identity, value)),
        server_epoch="epoch-1",
    )
    old_identity = contexts.begin("robot-7", old_resource)
    contexts.activate(old_identity)
    new_identity = contexts.begin_replace(old_identity, new_resource)
    contexts.activate_provisional_replace(new_identity)
    task_identity = ActionTaskIdentity(6, 104, 104)
    owner_store = _OwnerStore(ActionContextOwnerDisposition.DISCARDED)

    lifecycle = VLARequestLifecycle.__new__(VLARequestLifecycle)
    lifecycle.backend = SimpleNamespace(
        rank_in_node=0,
        logger=SimpleNamespace(exception=lambda *_args, **_kwargs: None),
    )
    lifecycle.contexts = contexts
    lifecycle.action_branch = SimpleNamespace(output_store=owner_store)
    lifecycle._new_context_reqs = {
        104: (new_identity, task_identity),
    }
    monkeypatch.setattr(g_infer_context, "requests_mapping", {})

    lifecycle._poll_context_owners()

    assert contexts.current("robot-7") == old_identity
    assert contexts.snapshot(old_identity).state is PrefixContextState.ACTIVE
    assert contexts.snapshot(new_identity).state is PrefixContextState.RETIRED
    assert not contexts.is_provisional_replace(new_identity)
    assert released == [(new_identity, new_resource)]
    assert lifecycle._new_context_reqs == {}
    assert owner_store.acks == [(task_identity, 0)]


def test_filtered_unsubmitted_create_releases_adopted_context(monkeypatch):
    resource = object()
    released = []
    contexts = PrefixContextRegistry(
        lambda identity, value: released.append((identity, value)),
        server_epoch="epoch-1",
    )
    context_identity = contexts.begin("robot-7", resource)
    contexts.activate(context_identity)
    contexts.enqueue(context_identity, 105, payload=105)
    task_identity = ActionTaskIdentity(7, 105, 105)

    lifecycle = VLARequestLifecycle.__new__(VLARequestLifecycle)
    lifecycle.backend = SimpleNamespace(
        logger=SimpleNamespace(exception=lambda *_args, **_kwargs: None),
    )
    lifecycle.contexts = contexts
    lifecycle.action_branch = _DeferredActionBranch()
    lifecycle._context_identities = {context_identity}
    lifecycle._new_context_reqs = {
        105: (context_identity, task_identity),
    }
    monkeypatch.setattr(g_infer_context, "requests_mapping", {})

    lifecycle._dispatch_context_tasks()

    assert contexts.current("robot-7") is None
    assert contexts.snapshot(context_identity).state is PrefixContextState.RETIRED
    assert released == [(context_identity, resource)]
    assert lifecycle._new_context_reqs == {}
    assert lifecycle.action_branch.submissions == []


def test_filtered_unsubmitted_replace_rolls_back_to_old_context(monkeypatch):
    old_resource = object()
    new_resource = object()
    released = []
    contexts = PrefixContextRegistry(
        lambda identity, value: released.append((identity, value)),
        server_epoch="epoch-1",
    )
    old_identity = contexts.begin("robot-7", old_resource)
    contexts.activate(old_identity)
    new_identity = contexts.begin_replace(old_identity, new_resource)
    contexts.activate_provisional_replace(new_identity)
    contexts.enqueue(new_identity, 106, payload=106)
    task_identity = ActionTaskIdentity(8, 106, 106)

    lifecycle = VLARequestLifecycle.__new__(VLARequestLifecycle)
    lifecycle.backend = SimpleNamespace(
        logger=SimpleNamespace(exception=lambda *_args, **_kwargs: None),
    )
    lifecycle.contexts = contexts
    lifecycle.action_branch = _DeferredActionBranch()
    lifecycle._context_identities = {old_identity, new_identity}
    lifecycle._new_context_reqs = {
        106: (new_identity, task_identity),
    }
    monkeypatch.setattr(g_infer_context, "requests_mapping", {})

    lifecycle._dispatch_context_tasks()

    assert contexts.current("robot-7") == old_identity
    assert contexts.snapshot(old_identity).state is PrefixContextState.ACTIVE
    assert contexts.snapshot(new_identity).state is PrefixContextState.RETIRED
    assert released == [(new_identity, new_resource)]
    assert lifecycle._new_context_reqs == {}
    assert lifecycle.action_branch.submissions == []


def test_close_reapplies_after_provisional_replace_rolls_back(monkeypatch):
    old_resource = object()
    new_resource = object()
    released = []
    contexts = PrefixContextRegistry(
        lambda identity, value: released.append((identity, value)),
        server_epoch="epoch-1",
    )
    old_identity = contexts.begin("robot-7", old_resource)
    contexts.activate(old_identity)
    new_identity = contexts.begin_replace(old_identity, new_resource)
    contexts.activate_provisional_replace(new_identity)
    close_req = SimpleNamespace(req_id=107)

    lifecycle = VLARequestLifecycle.__new__(VLARequestLifecycle)
    lifecycle.contexts = contexts
    lifecycle.action_branch = _ControlActionBranch()
    lifecycle._close_reqs = {107: old_identity}
    monkeypatch.setattr(
        g_infer_context,
        "requests_mapping",
        {107: close_req},
    )

    # The old identity is still the public handle, but its retirement is held
    # until the provisional replacement owner resolves.
    lifecycle._poll_close_requests()
    assert lifecycle.action_branch.successes == []

    contexts.rollback_provisional_replace(new_identity)
    lifecycle._poll_close_requests()

    assert contexts.current("robot-7") is None
    assert contexts.snapshot(old_identity).state is PrefixContextState.RETIRED
    assert contexts.snapshot(new_identity).state is PrefixContextState.RETIRED
    assert released == [
        (new_identity, new_resource),
        (old_identity, old_resource),
    ]
    assert lifecycle.action_branch.successes == [(107, old_identity.version)]
    assert lifecycle.action_branch.errors == []
    assert lifecycle._close_reqs == {}


def test_immediate_close_of_delivered_replace_waits_for_local_owner_commit():
    contexts = PrefixContextRegistry(lambda *_args: None, server_epoch="epoch-1")
    old_identity = contexts.begin("robot-7", object())
    contexts.activate(old_identity)
    new_identity = contexts.begin_replace(old_identity, object())
    contexts.activate_provisional_replace(new_identity)
    close_req = SimpleNamespace(req_id=108)
    close_action = ActionRequest(
        state=None,
        prefix_context=PrefixContextRef(
            op=PrefixContextOp.CLOSE,
            identity=new_identity,
        ),
    )

    lifecycle = VLARequestLifecycle.__new__(VLARequestLifecycle)
    lifecycle.backend = SimpleNamespace(
        logger=SimpleNamespace(exception=lambda *_args, **_kwargs: None),
    )
    lifecycle.contexts = contexts
    lifecycle._routed_context_reqs = set()
    lifecycle._close_reqs = {}

    # ASGI may have sent the new handle just before this rank observes the
    # shared DELIVERED disposition. The control request parks without error.
    lifecycle._route_context_request(close_req, close_action)
    assert lifecycle._routed_context_reqs == set()
    assert lifecycle._close_reqs == {}

    contexts.commit_provisional_replace(new_identity)
    lifecycle._route_context_request(close_req, close_action)

    assert lifecycle._routed_context_reqs == {108}
    assert lifecycle._close_reqs == {108: new_identity}
    assert contexts.snapshot(new_identity).state is PrefixContextState.RETIRED


def test_zero_token_hot_finish_satisfies_standard_detokenizer_release_path():
    finish_status = FinishStatus()
    shm_req = SimpleNamespace(
        request_id=109,
        group_req_id=109,
        input_len=3,
        shm_prompt_ids=SimpleNamespace(arr=torch.tensor([1, 2, 3])),
        sample_params=SimpleNamespace(
            stop_sequences=SimpleNamespace(to_strings=lambda: []),
        ),
        shm_cur_output_len=-1,
        candetoken_out_len=-1,
        finish_token_index=-1,
        finish_status=FinishStatus(),
        is_aborted=False,
        stop_str_matched=False,
        can_released_mark=False,
        ref_count=1,
        out_tokens_queue=SimpleNamespace(is_empty=lambda: True),
    )
    infer_req = SimpleNamespace(
        finish_status=finish_status,
        shm_req=shm_req,
    )
    lifecycle = VLARequestLifecycle.__new__(VLARequestLifecycle)
    lifecycle.backend = SimpleNamespace(is_master_in_dp=True)

    lifecycle._finish_zero_token_request(infer_req)

    decode_req = DecodeReq(shm_req, is_pd_decode_mode=False)
    assert finish_status.is_finished()
    assert shm_req.shm_cur_output_len == 0
    assert shm_req.candetoken_out_len == 0
    assert shm_req.finish_token_index == shm_req.input_len - 1
    assert decode_req.can_set_release_mark()

    # This is the ordinary detokenizer manager's final hand-off followed by
    # the unchanged HTTP recycler predicate.
    shm_req.can_released_mark = True
    assert Req.can_release(shm_req)
