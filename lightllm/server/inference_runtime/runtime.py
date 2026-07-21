from __future__ import annotations

from functools import partial

from lightllm.server.actionserver.objs import (
    ActionContextOwnerDisposition,
    ActionOutcome,
    ActionRequest,
    ActionTaskIdentity,
    PrefixContextIdentity,
    PrefixContextOp,
)
from lightllm.server.core.objs import FinishStatus
from lightllm.server.router.model_infer.infer_batch import InferReq, g_infer_context
from lightllm.utils.envs_utils import get_unique_server_name

from .output_plan import OutputPlan
from .prefix_context import (
    PrefixContextRegistry,
    PrefixContextState,
    PrefixTaskPin,
)


class VLARequestLifecycle:
    """Thin Pi request-lifecycle extension mounted at request classification.

    Ordinary prefill, sampling, post handling, decode, and final cleanup stay
    untouched.  Persistent contexts adopt physical prefix pages after the
    normal prefill has committed; high-frequency action requests then remain
    parked in the classifier and never enter a target-model forward.
    """

    def __init__(self, backend) -> None:
        self.backend = backend
        from .action_branch import ActionBranchRuntime

        self.action_branch = ActionBranchRuntime(backend)
        self.contexts = PrefixContextRegistry(
            self._release_context_resource,
            server_epoch=get_unique_server_name(),
        )
        self._routed_context_reqs: set[int] = set()
        self._context_identities: set[PrefixContextIdentity] = set()
        self._close_reqs: dict[int, PrefixContextIdentity] = {}
        # CREATE/REPLACE owns a newly committed handle until its HTTP request
        # completes.  If that request is aborted before the client can receive
        # the handle, close it automatically instead of orphaning target KV.
        self._new_context_reqs: dict[
            int,
            tuple[PrefixContextIdentity, ActionTaskIdentity],
        ] = {}

    def init(self) -> None:
        self.action_branch.init()

    @staticmethod
    def output_plan(req: InferReq) -> OutputPlan:
        params = req.multimodal_params or {}
        return OutputPlan.resolve(
            params.get("outputs"),
            legacy_action_requested=params.get("action") is not None,
        )

    @classmethod
    def action_request(cls, req: InferReq) -> ActionRequest | None:
        if not cls.output_plan(req).wants_action:
            return None
        value = (req.multimodal_params or {}).get("action")
        if value is None:
            return None
        return value if isinstance(value, ActionRequest) else ActionRequest.from_dict(value)

    def poll(self) -> None:
        self.action_branch.poll()

        live_ids = set(g_infer_context.infer_req_ids)
        self._routed_context_reqs.intersection_update(live_ids)
        for req_id in tuple(self._close_reqs):
            if req_id not in live_ids:
                self._close_reqs.pop(req_id, None)
        for completed_id in self.action_branch.completed_ids():
            if completed_id not in live_ids:
                self.action_branch.forget(completed_id)
                self._routed_context_reqs.discard(completed_id)

        # ``prefill_normal`` synchronizes before returning. On this following
        # classifier pass cur_kv_len is therefore a safe commit boundary.
        for request_id in tuple(g_infer_context.infer_req_ids):
            req = g_infer_context.requests_mapping.get(request_id)
            if req is None:
                continue
            action = self.action_request(req)
            if action is None:
                continue
            if action.context_op is PrefixContextOp.ONESHOT:
                self._poll_oneshot(req)
            else:
                self._route_context_request(req, action)

            if self.action_branch.is_completed(req.req_id) and action.context_op in {
                PrefixContextOp.REUSE,
                PrefixContextOp.CLOSE,
            }:
                self._finish_zero_token_request(req)

        self._poll_close_requests()
        self._dispatch_context_tasks()
        self._poll_context_owners()

    def _poll_oneshot(self, req: InferReq) -> None:
        if (
            req.cur_kv_len >= req.shm_req.input_len
            and not self.action_branch.has_pending(req.req_id)
            and not self.action_branch.is_completed(req.req_id)
        ):
            self.action_branch.submit(req, prefix_len=req.shm_req.input_len)

    def _route_context_request(
        self,
        req: InferReq,
        action: ActionRequest,
    ) -> None:
        if req.req_id in self._routed_context_reqs:
            return
        identity = action.context_identity
        resource = None
        new_identity = None
        try:
            if (
                identity is not None
                and action.context_op
                in {
                    PrefixContextOp.REPLACE,
                    PrefixContextOp.CLOSE,
                }
                and self.contexts.is_provisional_replace(identity)
            ):
                # The client can receive a REPLACE handle just before this
                # target rank polls the HTTP owner disposition.  Park an
                # immediate follow-up control/prefix update instead of
                # spuriously reporting that newly issued version as stale.
                return
            if action.context_op is PrefixContextOp.REUSE:
                assert identity is not None
                self.contexts.enqueue_task(identity, req.req_id, req.req_id)
                self._context_identities.add(identity)
                self._routed_context_reqs.add(req.req_id)
                return

            if action.context_op is PrefixContextOp.CLOSE:
                assert identity is not None
                current = self.contexts.current(identity.context_id)
                if current is not None and current != identity:
                    raise ValueError(f"stale prefix context {identity}; current version is {current.version}")
                self.contexts.close(identity, reason="client_close")
                self._routed_context_reqs.add(req.req_id)
                self._close_reqs[req.req_id] = identity
                return

            if req.cur_kv_len < req.shm_req.input_len:
                return
            if self.output_plan(req).wants_text:
                raise ValueError("persistent prefix creation/replacement is action-only")
            resource = self.action_branch.capture_context_resource(
                req,
                prefix_len=req.shm_req.input_len,
            )
            if action.context_op is PrefixContextOp.CREATE:
                assert action.context_id is not None
                new_identity = self.contexts.begin_build(
                    action.context_id,
                    resource,
                )
                self.contexts.activate(new_identity)
            elif action.context_op is PrefixContextOp.REPLACE:
                assert identity is not None
                new_identity = self.contexts.begin_replace(identity, resource)
                self.contexts.activate_provisional_replace(new_identity)
            else:
                raise ValueError(f"unsupported prefix context op {action.context_op}")
            self.contexts.enqueue_task(new_identity, req.req_id, req.req_id)
            self._context_identities.add(new_identity)
            self._routed_context_reqs.add(req.req_id)
            self._new_context_reqs[req.req_id] = (
                new_identity,
                self.action_branch.task_identity(req),
            )
        except Exception as exc:
            if new_identity is not None:
                try:
                    if self.contexts.is_provisional_replace(new_identity):
                        self.contexts.rollback_provisional_replace(new_identity)
                    else:
                        self.contexts.close(
                            new_identity,
                            reason="context_build_failed",
                        )
                except Exception:
                    self.backend.logger.exception("failed to discard prefix context %s", new_identity)
            elif resource is not None:
                try:
                    self.action_branch.release_context_resource(resource)
                except Exception:
                    self.backend.logger.exception("failed to release uncommitted prefix resource")
            self.backend.logger.exception("failed to route prefix-context request %s", req.req_id)
            self._routed_context_reqs.add(req.req_id)
            self.action_branch.publish_request_error(req, exc)

    def _dispatch_context_tasks(self) -> None:
        if self.action_branch.has_context_retirement_in_progress():
            return
        identities = tuple(
            sorted(
                self._context_identities,
                key=lambda value: (value.context_id, value.version),
            )
        )
        # A provisional REPLACE can be discarded while its first task is
        # still aborting.  Rollback makes the older version ACTIVE again, but
        # it must not acquire another pin until the discarded version proves
        # its worker lease safe.  Pre-scan all versions so a higher-version
        # pin blocks the complete logical context, independent of sort order.
        blocked_context_ids = set()
        for identity in identities:
            try:
                if self.contexts.snapshot(identity).has_pin:
                    blocked_context_ids.add(identity.context_id)
            except Exception:
                continue

        for identity in identities:
            try:
                snapshot = self.contexts.snapshot(identity)
            except Exception:
                self._context_identities.discard(identity)
                continue
            if snapshot.state is PrefixContextState.RETIRED:
                self._context_identities.discard(identity)
                try:
                    self.contexts.forget_retired(identity)
                except Exception:
                    self.backend.logger.exception("failed to forget retired prefix context %s", identity)
                continue
            if snapshot.state is PrefixContextState.POISONED:
                # A poisoned predecessor is a permanent observation barrier.
                # Fail every successor instead of leaving its accepted tasks
                # parked behind that barrier forever.
                self._fail_poisoned_context_chain(
                    identity,
                    RuntimeError(
                        "prefix context is poisoned because KV safety was not "
                        "proved; process restart is required"
                    ),
                )
                blocked_context_ids.add(identity.context_id)
                continue
            # A replacement is an observation barrier: every task already
            # accepted by the predecessor must retire before the new version
            # may run. This also keeps one active task per logical context,
            # rather than merely one per concrete version.
            if identity.context_id in blocked_context_ids:
                continue
            if snapshot.has_pin:
                blocked_context_ids.add(identity.context_id)
                continue
            pin = self.contexts.pin_next_task(identity)
            if pin is None:
                # An idle provisional predecessor keeps its rollback KV but
                # must not prevent the replacement's first action from
                # running. A pinned/queued predecessor is handled first and
                # becomes the barrier below.
                continue
            blocked_context_ids.add(identity.context_id)
            req = g_infer_context.requests_mapping.get(int(pin.payload))
            if req is None:
                self._discard_unsubmitted_context_task(pin)
                continue
            resource = self.contexts.resource(identity)
            completion_callback = partial(
                self._complete_context_task,
                pin,
                resource,
            )
            register_remote_mapping = not resource.remote_registered
            try:
                submitted = self.action_branch.submit(
                    req,
                    prefix_len=resource.prefix_len,
                    prefix_rank_major=(resource.prefix_rank_major if register_remote_mapping else None),
                    scratch_mem_indexes=resource.local_scratch_mem_indexes,
                    scratch_rank_major=(resource.scratch_rank_major if register_remote_mapping else None),
                    completion_callback=completion_callback,
                    context_version=identity.version,
                    prefix_context_identity=identity,
                    context_owner_required=req.req_id in self._new_context_reqs,
                )
            except Exception as exc:
                self.backend.logger.exception("failed to submit context task %s", pin.task_id)
                if self.action_branch.has_pending(req.req_id):
                    # ``submit`` installed the callback before dispatch. Its
                    # safe/unsafe terminal ACK remains authoritative.
                    continue
                self.action_branch.publish_request_error(
                    req,
                    exc,
                    completion_callback=completion_callback,
                )
                continue
            if not submitted:
                self.contexts.acknowledge_task(pin, safe=True)

    def _discard_unsubmitted_context_task(self, pin: PrefixTaskPin) -> None:
        """Retire a task whose ordinary InferReq disappeared before submit.

        CREATE/REPLACE has already adopted target KV by this point.  Resolve
        that provisional ownership before releasing the serial task pin, so
        an aborted request cannot orphan a new context (or leave both sides of
        a REPLACE transaction held forever).
        """

        req_id = int(pin.payload)
        owner = self._new_context_reqs.get(req_id)
        try:
            if owner is not None:
                context_identity, _ = owner
                if context_identity != pin.context:
                    raise RuntimeError(
                        "context owner identity does not match its queued task"
                    )
                if self.contexts.is_provisional_replace(context_identity):
                    self.contexts.rollback_provisional_replace(context_identity)
                else:
                    self.contexts.close(
                        context_identity,
                        reason="owner_request_disappeared_before_submit",
                    )
            self.contexts.acknowledge_task(pin, safe=True)
        except Exception:
            # Keep both the owner record and task pin when cleanup cannot be
            # proved.  Silently forgetting either could make physical KV
            # reusable while another component still believes it is owned.
            self.backend.logger.exception(
                "failed to discard unsubmitted prefix-context task %s",
                pin.task_id,
            )
            return
        self._new_context_reqs.pop(req_id, None)

    def _complete_context_task(
        self,
        pin: PrefixTaskPin,
        resource,
        outcome: ActionOutcome,
        safe: bool,
    ) -> None:
        try:
            if not safe and self.contexts.is_provisional_replace(pin.context):
                # The replacement can no longer be delivered, but its base
                # KV is still safe. Restore the client-visible predecessor
                # before poisoning only the failed provisional version.
                self.contexts.rollback_provisional_replace(pin.context)
            if safe and outcome is ActionOutcome.SUCCESS:
                # A successful result proves every action rank resolved the
                # registration.  Later state ticks can now omit O(prefix_len)
                # physical mappings from both ZMQ and RPyC payloads.
                resource.remote_registered = True
            self.contexts.acknowledge_task(pin, safe=safe)
            if not safe:
                self._new_context_reqs.pop(int(pin.payload), None)
                error = RuntimeError(
                    "prefix context is poisoned because an action worker did "
                    "not prove KV release; process restart is required"
                )
                self._fail_poisoned_context_chain(pin.context, error)
        except Exception:
            self.backend.logger.exception(
                "failed to retire context task %s with outcome %s",
                pin.task_id,
                outcome.name,
            )

    def _fail_poisoned_context_chain(
        self,
        predecessor: PrefixContextIdentity,
        error: Exception,
    ) -> None:
        """Fail work blocked behind an unsafe immutable context version.

        REPLACE is copy-on-write, so a successor may already be ACTIVE while
        its predecessor is still draining.  Once that predecessor is unsafe,
        the successor can never cross the serial observation barrier.  Poison
        every published successor, drain its accepted requests to terminal
        errors, and deliberately retain all associated KV resources until a
        process restart.
        """

        identities = sorted(
            {
                predecessor,
                *(
                    identity
                    for identity in self._context_identities
                    if identity.context_id == predecessor.context_id
                    and identity.version >= predecessor.version
                ),
            },
            key=lambda identity: identity.version,
        )
        for identity in identities:
            snapshot = self.contexts.snapshot(identity)
            if snapshot.state is PrefixContextState.RETIRED:
                continue
            if snapshot.state is not PrefixContextState.POISONED:
                self.contexts.poison(
                    identity,
                    "a predecessor did not prove KV safety; process restart is required",
                )
            for task in self.contexts.drain_queued_tasks(identity):
                self._new_context_reqs.pop(int(task.payload), None)
                req = g_infer_context.requests_mapping.get(int(task.payload))
                if req is not None:
                    self.action_branch.publish_request_error(req, error)

    def _poll_close_requests(self) -> None:
        for req_id, identity in tuple(self._close_reqs.items()):
            req = g_infer_context.requests_mapping.get(req_id)
            if req is None:
                self._close_reqs.pop(req_id, None)
                continue
            try:
                state = self.contexts.snapshot(identity).state
                if state is PrefixContextState.POISONED:
                    raise RuntimeError("prefix context cannot close safely; process restart is required")
                if state is PrefixContextState.ACTIVE:
                    # CLOSE may have arrived while a provisional REPLACE kept
                    # the public predecessor in DRAINING state.  If that
                    # replacement is later rolled back, apply the deferred
                    # close to the restored predecessor instead of waiting
                    # forever for a retirement that has been cancelled.
                    self.contexts.close(identity, reason="client_close")
                    state = self.contexts.snapshot(identity).state
                if state is not PrefixContextState.RETIRED:
                    continue
                self.action_branch.publish_control_success(
                    req,
                    context_version=identity.version,
                )
            except Exception as exc:
                self.action_branch.publish_request_error(req, exc)
            self._close_reqs.pop(req_id, None)

    def _poll_context_owners(self) -> None:
        """Resolve ownership of a newly committed CREATE/REPLACE handle.

        The InferReq may already have passed through the ordinary finish/filter
        path.  Keep this small task/context identity record until HTTP either
        assembles the public response or discards it.  A discarded owner is
        closed on every target rank before the shared ShmReq slot can be
        recycled.
        """

        for req_id, (context_identity, task_identity) in tuple(
            self._new_context_reqs.items()
        ):
            disposition = self.action_branch.output_store.get_context_owner_disposition(
                task_identity
            )
            if disposition is ActionContextOwnerDisposition.DISCARDED:
                try:
                    if self.contexts.is_provisional_replace(context_identity):
                        self.contexts.rollback_provisional_replace(
                            context_identity
                        )
                    else:
                        self.contexts.close(
                            context_identity,
                            reason="owner_response_discarded",
                        )
                except Exception:
                    self.backend.logger.exception(
                        "failed to close unclaimed prefix context %s",
                        context_identity,
                    )
                    continue
            elif disposition is ActionContextOwnerDisposition.DELIVERED:
                try:
                    if self.contexts.is_provisional_replace(context_identity):
                        self.contexts.commit_provisional_replace(
                            context_identity
                        )
                except Exception:
                    self.backend.logger.exception(
                        "failed to commit delivered prefix context %s",
                        context_identity,
                    )
                    continue
            else:
                continue

            if not self.action_branch.output_store.mark_context_owner_rank_acked(
                task_identity,
                self.backend.rank_in_node,
            ):
                continue
            self._new_context_reqs.pop(req_id, None)

    def _release_context_resource(self, identity, resource) -> None:
        self.action_branch.release_remote_context(identity)
        self.action_branch.release_context_resource(resource)

    def should_defer_release(self, req: InferReq) -> bool:
        action = self.action_request(req)
        if action is None:
            return False
        if self.action_branch.is_completed(req.req_id):
            return False
        if self._is_aborted(req):
            if self.action_branch.has_pending(req.req_id):
                self.action_branch.abort(req)
                return True
            return False
        return self.action_branch.has_pending(req.req_id) or (
            action.context_op is not PrefixContextOp.ONESHOT and req.req_id in self._routed_context_reqs
        )

    def is_schedulable(self, req: InferReq) -> bool:
        if req.infer_aborted:
            if self.action_branch.has_pending(req.req_id):
                self.action_branch.abort(req)
            return False
        action = self.action_request(req)
        if action is not None and action.context_op in {
            PrefixContextOp.REUSE,
            PrefixContextOp.CLOSE,
        }:
            return False
        if (
            action is not None
            and action.context_op is not PrefixContextOp.ONESHOT
            and req.req_id in self._routed_context_reqs
        ):
            return False
        plan = self.output_plan(req)
        if plan.wants_text:
            return not req.finish_status.is_finished()
        return req.cur_kv_len < req.shm_req.input_len

    def can_pause(self, req: InferReq) -> bool:
        return not self.action_branch.has_pending(req.req_id)

    @staticmethod
    def _is_aborted(req: InferReq) -> bool:
        return req.infer_aborted or bool(getattr(req.shm_req, "is_aborted", False))

    def _finish_zero_token_request(self, req: InferReq) -> None:
        if req.finish_status.is_finished():
            return
        req.finish_status.set_status(FinishStatus.FINISHED_STOP)
        if self.backend.is_master_in_dp:
            req.shm_req.shm_cur_output_len = 0
            req.shm_req.candetoken_out_len = 0
            req.shm_req.finish_token_index = req.shm_req.input_len - 1
            req.shm_req.finish_status = req.finish_status


__all__ = ["VLARequestLifecycle"]
