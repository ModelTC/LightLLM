from __future__ import annotations

import math
import pickle
import threading
import time
from dataclasses import dataclass
from typing import Callable

import torch
import torch.distributed as dist
import zmq

from lightllm.models.pi0.config import Pi0VLAConfig
from lightllm.server.actionserver.objs import (
    ActionControlKind,
    ActionControlRequest,
    ActionExpertTask,
    ActionOutcome,
    ActionPrefixContextRelease,
    ActionReleaseDecision,
    ActionRequest,
    ActionStatus,
    ActionTaskIdentity,
    PrefixContextIdentity,
)
from lightllm.server.actionserver.shared_store import ActionOutputStore
from lightllm.server.router.model_infer.infer_batch import InferReq, g_infer_context


@dataclass
class _PendingAction:
    req_id: int
    slot_index: int
    identity: ActionTaskIdentity
    scratch_mem_indexes: torch.Tensor
    deadline_at: float
    ack_grace_seconds: float
    request_id: str | None
    terminal_observed: bool = False
    rank_consumed: bool = False
    scratch_released: bool = False
    control_sent: bool = False
    control_deadline_at: float | None = None
    outcome: ActionOutcome | None = None
    release_scratch: bool = True
    completion_callback: Callable[[ActionOutcome, bool], None] | None = None
    callback_notified: bool = False
    done_observed: bool = False
    rank_retired: bool = False


@dataclass
class PrefixKVResource:
    """Router-owned immutable prefix plus serially reused action scratch."""

    prefix_len: int
    local_prefix_mem_indexes: torch.Tensor
    prefix_rank_major: torch.Tensor
    local_scratch_mem_indexes: torch.Tensor
    scratch_rank_major: torch.Tensor
    remote_registered: bool = False

    @property
    def scratch_len(self) -> int:
        return int(self.local_scratch_mem_indexes.numel())


class ActionBranchRuntime:
    """Asynchronous action branch mounted on an ordinary model backend.

    The target backend remains the sole allocator of prefix and action-scratch
    physical KV.  Action workers receive immutable rank-major mappings and
    install them in their own logical request table; this runtime never writes
    an action suffix into the target model's ``req_to_token_indexs``.
    """

    def __init__(self, backend) -> None:
        self.backend = backend
        self.config: Pi0VLAConfig | None = None
        self.output_store: ActionOutputStore | None = None
        self._zmq_context = None
        self._socket = None
        self._pending: dict[int, _PendingAction] = {}
        self._completed: dict[int, ActionOutcome] = {}
        self._lock = threading.RLock()

    def init(self) -> None:
        self.config = Pi0VLAConfig.from_start_args(self.backend.args)
        self.output_store = ActionOutputStore.from_args(
            self.backend.args,
            initialize=self.backend.is_master_in_node,
        )
        if self.backend.is_master_in_node:
            self._zmq_context = zmq.Context(1)
            self._socket = self._zmq_context.socket(zmq.PUSH)
            self._socket.setsockopt(zmq.LINGER, 0)
            self._socket.setsockopt(zmq.IMMEDIATE, 1)
            self._socket.setsockopt(zmq.SNDTIMEO, 1000)
            self._socket.connect(f"{self.backend.args.zmq_mode}127.0.0.1:{self.backend.args.action_port}")

        self.backend.model.mem_manager.write_to_shm(
            req_manager=self.backend.model.req_manager,
        )
        dist.barrier(group=self.backend.node_nccl_group)

    def has_pending(self, req_id: int) -> bool:
        with self._lock:
            return req_id in self._pending

    def has_context_retirement_in_progress(self) -> bool:
        """Whether a context callback ran before every TP rank retired."""

        with self._lock:
            return any(
                pending.completion_callback is not None
                and pending.callback_notified
                and not self.output_store.all_ranks_retired(pending.slot_index)
                for pending in self._pending.values()
            )

    def capture_context_resource(
        self,
        req: InferReq,
        *,
        prefix_len: int,
    ) -> PrefixKVResource:
        """Adopt committed target KV from a prefix-only InferReq.

        The physical pages move to the returned context resource.  The
        ordinary request retains its logical row but has ``cur_kv_len=0``, so
        the unchanged LLM filter path later releases only the req_idx/ShmReq.
        """

        mapping = self.backend.model.req_manager.req_to_token_indexs
        if prefix_len <= 0 or req.cur_kv_len < prefix_len:
            raise ValueError("cannot adopt an uncommitted action prefix")
        local_prefix = mapping[req.req_idx, :prefix_len].detach().to(dtype=torch.int32).clone()
        suffix_length = self.config.action_horizon + (0 if self.config.is_pi05 else 1)
        scratch = self.backend.model.mem_manager.alloc(suffix_length)
        if not self._all_ranks_true(scratch is not None):
            if scratch is not None:
                self.backend.model.mem_manager.free(scratch)
            raise MemoryError("not enough target KV memory for context scratch")
        assert scratch is not None
        try:
            prefix_rank_major, scratch_rank_major = self._gather_rank_mappings(
                req=req,
                prefix_len=prefix_len,
                scratch=scratch,
            )
        except Exception:
            self.backend.model.mem_manager.free(scratch)
            raise

        mapping[req.req_idx, :prefix_len].zero_()
        req.cur_kv_len = 0
        if self.backend.is_master_in_dp:
            req.shm_req.shm_cur_kv_len = 0
        return PrefixKVResource(
            prefix_len=prefix_len,
            local_prefix_mem_indexes=local_prefix,
            prefix_rank_major=prefix_rank_major,
            local_scratch_mem_indexes=scratch,
            scratch_rank_major=scratch_rank_major,
        )

    def release_context_resource(self, resource: PrefixKVResource) -> None:
        self.backend.model.mem_manager.free(resource.local_prefix_mem_indexes)
        self.backend.model.mem_manager.free(resource.local_scratch_mem_indexes)

    def release_remote_context(self, identity: PrefixContextIdentity) -> None:
        """Best-effort cleanup of worker-local mapping metadata.

        The registry invokes this only after the last action task proved that
        no worker can touch the target pages.  The full versioned identity
        makes a delayed release harmless after REPLACE.
        """

        if not self.backend.is_master_in_node:
            return
        try:
            self._socket.send_pyobj(
                ActionPrefixContextRelease(identity),
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        except Exception:
            # Target KV safety does not depend on retaining this small worker
            # cache.  A failed notification can only leak mapping metadata
            # until the action process exits.
            self.backend.logger.exception("failed to unregister action prefix context %s", identity)

    def is_completed(self, req_id: int) -> bool:
        with self._lock:
            return req_id in self._completed

    def completion(self, req_id: int) -> ActionOutcome | None:
        with self._lock:
            return self._completed.get(req_id)

    def completed_ids(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(self._completed)

    def forget(self, req_id: int) -> None:
        with self._lock:
            self._completed.pop(req_id, None)

    def publish_request_error(
        self,
        req: InferReq,
        exc: Exception,
        *,
        completion_callback: Callable[[ActionOutcome, bool], None] | None = None,
    ) -> None:
        """Complete a router-side context/control failure via normal output."""

        action_request = None
        request_config = self.config
        try:
            action_request, request_config = self._validate_request(req)
        except Exception:
            pass
        self._publish_local_failure(
            req,
            self._identity(req),
            exc,
            request_config=request_config,
            action_request=action_request,
            completion_callback=completion_callback,
        )

    def publish_control_success(self, req: InferReq, *, context_version: int) -> None:
        identity = self._identity(req)
        action_request, _ = self._validate_request(req)
        pending = _PendingAction(
            req_id=req.req_id,
            slot_index=req.shm_index,
            identity=identity,
            scratch_mem_indexes=torch.empty(0, dtype=torch.int32),
            deadline_at=time.perf_counter() + float(action_request.timeout),
            ack_grace_seconds=self._ack_grace_seconds(action_request),
            request_id=action_request.request_id,
            release_scratch=False,
        )
        self._pending[req.req_id] = pending
        if self.backend.is_master_in_node:
            self._ensure_task_slot(identity, horizon=1, action_dim=1)
            self._set_context_version(identity, context_version)
            if not self.output_store.write_control_success(identity):
                raise RuntimeError("failed to publish prefix-context control result")

    def _set_context_version(
        self,
        identity: ActionTaskIdentity,
        version: int,
    ) -> None:
        if not self.output_store.set_context_version(identity, version):
            raise RuntimeError("failed to attach prefix context version to action output")

    def submit(
        self,
        req: InferReq,
        *,
        prefix_len: int,
        prefix_rank_major: torch.Tensor | None = None,
        scratch_mem_indexes: torch.Tensor | None = None,
        scratch_rank_major: torch.Tensor | None = None,
        completion_callback: Callable[[ActionOutcome, bool], None] | None = None,
        context_version: int | None = None,
        prefix_context_identity: PrefixContextIdentity | None = None,
        context_owner_required: bool = False,
    ) -> bool:
        """Submit exactly once after the target-prefill commit event."""

        with self._lock:
            if req.req_id in self._pending:
                return False
            if req.req_id in self._completed:
                return False

            identity = self._identity(req)
            try:
                action_request, request_config = self._validate_request(req)
                ack_grace_seconds = self._ack_grace_seconds(action_request)
            except Exception as exc:
                self.backend.logger.exception("action request %s was rejected before dispatch", req.req_id)
                self._publish_local_failure(
                    req,
                    identity,
                    exc,
                    completion_callback=completion_callback,
                    context_owner_required=context_owner_required,
                )
                return True

            suffix_length = request_config.action_horizon + (0 if request_config.is_pi05 else 1)
            owns_scratch = scratch_mem_indexes is None
            scratch_capacity = scratch_mem_indexes
            if scratch_capacity is None:
                alloc_error = None
                try:
                    scratch_capacity = self.backend.model.mem_manager.alloc(suffix_length)
                except Exception as exc:
                    scratch_capacity = None
                    alloc_error = exc
                local_allocated = scratch_capacity is not None
                if not self._all_ranks_true(local_allocated):
                    if scratch_capacity is not None:
                        self.backend.model.mem_manager.free(scratch_capacity)
                    self._publish_local_failure(
                        req,
                        identity,
                        alloc_error or MemoryError("not enough target KV memory for the action suffix"),
                        request_config=request_config,
                        action_request=action_request,
                        completion_callback=completion_callback,
                        context_owner_required=context_owner_required,
                    )
                    return True
            assert scratch_capacity is not None
            if scratch_capacity.numel() < suffix_length:
                self._publish_local_failure(
                    req,
                    identity,
                    ValueError("context scratch is smaller than the requested suffix"),
                    request_config=request_config,
                    action_request=action_request,
                    completion_callback=completion_callback,
                    context_owner_required=context_owner_required,
                )
                return True
            active_scratch = scratch_capacity[:suffix_length]

            deadline_at = time.perf_counter() + float(action_request.timeout)
            pending = _PendingAction(
                req_id=req.req_id,
                slot_index=req.shm_index,
                identity=identity,
                scratch_mem_indexes=active_scratch,
                deadline_at=deadline_at,
                ack_grace_seconds=ack_grace_seconds,
                request_id=action_request.request_id,
                release_scratch=owns_scratch,
                completion_callback=completion_callback,
            )
            self._pending[req.req_id] = pending

            try:
                if self.backend.is_master_in_node and context_owner_required:
                    self._ensure_task_slot(
                        identity,
                        horizon=request_config.action_horizon,
                        action_dim=request_config.action_dim,
                    )
                    if not self.output_store.require_context_owner_ack(identity):
                        raise RuntimeError(
                            "failed to install persistent context owner handshake"
                        )
                has_prefix_mapping = prefix_rank_major is not None
                has_scratch_mapping = scratch_rank_major is not None
                if has_prefix_mapping != has_scratch_mapping:
                    raise ValueError("prefix and scratch rank-major mappings must be provided together")
                if not has_prefix_mapping and prefix_context_identity is None:
                    prefix_rank_major, scratch_rank_major = self._gather_rank_mappings(
                        req=req,
                        prefix_len=prefix_len,
                        scratch=active_scratch,
                    )
                elif has_prefix_mapping and prefix_context_identity is None:
                    # One-shot mappings only need the current suffix slice.
                    scratch_rank_major = scratch_rank_major[..., :suffix_length]
                elif has_prefix_mapping:
                    # Context registration carries the full scratch capacity,
                    # allowing later ticks to choose a different horizon.
                    if scratch_rank_major.shape[-1] < suffix_length:
                        raise ValueError("registered context scratch is shorter than the action suffix")
                if self.backend.is_master_in_node:
                    self._dispatch(
                        req=req,
                        identity=identity,
                        prefix_len=prefix_len,
                        prefix_rank_major=prefix_rank_major,
                        scratch_rank_major=scratch_rank_major,
                        action_request=action_request,
                        request_config=request_config,
                        deadline_at=deadline_at,
                        ack_grace_seconds=ack_grace_seconds,
                        context_version=context_version,
                        prefix_context_identity=prefix_context_identity,
                    )
            except Exception as exc:
                # Once the lease is installed, failures also flow through the
                # guarded store so every rank retires scratch in the same poll.
                self.backend.logger.exception("failed to dispatch action request %s", req.req_id)
                if self.backend.is_master_in_node:
                    self._publish_dispatch_failure(
                        identity=identity,
                        horizon=request_config.action_horizon,
                        action_dim=request_config.action_dim,
                        error_info=repr(exc),
                    )
            return True

    def abort(self, req: InferReq, reason: str = "inference request aborted") -> bool:
        with self._lock:
            pending = self._pending.get(req.req_id)
            if pending is None or pending.control_sent:
                return False
            return self._send_control(
                pending,
                kind=ActionControlKind.ABORT,
                reason=reason,
            )

    def poll(self) -> None:
        with self._lock:
            for req_id, pending in tuple(sorted(self._pending.items())):
                req = g_infer_context.requests_mapping.get(req_id)
                if (
                    req is not None
                    and (req.infer_aborted or bool(getattr(req.shm_req, "is_aborted", False)))
                    and not pending.control_sent
                ):
                    self._send_control(
                        pending,
                        kind=ActionControlKind.ABORT,
                        reason="target inference request was aborted",
                    )
                elif not pending.control_sent and time.perf_counter() >= pending.deadline_at:
                    self._send_control(
                        pending,
                        kind=ActionControlKind.TIMEOUT,
                        reason="action task deadline expired",
                    )

                if (
                    pending.control_sent
                    and pending.control_deadline_at is not None
                    and time.perf_counter() >= pending.control_deadline_at
                    and self.output_store.get_status(pending.slot_index) is ActionStatus.RUNNING
                    and self.backend.is_master_in_node
                ):
                    # PUSH success only proves local queuing.  If the manager
                    # never publishes a terminal outcome within the safe-ACK
                    # grace, retain every KV lease and require process restart.
                    self.output_store.mark_restart_required(
                        pending.identity,
                        "action manager did not publish a terminal outcome " "before the safe-ACK grace expired",
                    )

                decision = self.output_store.get_release_decision(
                    pending.slot_index,
                    identity=pending.identity,
                )
                if decision is ActionReleaseDecision.RESTART_REQUIRED:
                    # A worker may still access these pages.  Surface failure,
                    # but intentionally retain the lease until process restart.
                    # The release decision is authoritative even if manager
                    # output publication raced the local grace deadline.
                    self._observe_terminal(
                        req,
                        pending,
                        outcome_override=ActionOutcome.RESTART_REQUIRED,
                    )
                    self._notify_completion(pending, safe=False)
                    continue

                status = self.output_store.get_status(pending.slot_index)
                if status in {
                    ActionStatus.HAS_OUTPUT,
                    ActionStatus.ERROR,
                    ActionStatus.DONE,
                }:
                    self._observe_terminal(req, pending)
                if decision is ActionReleaseDecision.RELEASE and pending.terminal_observed:
                    self._retire_local_rank(req, pending)

    def _retire_local_rank(self, req: InferReq | None, pending: _PendingAction) -> None:
        if not pending.rank_consumed:
            matches = getattr(self.output_store, "matches", None)
            if matches is not None and not matches(pending.identity):
                return
            if pending.release_scratch and not pending.scratch_released and pending.scratch_mem_indexes.numel() != 0:
                self.backend.model.mem_manager.free(pending.scratch_mem_indexes)
                pending.scratch_released = True
            if not self.output_store.mark_rank_consumed(
                pending.slot_index,
                self.backend.rank_in_node,
                identity=pending.identity,
            ):
                return
            pending.rank_consumed = True

        if self.backend.is_master_in_node and self.output_store.all_ranks_consumed(pending.slot_index):
            status = self.output_store.get_status(pending.slot_index)
            if status is not ActionStatus.DONE:
                self.output_store.acknowledge(
                    pending.slot_index,
                    identity=pending.identity,
                )
        if self.output_store.get_status(pending.slot_index) is not ActionStatus.DONE:
            # Every target rank observes the same DONE publication before it
            # retires local lifecycle state.  This keeps subsequent TP
            # collectives ordered even when ranks poll at different speeds.
            return

        if not pending.done_observed:
            if not self.output_store.mark_rank_done_observed(
                pending.slot_index,
                self.backend.rank_in_node,
                identity=pending.identity,
            ):
                return
            pending.done_observed = True
        if not self.output_store.all_ranks_done_observed(pending.slot_index):
            return

        outcome = pending.outcome or ActionOutcome.ERROR
        self._notify_completion(pending, safe=True)
        if not pending.rank_retired:
            if not self.output_store.mark_rank_retired(
                pending.slot_index,
                self.backend.rank_in_node,
                identity=pending.identity,
            ):
                return
            pending.rank_retired = True
        if not self.output_store.all_ranks_retired(pending.slot_index):
            return

        self._completed[pending.req_id] = outcome
        self._pending.pop(pending.req_id, None)

    @staticmethod
    def _notify_completion(pending: _PendingAction, *, safe: bool) -> None:
        if pending.callback_notified:
            return
        pending.callback_notified = True
        if pending.completion_callback is not None:
            pending.completion_callback(
                pending.outcome or ActionOutcome.ERROR,
                safe,
            )

    def _observe_terminal(
        self,
        req: InferReq | None,
        pending: _PendingAction,
        *,
        outcome_override: ActionOutcome | None = None,
    ) -> None:
        if pending.terminal_observed:
            return
        try:
            response = self.output_store.read_terminal_response(
                pending.identity,
                pending.request_id,
            )
        except RuntimeError:
            return
        pending.terminal_observed = True
        outcome = response.outcome if outcome_override is None else outcome_override
        pending.outcome = outcome

    def _dispatch(
        self,
        *,
        req: InferReq,
        identity: ActionTaskIdentity,
        prefix_len: int,
        prefix_rank_major: torch.Tensor | None,
        scratch_rank_major: torch.Tensor | None,
        action_request: ActionRequest,
        request_config: Pi0VLAConfig,
        deadline_at: float,
        ack_grace_seconds: float,
        context_version: int | None,
        prefix_context_identity: PrefixContextIdentity | None,
    ) -> None:
        self._ensure_task_slot(
            identity,
            horizon=request_config.action_horizon,
            action_dim=request_config.action_dim,
        )
        if context_version is not None:
            self._set_context_version(identity, context_version)
        state = None if action_request.state is None else torch.as_tensor(action_request.state, dtype=torch.float32)
        if state is not None and state.ndim == 1:
            state = state.unsqueeze(0)
        noise = self._build_noise(req, action_request, request_config)
        task = ActionExpertTask(
            request_id=req.req_id,
            slot_index=req.shm_index,
            prefix_req_indexes=torch.tensor([req.req_idx], dtype=torch.int32),
            prefix_mem_indexes=prefix_rank_major,
            scratch_mem_indexes=scratch_rank_major,
            prefix_seq_lens=torch.tensor([prefix_len], dtype=torch.int32),
            action_req_indexes=torch.tensor([req.req_idx], dtype=torch.int32),
            state=None if request_config.is_pi05 else state,
            noisy_actions=noise,
            action_horizon=request_config.action_horizon,
            action_dim=request_config.action_dim,
            num_denoise_steps=request_config.num_denoise_steps,
            generation=identity.generation,
            task_id=identity.task_id,
            submitted_at=time.perf_counter(),
            deadline_at=deadline_at,
            ack_grace_seconds=ack_grace_seconds,
            prefix_context_identity=prefix_context_identity,
        )
        try:
            self._socket.send_pyobj(task, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            self.output_store.write_dispatch_error(identity, repr(exc))
            raise

    def _publish_local_failure(
        self,
        req: InferReq,
        identity: ActionTaskIdentity,
        exc: Exception,
        *,
        request_config: Pi0VLAConfig | None = None,
        action_request: ActionRequest | None = None,
        completion_callback: Callable[[ActionOutcome, bool], None] | None = None,
        context_owner_required: bool = False,
    ) -> None:
        request_config = request_config or self.config
        timeout = 120.0 if action_request is None else float(action_request.timeout)
        grace = 5.0 if action_request is None else self._ack_grace_seconds(action_request)
        pending = _PendingAction(
            req_id=req.req_id,
            slot_index=req.shm_index,
            identity=identity,
            scratch_mem_indexes=torch.empty(0, dtype=torch.int32, device="cpu"),
            deadline_at=time.perf_counter() + timeout,
            ack_grace_seconds=grace,
            request_id=None if action_request is None else action_request.request_id,
            completion_callback=completion_callback,
        )
        self._pending[req.req_id] = pending
        if self.backend.is_master_in_node:
            if context_owner_required:
                self._ensure_task_slot(
                    identity,
                    horizon=request_config.action_horizon,
                    action_dim=request_config.action_dim,
                )
                if not self.output_store.require_context_owner_ack(identity):
                    raise RuntimeError(
                        "failed to install persistent context owner handshake"
                    )
            self._publish_dispatch_failure(
                identity=identity,
                horizon=request_config.action_horizon,
                action_dim=request_config.action_dim,
                error_info=repr(exc),
            )

    def _publish_dispatch_failure(
        self,
        *,
        identity: ActionTaskIdentity,
        horizon: int,
        action_dim: int,
        error_info: str,
    ) -> None:
        if self.output_store.matches(identity):
            status = self.output_store.get_status(identity.slot_index)
            if status is ActionStatus.RUNNING:
                self.output_store.write_dispatch_error(identity, error_info)
                return
            if status in {
                ActionStatus.HAS_OUTPUT,
                ActionStatus.ERROR,
                ActionStatus.DONE,
            }:
                # The direct send path may already have published its guarded
                # dispatch error before propagating the local exception.
                return
        self._ensure_task_slot(
            identity,
            horizon=horizon,
            action_dim=action_dim,
        )
        self.output_store.write_dispatch_error(identity, error_info)

    def _ensure_task_slot(
        self,
        identity: ActionTaskIdentity,
        *,
        horizon: int,
        action_dim: int,
    ) -> None:
        if self.output_store.matches(identity):
            if self.output_store.get_status(identity.slot_index) is ActionStatus.RUNNING:
                return
            raise RuntimeError("matching action generation is not an active RUNNING task")
        if not self.output_store.begin_task(
            identity,
            horizon=horizon,
            action_dim=action_dim,
        ):
            raise RuntimeError("action output slot is not idle for the current request generation")

    def _send_control(
        self,
        pending: _PendingAction,
        *,
        kind: ActionControlKind,
        reason: str,
    ) -> bool:
        pending.control_sent = True
        pending.control_deadline_at = time.perf_counter() + pending.ack_grace_seconds
        if not self.backend.is_master_in_node:
            return True
        control = ActionControlRequest(
            slot_index=pending.slot_index,
            generation=pending.identity.generation,
            task_id=pending.identity.task_id,
            kind=kind,
            reason=reason,
            grace_seconds=pending.ack_grace_seconds,
        )
        try:
            self._socket.send_pyobj(control, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            self.output_store.mark_restart_required(
                pending.identity,
                f"failed to send action {kind.value} control: {exc!r}",
            )
            self.backend.logger.exception("failed to send %s for action task %s", kind.value, pending.identity)
        return True

    def _gather_rank_mappings(
        self,
        *,
        req: InferReq,
        prefix_len: int,
        scratch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mapping = self.backend.model.req_manager.req_to_token_indexs
        local_error = None
        prefix = None
        scratch_gpu = None
        try:
            if prefix_len <= 0 or prefix_len > mapping.shape[1]:
                raise ValueError("action prefix length is outside the target logical table")
            prefix = mapping[req.req_idx, :prefix_len].detach().to(dtype=torch.int32).clone()
            if bool(torch.any(prefix < 0)):
                raise RuntimeError("target prefix mapping was captured before KV commit")
            scratch_gpu = scratch.to(
                device=mapping.device,
                dtype=torch.int32,
                non_blocking=True,
            ).clone()
        except Exception as exc:
            local_error = exc
        if not self._all_ranks_true(local_error is None):
            raise RuntimeError(
                "at least one target rank could not capture committed action KV mappings"
            ) from local_error
        assert prefix is not None and scratch_gpu is not None
        prefix_all = [torch.empty_like(prefix) for _ in range(self.backend.node_world_size)]
        scratch_all = [torch.empty_like(scratch_gpu) for _ in range(self.backend.node_world_size)]
        dist.all_gather(prefix_all, prefix, group=self.backend.node_nccl_group)
        dist.all_gather(scratch_all, scratch_gpu, group=self.backend.node_nccl_group)
        if not self.backend.is_master_in_node:
            return (
                torch.empty((0, prefix_len), dtype=torch.int32),
                torch.empty((0, scratch.numel()), dtype=torch.int32),
            )
        return (
            torch.stack(prefix_all).detach().cpu().clone(),
            torch.stack(scratch_all).detach().cpu().clone(),
        )

    def _all_ranks_true(self, value: bool) -> bool:
        if self.backend.node_world_size == 1:
            return value
        flag = torch.tensor(
            [int(value)],
            dtype=torch.int32,
            device=self.backend.model.req_manager.req_to_token_indexs.device,
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=self.backend.node_nccl_group)
        return bool(flag.item())

    def _validate_request(self, req: InferReq) -> tuple[ActionRequest, Pi0VLAConfig]:
        action_dict = req.multimodal_params.get("action")
        if action_dict is None:
            raise ValueError("action output branch is missing action parameters")
        action_request = action_dict if isinstance(action_dict, ActionRequest) else ActionRequest.from_dict(action_dict)
        return action_request, action_request.validate(self.config)

    @staticmethod
    def _identity(req: InferReq) -> ActionTaskIdentity:
        # Normal-mode request ids are globally monotonic.  Using the id as the
        # generation makes slot reuse ABA-safe and deterministic on every TP
        # rank without another control-plane collective.
        generation = int(req.req_id)
        if generation < 0:
            raise ValueError("action request ids must be non-negative")
        return ActionTaskIdentity(req.shm_index, generation, generation)

    @classmethod
    def task_identity(cls, req: InferReq) -> ActionTaskIdentity:
        return cls._identity(req)

    @staticmethod
    def _build_noise(
        req: InferReq,
        action_request: ActionRequest,
        request_config: Pi0VLAConfig,
    ) -> torch.Tensor:
        if action_request.noise is None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(req.sampling_param.shm_param.seed))
            noise = torch.randn(
                1,
                request_config.action_horizon,
                request_config.action_dim,
                generator=generator,
                dtype=torch.float32,
            )
        else:
            noise = torch.as_tensor(action_request.noise, dtype=torch.float32)
            if noise.ndim == 2:
                noise = noise.unsqueeze(0)
        expected = (
            1,
            request_config.action_horizon,
            request_config.action_dim,
        )
        if tuple(noise.shape) != expected:
            raise ValueError(f"noise must have shape {expected}, got {tuple(noise.shape)}")
        return noise

    @staticmethod
    def _ack_grace_seconds(action_request: ActionRequest) -> float:
        raw_value = action_request.metadata.get("ack_grace_seconds", 5.0)
        value = float(raw_value)
        if not math.isfinite(value) or value < 0:
            raise ValueError("ack_grace_seconds must be finite and non-negative")
        return value


__all__ = ["ActionBranchRuntime", "PrefixKVResource"]
