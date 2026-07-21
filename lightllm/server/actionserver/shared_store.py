from __future__ import annotations

import numpy as np
import torch

from lightllm.models.pi0.config import Pi0VLAConfig
from lightllm.server.core.objs.shm_array import ShmArray
from lightllm.utils.envs_utils import get_unique_server_name

from .objs import (
    ActionAckStatus,
    ActionContextOwnerDisposition,
    ActionOutcome,
    ActionReleaseDecision,
    ActionResponse,
    ActionStatus,
    ActionTaskIdentity,
)


class ActionOutputStore:
    """Named shared-memory result slots indexed by the standard ShmReq slot."""

    ERROR_BYTES = 2048

    def __init__(
        self,
        max_requests: int,
        max_horizon: int,
        max_action_dim: int,
        *,
        consumer_ranks: int = 1,
        worker_ranks: int | None = None,
        initialize: bool = False,
    ):
        self.max_requests = max_requests
        self.max_horizon = max_horizon
        self.max_action_dim = max_action_dim
        self.consumer_ranks = consumer_ranks
        self.worker_ranks = consumer_ranks if worker_ranks is None else worker_ranks
        if self.consumer_ranks <= 0 or self.worker_ranks <= 0:
            raise ValueError("action store rank counts must be positive")
        prefix = f"{get_unique_server_name()}_action_output"

        self._status = self._array(prefix + "_status", (max_requests,), np.int32)
        self._outcome = self._array(prefix + "_outcome", (max_requests,), np.int32)
        self._ack_status = self._array(prefix + "_ack_status", (max_requests,), np.int32)
        self._generation = self._array(prefix + "_generation", (max_requests,), np.int64)
        self._task_id = self._array(prefix + "_task_id", (max_requests,), np.int64)
        self._context_version = self._array(prefix + "_context_version", (max_requests,), np.int64)
        self._context_owner_disposition = self._array(
            prefix + "_context_owner_disposition",
            (max_requests,),
            np.int32,
        )
        self._has_error = self._array(prefix + "_has_error", (max_requests,), np.bool_)
        self._horizon = self._array(prefix + "_horizon", (max_requests,), np.int32)
        self._action_dim = self._array(prefix + "_dim", (max_requests,), np.int32)
        self._actions = self._array(
            prefix + "_actions",
            (max_requests, max_horizon, max_action_dim),
            np.float32,
        )
        self._timing = self._array(prefix + "_timing", (max_requests, 3), np.float64)
        self._error_length = self._array(prefix + "_error_len", (max_requests,), np.int32)
        self._errors = self._array(prefix + "_errors", (max_requests, self.ERROR_BYTES), np.uint8)
        # The HTTP process must not recycle the standard ShmReq slot while a
        # slower model-infer TP rank can still hold the corresponding KV
        # allocation.  Each rank owns one byte, so acknowledgements need no
        # cross-process read/modify/write operation.
        self._rank_consumed = self._array(
            prefix + "_rank_consumed",
            (max_requests, consumer_ranks),
            np.bool_,
        )
        self._rank_done_observed = self._array(
            prefix + "_rank_done_observed",
            (max_requests, consumer_ranks),
            np.bool_,
        )
        self._rank_retired = self._array(
            prefix + "_rank_retired",
            (max_requests, consumer_ranks),
            np.bool_,
        )
        self._worker_acked = self._array(
            prefix + "_worker_acked",
            (max_requests, self.worker_ranks),
            np.bool_,
        )
        self._context_owner_rank_acked = self._array(
            prefix + "_context_owner_rank_acked",
            (max_requests, consumer_ranks),
            np.bool_,
        )
        if initialize:
            self._status.arr.fill(int(ActionStatus.IDLE))
            self._outcome.arr.fill(int(ActionOutcome.NONE))
            self._ack_status.arr.fill(int(ActionAckStatus.NONE))
            self._generation.arr.fill(-1)
            self._task_id.arr.fill(-1)
            self._context_version.arr.fill(-1)
            self._context_owner_disposition.arr.fill(
                int(ActionContextOwnerDisposition.NONE)
            )
            self._has_error.arr.fill(False)
            self._horizon.arr.fill(0)
            self._action_dim.arr.fill(0)
            self._timing.arr.fill(0)
            self._error_length.arr.fill(0)
            self._rank_consumed.arr.fill(False)
            self._rank_done_observed.arr.fill(False)
            self._rank_retired.arr.fill(False)
            self._worker_acked.arr.fill(False)
            self._context_owner_rank_acked.arr.fill(False)

    @staticmethod
    def _array(name: str, shape: tuple[int, ...], dtype) -> ShmArray:
        value = ShmArray(name, shape, dtype)
        value.create_shm()
        return value

    @classmethod
    def from_args(cls, args, *, initialize: bool = False) -> "ActionOutputStore":
        config = Pi0VLAConfig.from_start_args(args)
        return cls(
            args.running_max_req_size,
            config.action_horizon,
            config.max_action_dim,
            consumer_ranks=args.action_tp,
            worker_ranks=args.action_tp,
            initialize=initialize,
        )

    def _clear_payload(self, slot_index: int):
        self._has_error.arr[slot_index] = False
        self._horizon.arr[slot_index] = 0
        self._action_dim.arr[slot_index] = 0
        self._timing.arr[slot_index].fill(0)
        self._error_length.arr[slot_index] = 0
        self._rank_consumed.arr[slot_index].fill(False)
        self._rank_done_observed.arr[slot_index].fill(False)
        self._rank_retired.arr[slot_index].fill(False)
        self._worker_acked.arr[slot_index].fill(False)
        self._context_version.arr[slot_index] = -1
        self._context_owner_disposition.arr[slot_index] = int(
            ActionContextOwnerDisposition.NONE
        )
        self._context_owner_rank_acked.arr[slot_index].fill(False)

    def reset(
        self,
        slot_index: int,
        *,
        identity: ActionTaskIdentity,
    ) -> bool:
        if not self.matches(identity):
            return False
        self._clear_payload(slot_index)
        self._outcome.arr[slot_index] = int(ActionOutcome.NONE)
        self._ack_status.arr[slot_index] = int(ActionAckStatus.NONE)
        self._status.arr[slot_index] = int(ActionStatus.IDLE)
        return True

    def begin_task(
        self,
        identity: ActionTaskIdentity,
        *,
        horizon: int,
        action_dim: int,
    ) -> bool:
        slot_index = identity.slot_index
        if not 0 <= slot_index < self.max_requests:
            raise IndexError("action output slot is out of range")
        if not 0 < horizon <= self.max_horizon:
            raise ValueError("action horizon exceeds the shared output store")
        if not 0 < action_dim <= self.max_action_dim:
            raise ValueError("action dimension exceeds the shared output store")

        current = self.get_identity(slot_index)
        if current is not None and identity.generation <= current.generation:
            return False
        if self.get_status(slot_index) is not ActionStatus.IDLE:
            return False

        self._clear_payload(slot_index)
        self._generation.arr[slot_index] = identity.generation
        self._task_id.arr[slot_index] = identity.task_id
        self._horizon.arr[slot_index] = horizon
        self._action_dim.arr[slot_index] = action_dim
        self._outcome.arr[slot_index] = int(ActionOutcome.NONE)
        self._ack_status.arr[slot_index] = int(ActionAckStatus.WAITING_FOR_WORKERS)
        self._status.arr[slot_index] = int(ActionStatus.RUNNING)
        return True

    def write_output(
        self,
        slot_index: int,
        actions: torch.Tensor,
        *,
        timing_ms: tuple[float, float, float],
        identity: ActionTaskIdentity,
    ) -> bool:
        if not self.matches(identity):
            return False
        if self.get_status(slot_index) is not ActionStatus.RUNNING:
            return False
        value = actions.detach().cpu().float()
        if value.ndim == 3:
            if value.shape[0] != 1:
                raise ValueError("one actionserver task must contain one observation")
            value = value[0]
        if value.ndim != 2:
            raise ValueError("actions must have shape [horizon, action_dim]")
        horizon, action_dim = value.shape
        if horizon > self.max_horizon or action_dim > self.max_action_dim:
            raise ValueError("action output exceeds the shared output store")
        self._actions.arr[slot_index, :horizon, :action_dim] = value.numpy()
        self._horizon.arr[slot_index] = horizon
        self._action_dim.arr[slot_index] = action_dim
        self._timing.arr[slot_index] = timing_ms
        self._has_error.arr[slot_index] = False
        self._outcome.arr[slot_index] = int(ActionOutcome.SUCCESS)
        self._status.arr[slot_index] = int(ActionStatus.HAS_OUTPUT)
        self._refresh_worker_ack_status(slot_index)
        return True

    def write_control_success(self, identity: ActionTaskIdentity) -> bool:
        """Publish a successful non-action context control operation."""

        slot_index = identity.slot_index
        if not self.matches(identity):
            return False
        if self.get_status(slot_index) is not ActionStatus.RUNNING:
            return False
        self._horizon.arr[slot_index] = 0
        self._action_dim.arr[slot_index] = 0
        self._timing.arr[slot_index].fill(0)
        self._has_error.arr[slot_index] = False
        self._outcome.arr[slot_index] = int(ActionOutcome.SUCCESS)
        self._worker_acked.arr[slot_index].fill(True)
        self._status.arr[slot_index] = int(ActionStatus.HAS_OUTPUT)
        self._refresh_worker_ack_status(slot_index)
        return True

    def set_context_version(
        self,
        identity: ActionTaskIdentity,
        version: int,
    ) -> bool:
        if not self.matches(identity):
            return False
        if version <= 0:
            raise ValueError("prefix context version must be positive")
        self._context_version.arr[identity.slot_index] = int(version)
        return True

    def require_context_owner_ack(self, identity: ActionTaskIdentity) -> bool:
        """Hold a CREATE/REPLACE result until HTTP and every target rank agree.

        The HTTP process decides whether the newly committed handle was
        delivered or discarded.  Every target rank then either retains or
        closes its local context before the reusable ShmReq slot can return to
        the allocator.
        """

        if not self.matches(identity):
            return False
        if self.get_status(identity.slot_index) is not ActionStatus.RUNNING:
            return False
        self._context_owner_disposition.arr[identity.slot_index] = int(
            ActionContextOwnerDisposition.PENDING
        )
        self._context_owner_rank_acked.arr[identity.slot_index].fill(False)
        return True

    def get_context_owner_disposition(
        self,
        identity: ActionTaskIdentity,
    ) -> ActionContextOwnerDisposition:
        if not self.matches(identity):
            return ActionContextOwnerDisposition.NONE
        return ActionContextOwnerDisposition(
            int(self._context_owner_disposition.arr[identity.slot_index])
        )

    def mark_context_owner_disposition(
        self,
        identity: ActionTaskIdentity,
        disposition: ActionContextOwnerDisposition,
    ) -> bool:
        if disposition not in {
            ActionContextOwnerDisposition.DELIVERED,
            ActionContextOwnerDisposition.DISCARDED,
        }:
            raise ValueError("context owner disposition must be delivered or discarded")
        if not self.matches(identity):
            return False
        current = self.get_context_owner_disposition(identity)
        if current is ActionContextOwnerDisposition.NONE:
            return False
        if current is disposition:
            return False
        if current is not ActionContextOwnerDisposition.PENDING:
            raise RuntimeError(
                f"context owner disposition already resolved as {current.name.lower()}"
            )
        self._context_owner_disposition.arr[identity.slot_index] = int(disposition)
        return True

    def mark_context_owner_rank_acked(
        self,
        identity: ActionTaskIdentity,
        rank: int,
    ) -> bool:
        if not self.matches(identity):
            return False
        if not 0 <= rank < self.consumer_ranks:
            raise IndexError(f"consumer rank {rank} is out of range")
        if (
            self.get_context_owner_disposition(identity)
            is ActionContextOwnerDisposition.NONE
        ):
            return False
        self._context_owner_rank_acked.arr[identity.slot_index, rank] = True
        return True

    def all_context_owner_ranks_acked(self, identity: ActionTaskIdentity) -> bool:
        return self.matches(identity) and bool(
            self._context_owner_rank_acked.arr[identity.slot_index].all()
        )

    def context_owner_release_ready(self, identity: ActionTaskIdentity) -> bool:
        disposition = self.get_context_owner_disposition(identity)
        return disposition in {
            ActionContextOwnerDisposition.DELIVERED,
            ActionContextOwnerDisposition.DISCARDED,
        } and self.all_context_owner_ranks_acked(identity)

    def write_error(
        self,
        slot_index: int,
        error_info: str,
        *,
        identity: ActionTaskIdentity,
        outcome: ActionOutcome = ActionOutcome.ERROR,
        workers_safe: bool,
    ) -> bool:
        if outcome not in {
            ActionOutcome.ERROR,
            ActionOutcome.TIMEOUT,
            ActionOutcome.ABORTED,
            ActionOutcome.RESTART_REQUIRED,
        }:
            raise ValueError("write_error requires an error-like action outcome")
        if not self.matches(identity):
            return False
        status = self.get_status(slot_index)
        if status not in {
            ActionStatus.RUNNING,
            ActionStatus.HAS_OUTPUT,
            ActionStatus.ERROR,
        }:
            return False
        encoded = error_info.encode("utf-8")[: self.ERROR_BYTES]
        if encoded:
            self._errors.arr[slot_index, : len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
        self._error_length.arr[slot_index] = len(encoded)
        self._has_error.arr[slot_index] = True
        self._outcome.arr[slot_index] = int(outcome)
        self._status.arr[slot_index] = int(ActionStatus.ERROR)
        if outcome is ActionOutcome.RESTART_REQUIRED:
            self._ack_status.arr[slot_index] = int(ActionAckStatus.RESTART_REQUIRED)
        elif workers_safe:
            self._worker_acked.arr[slot_index].fill(True)
            self._ack_status.arr[slot_index] = int(ActionAckStatus.WORKERS_SAFE)
        else:
            self._refresh_worker_ack_status(slot_index)
        return True

    def write_timeout(
        self,
        identity: ActionTaskIdentity,
        error_info: str = "action task timed out",
    ) -> bool:
        return self.write_error(
            identity.slot_index,
            error_info,
            identity=identity,
            outcome=ActionOutcome.TIMEOUT,
            workers_safe=False,
        )

    def write_dispatch_error(
        self,
        identity: ActionTaskIdentity,
        error_info: str,
    ) -> bool:
        """Publish a local send/setup failure before any worker saw the task."""

        return self.write_error(
            identity.slot_index,
            error_info,
            identity=identity,
            outcome=ActionOutcome.ERROR,
            workers_safe=True,
        )

    def write_abort(
        self,
        identity: ActionTaskIdentity,
        error_info: str = "action task aborted",
    ) -> bool:
        return self.write_error(
            identity.slot_index,
            error_info,
            identity=identity,
            outcome=ActionOutcome.ABORTED,
            workers_safe=False,
        )

    def mark_restart_required(
        self,
        identity: ActionTaskIdentity,
        error_info: str,
    ) -> bool:
        return self.write_error(
            identity.slot_index,
            error_info,
            identity=identity,
            outcome=ActionOutcome.RESTART_REQUIRED,
            workers_safe=False,
        )

    def acknowledge(
        self,
        slot_index: int,
        *,
        identity: ActionTaskIdentity,
    ) -> bool:
        if not self.matches(identity):
            return False
        if self.get_release_decision(slot_index, identity=identity) is ActionReleaseDecision.RESTART_REQUIRED:
            raise RuntimeError("action workers did not ACK; process restart is required")
        if not self.all_workers_acked(slot_index):
            raise RuntimeError("cannot acknowledge action output before all action workers ACK")
        if not self.all_ranks_consumed(slot_index):
            raise RuntimeError("cannot acknowledge action output before all TP ranks consumed it")
        self._ack_status.arr[slot_index] = int(ActionAckStatus.SAFE_TO_RELEASE)
        self._status.arr[slot_index] = int(ActionStatus.DONE)
        return True

    def mark_rank_consumed(
        self,
        slot_index: int,
        rank: int,
        *,
        identity: ActionTaskIdentity,
    ) -> bool:
        if not self.matches(identity):
            return False
        if not 0 <= rank < self.consumer_ranks:
            raise IndexError(f"consumer rank {rank} is out of range")
        self._rank_consumed.arr[slot_index, rank] = True
        return True

    def mark_worker_ack(
        self,
        identity: ActionTaskIdentity,
        rank: int,
    ) -> bool:
        if not self.matches(identity):
            return False
        if not 0 <= rank < self.worker_ranks:
            raise IndexError(f"action worker rank {rank} is out of range")
        self._worker_acked.arr[identity.slot_index, rank] = True
        self._refresh_worker_ack_status(identity.slot_index)
        return True

    def mark_rank_done_observed(
        self,
        slot_index: int,
        rank: int,
        *,
        identity: ActionTaskIdentity,
    ) -> bool:
        if not self.matches(identity):
            return False
        if self.get_status(slot_index) is not ActionStatus.DONE:
            return False
        if not 0 <= rank < self.consumer_ranks:
            raise IndexError(f"consumer rank {rank} is out of range")
        self._rank_done_observed.arr[slot_index, rank] = True
        return True

    def mark_rank_retired(
        self,
        slot_index: int,
        rank: int,
        *,
        identity: ActionTaskIdentity,
    ) -> bool:
        if not self.matches(identity):
            return False
        if not self.all_ranks_done_observed(slot_index):
            return False
        if not 0 <= rank < self.consumer_ranks:
            raise IndexError(f"consumer rank {rank} is out of range")
        self._rank_retired.arr[slot_index, rank] = True
        return True

    def mark_all_workers_acked(self, identity: ActionTaskIdentity) -> bool:
        if not self.matches(identity):
            return False
        self._worker_acked.arr[identity.slot_index].fill(True)
        self._refresh_worker_ack_status(identity.slot_index)
        return True

    def _refresh_worker_ack_status(self, slot_index: int):
        if self.get_ack_status(slot_index) is ActionAckStatus.RESTART_REQUIRED:
            return
        if self.all_workers_acked(slot_index):
            self._ack_status.arr[slot_index] = int(ActionAckStatus.WORKERS_SAFE)
        else:
            self._ack_status.arr[slot_index] = int(ActionAckStatus.WAITING_FOR_WORKERS)

    def all_ranks_consumed(self, slot_index: int) -> bool:
        return bool(self._rank_consumed.arr[slot_index].all())

    def all_ranks_done_observed(self, slot_index: int) -> bool:
        return bool(self._rank_done_observed.arr[slot_index].all())

    def all_ranks_retired(self, slot_index: int) -> bool:
        return bool(self._rank_retired.arr[slot_index].all())

    def all_workers_acked(self, slot_index: int) -> bool:
        return bool(self._worker_acked.arr[slot_index].all())

    def get_status(self, slot_index: int) -> ActionStatus:
        return ActionStatus(int(self._status.arr[slot_index]))

    def get_outcome(self, slot_index: int) -> ActionOutcome:
        return ActionOutcome(int(self._outcome.arr[slot_index]))

    def get_ack_status(self, slot_index: int) -> ActionAckStatus:
        return ActionAckStatus(int(self._ack_status.arr[slot_index]))

    def get_identity(self, slot_index: int) -> ActionTaskIdentity | None:
        generation = int(self._generation.arr[slot_index])
        task_id = int(self._task_id.arr[slot_index])
        if generation < 0 or task_id < 0:
            return None
        return ActionTaskIdentity(slot_index, generation, task_id)

    def matches(self, identity: ActionTaskIdentity) -> bool:
        return self.get_identity(identity.slot_index) == identity

    def get_release_decision(
        self,
        slot_index: int,
        *,
        identity: ActionTaskIdentity,
    ) -> ActionReleaseDecision:
        if not self.matches(identity):
            return ActionReleaseDecision.WAIT
        ack_status = self.get_ack_status(slot_index)
        if ack_status is ActionAckStatus.RESTART_REQUIRED:
            return ActionReleaseDecision.RESTART_REQUIRED
        if ack_status in {
            ActionAckStatus.WORKERS_SAFE,
            ActionAckStatus.SAFE_TO_RELEASE,
        }:
            return ActionReleaseDecision.RELEASE
        return ActionReleaseDecision.WAIT

    def release_slot(self, identity: ActionTaskIdentity) -> bool:
        """Recycle a slot exactly once after worker and consumer ACKs."""

        if not self.matches(identity):
            return False
        status = self.get_status(identity.slot_index)
        if status is ActionStatus.IDLE:
            return False
        if status is not ActionStatus.DONE:
            raise RuntimeError("action output cannot be recycled before final ACK")
        if not self.all_ranks_retired(identity.slot_index):
            raise RuntimeError("action output cannot be recycled before every target rank retires")
        disposition = self.get_context_owner_disposition(identity)
        if (
            disposition is not ActionContextOwnerDisposition.NONE
            and not self.context_owner_release_ready(identity)
        ):
            raise RuntimeError(
                "persistent context output cannot be recycled before owner disposition ACK"
            )
        return self.reset(identity.slot_index, identity=identity)

    def read_response(
        self,
        slot_index: int,
        request_id: str | None = None,
        *,
        identity: ActionTaskIdentity,
    ) -> ActionResponse:
        if not self.matches(identity):
            raise RuntimeError("action output belongs to another task generation")
        if self.get_status(slot_index) is not ActionStatus.DONE:
            raise RuntimeError("action output has not been acknowledged by model-infer")
        return self._read_response_payload(slot_index, request_id)

    def read_terminal_response(
        self,
        identity: ActionTaskIdentity,
        request_id: str | None = None,
    ) -> ActionResponse:
        """Read a published outcome without implying that KV can be reused.

        ActionBranchRuntime can surface timeout/abort/restart-required output
        immediately, then independently wait on ``get_release_decision``.
        """

        if not self.matches(identity):
            raise RuntimeError("action output belongs to another task generation")
        if self.get_status(identity.slot_index) not in {
            ActionStatus.HAS_OUTPUT,
            ActionStatus.ERROR,
            ActionStatus.DONE,
        }:
            raise RuntimeError("action task has not published a terminal outcome")
        return self._read_response_payload(identity.slot_index, request_id)

    def _read_response_payload(
        self,
        slot_index: int,
        request_id: str | None,
    ) -> ActionResponse:
        horizon = int(self._horizon.arr[slot_index])
        action_dim = int(self._action_dim.arr[slot_index])
        timing = self._timing.arr[slot_index].tolist()
        has_error = bool(self._has_error.arr[slot_index])
        outcome = self.get_outcome(slot_index)
        context_version = int(self._context_version.arr[slot_index])
        error_info = None
        actions = None
        if has_error:
            length = int(self._error_length.arr[slot_index])
            error_info = bytes(self._errors.arr[slot_index, :length]).decode("utf-8", errors="replace")
        else:
            actions = torch.from_numpy(self._actions.arr[slot_index, :horizon, :action_dim].copy())
        return ActionResponse(
            request_id=request_id,
            actions=actions,
            action_horizon=horizon,
            action_dim=action_dim,
            policy_timing={
                "queue_ms": timing[0],
                "action_expert_ms": timing[1],
                "total_ms": timing[2],
            },
            error_info=error_info,
            outcome=outcome,
            restart_required=outcome is ActionOutcome.RESTART_REQUIRED,
            prefix_context_version=(None if context_version < 0 else context_version),
        )
