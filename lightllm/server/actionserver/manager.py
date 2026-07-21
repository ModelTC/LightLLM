from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass, field

import setproctitle
import uvloop
import zmq
import zmq.asyncio

from lightllm.server.actionserver.lifecycle import (
    ActionTaskRegistry,
    TaskSubmissionDecision,
)
from lightllm.server.actionserver.objs import (
    ActionControlKind,
    ActionControlRequest,
    ActionExpertTask,
    ActionOutcome,
    ActionPrefixContextRelease,
    ActionStatus,
    ActionTaskIdentity,
    ActionWorkerAck,
    PrefixContextIdentity,
)
from lightllm.server.actionserver.shared_store import ActionOutputStore
from lightllm.server.core.objs import StartArgs
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.utils.log_utils import init_logger
from lightllm.utils.process_check import start_parent_check_thread

from .model_infer import start_model_process

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
logger = init_logger(__name__)


@dataclass
class _ManagedActionTask:
    task: ActionExpertTask
    control_event: asyncio.Event = field(default_factory=asyncio.Event)
    control: ActionControlRequest | None = None

    def request_control(self, control: ActionControlRequest) -> bool:
        if self.control is not None:
            return False
        self.control = control
        self.control_event.set()
        return True


class ActionManager:
    """Exactly-once dispatcher; policy computation stays in action ranks."""

    def __init__(self, args: StartArgs):
        self.args = args
        context = zmq.asyncio.Context(2)
        self.recv_socket = context.socket(zmq.PULL)
        self.recv_socket.bind(f"{args.zmq_mode}127.0.0.1:{args.action_port}")
        self.action_tp = args.action_tp
        self.output_store = ActionOutputStore.from_args(args)
        self.task_registry = ActionTaskRegistry()
        self._active: dict[ActionTaskIdentity, _ManagedActionTask] = {}
        self._executions: set[asyncio.Task] = set()
        self._maintenance: set[asyncio.Task] = set()
        self._registered_contexts: set[PrefixContextIdentity] = set()
        # Release is an ordering barrier, not merely cache cleanup.  Keep the
        # highest retired version per logical context so a delayed hot task can
        # never resolve indexes after the target has freed their physical KV.
        self._retired_context_versions: dict[tuple[str, str], int] = {}
        self._model_lock = asyncio.Lock()
        self._restart_required = False

    async def wait_to_model_ready(self):
        self.model_rpcs = [await start_model_process() for _ in range(self.action_tp)]
        init_tasks = []
        for tp_rank_id, rpc in enumerate(self.model_rpcs):
            init_tasks.append(
                rpc.init_model(
                    {
                        "args": self.args,
                        "device_id": self.args.action_gpu_ids[tp_rank_id],
                        "action_tp": self.action_tp,
                        "tp_rank_id": tp_rank_id,
                        "action_nccl_port": self.args.action_nccl_port,
                    }
                )
            )
        await asyncio.gather(*init_tasks)

    async def loop_for_requests(self):
        while True:
            message = await self.recv_socket.recv_pyobj()
            if isinstance(message, ActionExpertTask):
                self._accept_task(message)
            elif isinstance(message, ActionControlRequest):
                self._accept_control(message)
            elif isinstance(message, ActionPrefixContextRelease):
                self._accept_prefix_context_release(message)
            else:
                raise TypeError(f"invalid actionserver message: {type(message)!r}")

    def _accept_prefix_context_release(self, message: ActionPrefixContextRelease) -> None:
        identity = message.identity
        key = (identity.server_epoch, identity.context_id)
        self._retired_context_versions[key] = max(
            identity.version,
            self._retired_context_versions.get(key, 0),
        )
        self._registered_contexts.discard(identity)
        cleanup = asyncio.create_task(self._release_prefix_context(identity))
        self._maintenance.add(cleanup)
        cleanup.add_done_callback(self._maintenance_done)

    def _maintenance_done(self, future: asyncio.Task) -> None:
        self._maintenance.discard(future)
        try:
            future.result()
        except Exception:
            # Physical KV has already been retired safely by the target.  A
            # cleanup failure only retains small index tensors until restart.
            logger.exception("failed to release worker prefix-context metadata")

    async def _release_prefix_context(self, identity: PrefixContextIdentity) -> None:
        async with self._model_lock:
            # Losing a worker ACK makes worker execution state unknowable.  A
            # cleanup that was already queued behind the active task must not
            # issue another RPC after the fail-stop latch is set.
            if self._restart_required:
                logger.warning(
                    "skipping prefix-context metadata release after worker "
                    "quiescence was lost: %s",
                    identity,
                )
                return
            results = await asyncio.gather(
                *(rpc.release_prefix_context(identity) for rpc in self.model_rpcs),
                return_exceptions=True,
            )
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            raise RuntimeError(
                "one or more action ranks failed to release prefix metadata: "
                + "; ".join(repr(error) for error in failures)
            )

    def _accept_task(self, task: ActionExpertTask) -> asyncio.Task | None:
        identity = task.identity
        decision = self.task_registry.submit(identity)
        if decision is not TaskSubmissionDecision.ACCEPT:
            logger.warning(
                "ignoring %s action task slot=%s generation=%s task_id=%s",
                decision.value,
                identity.slot_index,
                identity.generation,
                identity.task_id,
            )
            return None
        if (
            not self.output_store.matches(identity)
            or self.output_store.get_status(identity.slot_index) is not ActionStatus.RUNNING
        ):
            logger.warning("ignoring action task whose output slot is no longer current: %s", identity)
            self.task_registry.mark_terminal(identity)
            return None

        context_identity = task.prefix_context_identity
        has_prefix_mapping = task.prefix_mem_indexes is not None
        has_scratch_mapping = task.scratch_mem_indexes is not None
        if has_prefix_mapping != has_scratch_mapping:
            self.output_store.write_dispatch_error(
                identity,
                "action task must carry both prefix and scratch mappings or neither",
            )
            self.task_registry.mark_terminal(identity)
            return None
        if context_identity is None and not has_prefix_mapping:
            self.output_store.write_dispatch_error(
                identity,
                "one-shot action task is missing physical KV mappings",
            )
            self.task_registry.mark_terminal(identity)
            return None
        if context_identity is not None:
            key = (
                context_identity.server_epoch,
                context_identity.context_id,
            )
            if context_identity.version <= self._retired_context_versions.get(key, 0):
                logger.warning(
                    "rejecting action task for retired prefix context %s",
                    context_identity,
                )
                self.output_store.write_dispatch_error(
                    identity,
                    f"prefix context {context_identity} is already retired",
                )
                self.task_registry.mark_terminal(identity)
                return None
            if not has_prefix_mapping and context_identity not in self._registered_contexts:
                self.output_store.write_dispatch_error(
                    identity,
                    f"prefix context {context_identity} is not registered",
                )
                self.task_registry.mark_terminal(identity)
                return None

        record = _ManagedActionTask(task=task)
        self._active[identity] = record
        execution = asyncio.create_task(self._execute(record))
        self._executions.add(execution)
        execution.add_done_callback(lambda future, task_identity=identity: self._execution_done(task_identity, future))
        return execution

    def _execution_done(self, identity: ActionTaskIdentity, future: asyncio.Task):
        self._active.pop(identity, None)
        self._executions.discard(future)
        self.task_registry.mark_terminal(identity)
        try:
            future.result()
        except Exception:
            logger.exception("unhandled action manager failure for %s", identity)
            if self.output_store.matches(identity):
                self.output_store.mark_restart_required(
                    identity,
                    "action manager failed before proving worker lease release",
                )
            self._restart_required = True

    def _accept_control(self, control: ActionControlRequest) -> bool:
        identity = control.identity
        record = self._active.get(identity)
        if record is not None:
            return record.request_control(control)

        # A control can race ahead of dispatch when multiple PUSH producers
        # are used.  No worker has seen the task, so the lease is immediately
        # safe; a later task message will fail the RUNNING-status check.
        if (
            self.output_store.matches(identity)
            and self.output_store.get_status(identity.slot_index) is ActionStatus.RUNNING
        ):
            self._publish_control(control)
            self.output_store.mark_all_workers_acked(identity)
            return True
        logger.info("ignoring control for inactive/stale action task %s", identity)
        return False

    async def _execute(self, record: _ManagedActionTask):
        identity = record.task.identity
        if self._restart_required:
            self.output_store.mark_restart_required(
                identity,
                "an action worker previously failed to ACK; process restart is required",
            )
            return

        acquired = await self._acquire_model_or_control(record)
        if not acquired:
            return
        try:
            # Multiple executions can pass the optimistic check above and then
            # queue on ``_model_lock``.  Recheck while owning the lock so a task
            # cannot enter workers after its predecessor lost an ACK.
            if self._restart_required:
                self.output_store.mark_restart_required(
                    identity,
                    "an action worker previously failed to ACK; process restart is required",
                )
                return
            control = self._effective_control(record)
            if control is not None:
                self._publish_control(control)
                self.output_store.mark_all_workers_acked(identity)
                return
            await self._dispatch_to_workers(record)
        except BaseException:
            # Set the fail-stop latch before releasing the serialization lock;
            # a done callback runs too late to stop the next queued task from
            # entering workers.
            self._restart_required = True
            raise
        finally:
            self._model_lock.release()

    async def _acquire_model_or_control(self, record: _ManagedActionTask) -> bool:
        acquire = asyncio.create_task(self._model_lock.acquire())
        control_wait = asyncio.create_task(record.control_event.wait())
        timeout = self._deadline_remaining(record.task)
        done, _ = await asyncio.wait(
            {acquire, control_wait},
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if acquire in done and not record.control_event.is_set():
            control_wait.cancel()
            return True

        if acquire.done() and acquire.result():
            self._model_lock.release()
        else:
            acquire.cancel()
        control_wait.cancel()
        control = self._effective_control(record)
        if control is None:
            control = self._deadline_control(record.task)
        self._publish_control(control)
        # The task never reached a worker, so every worker is safe by
        # construction and can be ACKed without an RPC round trip.
        self.output_store.mark_all_workers_acked(record.task.identity)
        return False

    async def _dispatch_to_workers(self, record: _ManagedActionTask):
        task = record.task
        identity = task.identity
        queue_ms = (time.perf_counter() - task.submitted_at) * 1000.0
        worker_calls = [asyncio.create_task(rpc.run_task(task)) for rpc in self.model_rpcs]
        worker_results = asyncio.gather(*worker_calls, return_exceptions=True)
        control_wait = asyncio.create_task(record.control_event.wait())
        timeout = self._deadline_remaining(task)
        done, _ = await asyncio.wait(
            {worker_results, control_wait},
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )

        if worker_results in done and not record.control_event.is_set():
            control_wait.cancel()
            self._publish_worker_results(
                task,
                worker_results.result(),
                queue_ms=queue_ms,
                publish_output=True,
            )
            return

        control_wait.cancel()
        control = self._effective_control(record)
        if control is None:
            control = self._deadline_control(task)
        self._publish_control(control)

        grace = max(0.0, float(control.grace_seconds))
        grace_done, _ = await asyncio.wait({worker_results}, timeout=grace)
        if worker_results in grace_done:
            self._publish_worker_results(
                task,
                worker_results.result(),
                queue_ms=queue_ms,
                publish_output=False,
            )
            return

        worker_results.add_done_callback(self._discard_late_worker_results)
        self.output_store.mark_restart_required(
            identity,
            f"action {control.kind.value} grace expired before every worker ACK; restart required",
        )
        self._restart_required = True

    @staticmethod
    def _discard_late_worker_results(future: asyncio.Future):
        try:
            future.result()
        except BaseException:
            pass

    def _publish_worker_results(
        self,
        task: ActionExpertTask,
        results: list,
        *,
        queue_ms: float,
        publish_output: bool,
    ):
        identity = task.identity
        valid_acks: dict[int, ActionWorkerAck] = {}
        unsafe_errors = []
        for result in results:
            if isinstance(result, BaseException):
                unsafe_errors.append(repr(result))
                continue
            if not isinstance(result, ActionWorkerAck):
                unsafe_errors.append(f"invalid worker ACK {type(result)!r}")
                continue
            if result.identity != identity:
                unsafe_errors.append(f"stale worker ACK for {result.identity}; expected {identity}")
                continue
            if result.rank in valid_acks:
                unsafe_errors.append(f"duplicate worker ACK from rank {result.rank}")
                continue
            valid_acks[result.rank] = result
            if result.safe_to_release:
                self.output_store.mark_worker_ack(identity, result.rank)
            else:
                unsafe_errors.append(result.error_info or f"worker rank {result.rank} did not release its KV lease")

        expected_ranks = set(range(self.action_tp))
        missing_ranks = expected_ranks.difference(valid_acks)
        if missing_ranks:
            unsafe_errors.append(f"missing worker ACKs from ranks {sorted(missing_ranks)}")
        if unsafe_errors or not self.output_store.all_workers_acked(identity.slot_index):
            self.output_store.mark_restart_required(
                identity,
                "; ".join(unsafe_errors) or "not every action worker released its KV lease",
            )
            self._restart_required = True
            return

        # Abort/timeout already owns the terminal outcome; worker outputs are
        # intentionally discarded after their safe-release ACKs arrive.
        if not publish_output:
            return

        worker_errors = [
            ack.error_info or f"action worker rank {ack.rank} failed"
            for ack in valid_acks.values()
            if ack.outcome is not ActionOutcome.SUCCESS
        ]
        if worker_errors:
            self.output_store.write_error(
                identity.slot_index,
                "; ".join(worker_errors),
                identity=identity,
                workers_safe=False,
            )
            logger.error("actionserver request %s failed: %s", task.request_id, worker_errors)
            return

        rank_zero = valid_acks.get(0)
        if rank_zero is None or rank_zero.actions is None:
            self.output_store.write_error(
                identity.slot_index,
                "action rank 0 returned no output",
                identity=identity,
                workers_safe=False,
            )
            return
        if task.prefix_context_identity is not None and task.prefix_mem_indexes is not None:
            # Every worker returned a successful, safe ACK, so the exact
            # version can now accept constant-size HOT_REUSE tasks.
            self._registered_contexts.add(task.prefix_context_identity)
        total_ms = (time.perf_counter() - task.submitted_at) * 1000.0
        logger.debug(
            "action timing request=%s queue=%.3fms expert=%.3fms " "model_rpc=%.3fms action_total=%.3fms",
            task.request_id,
            queue_ms,
            rank_zero.action_expert_ms,
            rank_zero.total_ms,
            total_ms,
        )
        self.output_store.write_output(
            identity.slot_index,
            rank_zero.actions,
            timing_ms=(queue_ms, rank_zero.action_expert_ms, total_ms),
            identity=identity,
        )

    def _publish_control(self, control: ActionControlRequest):
        reason = control.reason or f"action task {control.kind.value}"
        if control.kind is ActionControlKind.ABORT:
            self.output_store.write_abort(control.identity, reason)
        elif control.kind is ActionControlKind.TIMEOUT:
            self.output_store.write_timeout(control.identity, reason)
        else:
            raise ValueError(f"unsupported action control kind: {control.kind!r}")

    @staticmethod
    def _deadline_remaining(task: ActionExpertTask) -> float | None:
        if task.deadline_at is None:
            return None
        return max(0.0, task.deadline_at - time.perf_counter())

    @staticmethod
    def _deadline_control(task: ActionExpertTask) -> ActionControlRequest:
        return ActionControlRequest(
            slot_index=task.slot_index,
            generation=task.generation,
            task_id=task.task_id,
            kind=ActionControlKind.TIMEOUT,
            reason="action task deadline expired",
            grace_seconds=task.ack_grace_seconds,
        )

    def _effective_control(self, record: _ManagedActionTask) -> ActionControlRequest | None:
        if record.control is not None:
            return record.control
        remaining = self._deadline_remaining(record.task)
        if remaining is not None and remaining <= 0:
            return self._deadline_control(record.task)
        return None


def start_action_process(args, pipe_writer):
    graceful_registry(inspect.currentframe().f_code.co_name)
    setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::action_server")
    start_parent_check_thread()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        manager = ActionManager(args)
        loop.run_until_complete(manager.wait_to_model_ready())
    except Exception as exc:
        logger.exception("actionserver startup failed")
        pipe_writer.send(str(exc))
        raise
    pipe_writer.send("init ok")
    loop.run_until_complete(manager.loop_for_requests())
