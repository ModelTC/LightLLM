"""Autonomous VLA server backed by ActivationTransfer.

VLA owns its queue, context reuse, cancellation and model lifecycle.  The
shared transfer layer owns only producer readiness and KV lifetime.
"""

from __future__ import annotations

import asyncio
import inspect
import pickle
import time
from dataclasses import dataclass, field

import setproctitle
import uvloop
import zmq
import zmq.asyncio

from lightllm.server.core.objs import ShmReqManager, StartArgs
from lightllm.server.activation_transfer import (
    ConsumerExecutionError as GenerationBackendError,
    KVActivation,
)
from lightllm.server.activation_transfer.channel import (
    ActivationChannel,
    ActivationLease,
)
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.utils.log_utils import init_logger
from lightllm.utils.process_check import start_parent_check_thread
from lightllm.utils.shm_port_args import get_shm_port_args

from .backend import (
    ActionBackend,
    ActionGenerationResult,
    ActionGenerationTask,
)
from .io import create_action_processor, load_action_config
from .model_rpc import start_model_process
from .protocol import (
    ActionCancel,
    ActionContextTask,
    ActionOutcome,
    ActionRequest,
    ActionResult,
    ActionTask,
    ActionTaskIdentity,
    PrefixContextIdentity,
    PrefixContextOp,
)

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
logger = init_logger(__name__)


@dataclass
class _ActionSubmission:
    task: ActionTask | ActionContextTask
    route: tuple[bytes, ...]
    request_config: dict
    commits_new_context: bool = False
    cancel_reason: str | None = None
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def correlation_id(self) -> str:
        return self.task.correlation_id

    @property
    def action_request(self) -> ActionRequest:
        return self.task.action_request

    @property
    def deadline_at(self) -> float:
        return self.task.submitted_at + self.action_request.timeout


@dataclass(frozen=True)
class _ActionContextMetadata:
    domain: str | None


@dataclass
class _VLAContext:
    context_id: str
    version: int
    lease: ActivationLease
    handle: KVActivation
    metadata: _ActionContextMetadata
    active_tasks: int = 0
    closing: bool = False
    released: bool = False
    poisoned: bool = False
    release_error: BaseException | None = None
    closed: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass(frozen=True)
class _ActionExecutionResult:
    generation: ActionGenerationResult | None
    context: PrefixContextIdentity | None
    committed_new_context: bool = False


class _VLAExecutionStopped(RuntimeError):
    def __init__(self, outcome: ActionOutcome, message: str) -> None:
        super().__init__(message)
        self.outcome = outcome


class VLAServer:
    """VLA-specific control plane with a storage-neutral Prefix KV input."""

    READY_QUEUE_CAPACITY = 64
    PREFIX_POLL_SECONDS = 0.001

    def __init__(self, args: StartArgs):
        self.args = args
        self.config = args.generation_runtime_config
        if self.config is None:
            self.config = load_action_config(args)
        self.prepost_processor = create_action_processor(self.config)

        self.tp_size = int(args.tp)
        if not 0 < self.tp_size < 64:
            raise ValueError("the ShmReq KV-ready mask supports 1 to 63 ranks")
        self.expected_kv_ready_mask = (1 << self.tp_size) - 1
        self.server_epoch = get_unique_server_name()
        self.model_rpcs = []
        self.shm_req_manager = ShmReqManager()
        self.transfer = ActivationChannel.for_shm_kv(
            self.shm_req_manager,
            expected_ready_mask=self.expected_kv_ready_mask,
        )

        self.capacity = max(
            1,
            min(self.READY_QUEUE_CAPACITY, int(args.running_max_req_size)),
        )
        self._concurrency = asyncio.Semaphore(1)
        self._submissions: dict[str, _ActionSubmission] = {}
        self._relay_tasks: set[asyncio.Task] = set()
        self._contexts: dict[str, _VLAContext] = {}
        self._draining_contexts: dict[tuple[str, int], _VLAContext] = {}
        self._context_versions: dict[str, int] = {}
        self.restart_required = False
        self._closed = False

        ports = get_shm_port_args()
        self.generation_nccl_port = ports.generation_nccl_port
        self.context = zmq.asyncio.Context(2)
        self.recv_socket = self.context.socket(zmq.ROUTER)
        self.recv_socket.setsockopt(zmq.LINGER, 0)
        self.recv_socket.bind(f"{args.zmq_mode}127.0.0.1:{ports.generation_port}")
        self._send_lock = asyncio.Lock()

        self.backend = ActionBackend(
            self.model_rpcs,
            self.prepost_processor,
            tp_size=self.tp_size,
        )

    async def wait_to_model_ready(self):
        init_tasks = []
        try:
            for _ in range(self.tp_size):
                self.model_rpcs.append(await start_model_process())
            for tp_rank_id, rpc in enumerate(self.model_rpcs):
                init_tasks.append(
                    asyncio.create_task(
                        rpc.init_model(
                            {
                                "args": self.args,
                                "device_id": tp_rank_id,
                                "generation_tp": self.tp_size,
                                "tp_rank_id": tp_rank_id,
                                "generation_nccl_port": self.generation_nccl_port,
                            }
                        )
                    )
                )
            await asyncio.gather(*init_tasks)
        except BaseException:
            for task in init_tasks:
                task.cancel()
            await asyncio.gather(*init_tasks, return_exceptions=True)
            await self._close_model_rpcs()
            raise

    async def run(self):
        try:
            await self.loop_for_requests()
        finally:
            await self.close()

    async def _close_model_rpcs(self):
        model_rpcs, self.model_rpcs[:] = list(self.model_rpcs), []
        close_results = await asyncio.gather(
            *(rpc.close() for rpc in model_rpcs),
            return_exceptions=True,
        )
        for result in close_results:
            if isinstance(result, BaseException):
                logger.error(
                    "failed to close VLA model worker",
                    exc_info=(type(result), result, result.__traceback__),
                )

    async def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            for submission in tuple(self._submissions.values()):
                submission.cancel_reason = "VLA server is closing"
                submission.cancel_event.set()
            await asyncio.gather(*self._relay_tasks, return_exceptions=True)
            await self._close_model_rpcs()
            # Unsafe leases are intentionally pinned until workers stop.
            self.transfer.release_all()
        finally:
            self.recv_socket.close(linger=0)
            self.context.term()

    async def loop_for_requests(self):
        while True:
            frames = await self.recv_socket.recv_multipart()
            if len(frames) < 2:
                logger.error("discarding malformed VLA ROUTER message")
                continue
            route = tuple(frames[:-1])
            message = None
            try:
                message = pickle.loads(frames[-1])
                if isinstance(message, ActionTask):
                    await self._accept_cold_task(message, route)
                elif isinstance(message, ActionContextTask):
                    await self._accept_context_task(message, route)
                elif isinstance(message, ActionCancel):
                    await self._accept_cancel(message)
                else:
                    raise TypeError(f"invalid VLA message: {type(message)!r}")
            except Exception as exc:
                logger.exception("failed to accept VLA message")
                if isinstance(message, (ActionTask, ActionContextTask)):
                    await self._reply_rejected(message, route, exc)

    async def _accept_cold_task(
        self,
        task: ActionTask,
        route: tuple[bytes, ...],
    ) -> None:
        request = task.action_request
        if task.activation in self.transfer:
            raise ValueError("duplicate VLA Prefix KV owner")
        lease = self.transfer.acquire(task.activation)
        try:
            if request.context_op in {PrefixContextOp.REUSE, PrefixContextOp.CLOSE}:
                raise ValueError("context reuse/close must use ActionContextTask")
            self._check_accepting(task.correlation_id)
            if request.context_op is PrefixContextOp.REPLACE:
                context = self._validate_context_identity(request)
                self._apply_context_domain(request, context)
            elif request.context_op is PrefixContextOp.CREATE:
                self._validate_new_context(request)
            request_config = request.validate(self.config)
            generation_task = self._generation_task(
                task.correlation_id,
                task.identity,
                request,
                request_config,
                self._cold_seed(request, task.default_seed),
            )
            submission = _ActionSubmission(
                task=task,
                route=route,
                request_config=request_config,
                commits_new_context=request.context_op
                in {PrefixContextOp.CREATE, PrefixContextOp.REPLACE},
            )
            lease.mark_in_use()
            self._remember_and_relay(
                submission,
                self._execute_cold(submission, generation_task, lease),
            )
        except BaseException:
            lease.release()
            raise

    async def _accept_context_task(
        self,
        task: ActionContextTask,
        route: tuple[bytes, ...],
    ) -> None:
        self._check_accepting(task.correlation_id)
        request = task.action_request
        if request.context_op is PrefixContextOp.CLOSE:
            context = self._find_context_for_close(request)
            if context is not None:
                self._apply_context_domain(request, context)
            request_config = request.validate(self.config)
            submission = _ActionSubmission(task, route, request_config)
            self._remember_and_relay(
                submission,
                self._execute_close(
                    submission,
                    context,
                    request.context_identity,
                ),
            )
            return
        if request.context_op is not PrefixContextOp.REUSE:
            raise ValueError("ActionContextTask only supports reuse or close")

        context = self._validate_context_identity(request)
        self._apply_context_domain(request, context)
        request_config = request.validate(self.config)
        submission = _ActionSubmission(task, route, request_config)
        context.active_tasks += 1
        try:
            identity = ActionTaskIdentity(
                context.handle.shm_index,
                context.handle.request_id,
                task.task_id,
            )
            generation_task = self._generation_task(
                task.correlation_id,
                identity,
                request,
                request_config,
                self._context_seed(request, task.task_id),
            )
            self._remember_and_relay(
                submission,
                self._execute_reuse(submission, generation_task, context),
            )
        except BaseException:
            context.active_tasks -= 1
            raise

    async def _accept_cancel(self, cancel: ActionCancel) -> None:
        submission = self._submissions.get(cancel.correlation_id)
        if submission is None:
            return
        submission.cancel_reason = cancel.reason
        submission.cancel_event.set()

    async def _execute_cold(
        self,
        submission: _ActionSubmission,
        task: ActionGenerationTask,
        lease: ActivationLease,
    ) -> _ActionExecutionResult:
        release_lease = True
        try:
            handle = await self._wait_for_prefix(submission, lease)
            generation = await self._run_backend(submission, task, handle)
            context_identity = None
            request = submission.action_request
            if request.context_op in {
                PrefixContextOp.CREATE,
                PrefixContextOp.REPLACE,
            }:
                context = await self._commit_context(request, lease, handle)
                context_identity = self._public_context(context)
                release_lease = False
            return _ActionExecutionResult(
                generation,
                context_identity,
                committed_new_context=context_identity is not None,
            )
        except GenerationBackendError as exc:
            if not exc.safe_to_release:
                release_lease = False
                self.restart_required = True
                raise _VLAExecutionStopped(
                    ActionOutcome.RESTART_REQUIRED,
                    str(exc),
                ) from exc
            raise
        finally:
            if release_lease:
                try:
                    lease.release()
                except Exception as exc:
                    self.restart_required = True
                    raise _VLAExecutionStopped(
                        ActionOutcome.RESTART_REQUIRED,
                        f"failed to release understanding lease: {exc!r}",
                    ) from exc

    async def _execute_reuse(
        self,
        submission: _ActionSubmission,
        task: ActionGenerationTask,
        context: _VLAContext,
    ) -> _ActionExecutionResult:
        try:
            generation = await self._run_backend(
                submission,
                task,
                context.handle,
            )
            return _ActionExecutionResult(
                generation,
                self._public_context(context),
            )
        except GenerationBackendError as exc:
            if not exc.safe_to_release:
                context.poisoned = True
                self.restart_required = True
                raise _VLAExecutionStopped(
                    ActionOutcome.RESTART_REQUIRED,
                    str(exc),
                ) from exc
            raise
        finally:
            context.active_tasks -= 1
            await self._release_context_if_drained(context)

    async def _execute_close(
        self,
        submission: _ActionSubmission,
        context: _VLAContext | None,
        identity: PrefixContextIdentity,
    ) -> _ActionExecutionResult:
        self._raise_if_stopped(submission)
        if context is None:
            return _ActionExecutionResult(None, identity)
        if not context.closing:
            context.closing = True
            self._contexts.pop(context.context_id, None)
            self._draining_contexts[(context.context_id, context.version)] = context
        await self._release_context_if_drained(context)
        if context.poisoned:
            raise _VLAExecutionStopped(
                ActionOutcome.RESTART_REQUIRED,
                "context is pinned until VLA workers restart",
            )
        await context.closed.wait()
        if context.release_error is not None:
            raise _VLAExecutionStopped(
                ActionOutcome.RESTART_REQUIRED,
                f"failed to release VLA context: {context.release_error!r}",
            )
        return _ActionExecutionResult(None, self._public_context(context))

    async def _wait_for_prefix(
        self,
        submission: _ActionSubmission,
        lease: ActivationLease,
    ) -> KVActivation:
        while True:
            self._raise_if_stopped(submission)
            handle = lease.resolve()
            if handle is not None:
                return handle
            try:
                await asyncio.wait_for(
                    submission.cancel_event.wait(),
                    timeout=min(
                        self.PREFIX_POLL_SECONDS,
                        max(0.0, submission.deadline_at - time.perf_counter()),
                    ),
                )
            except asyncio.TimeoutError:
                pass

    async def _run_backend(
        self,
        submission: _ActionSubmission,
        task: ActionGenerationTask,
        handle: KVActivation,
    ) -> ActionGenerationResult:
        async with self._concurrency:
            self._raise_if_stopped(submission)
            backend_task = asyncio.create_task(self.backend.generate(task, handle))
            cancel_wait = asyncio.create_task(submission.cancel_event.wait())
            done, _ = await asyncio.wait(
                (backend_task, cancel_wait),
                timeout=max(0.0, submission.deadline_at - time.perf_counter()),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if backend_task in done:
                cancel_wait.cancel()
                await asyncio.gather(cancel_wait, return_exceptions=True)
                return await backend_task

            # TP RPCs and collectives cannot be interrupted. Wait for every
            # rank's CUDA-safe ACK, then report the deadline/cancellation.
            cancel_wait.cancel()
            await asyncio.gather(cancel_wait, return_exceptions=True)
            await backend_task
            self._raise_if_stopped(submission)
            raise RuntimeError("unreachable VLA execution state")

    def _raise_if_stopped(self, submission: _ActionSubmission) -> None:
        if self.restart_required:
            raise _VLAExecutionStopped(
                ActionOutcome.RESTART_REQUIRED,
                "VLA worker restart is required",
            )
        if submission.cancel_reason is not None:
            raise _VLAExecutionStopped(
                ActionOutcome.ABORTED,
                submission.cancel_reason,
            )
        if time.perf_counter() >= submission.deadline_at:
            raise _VLAExecutionStopped(
                ActionOutcome.TIMEOUT,
                "VLA request deadline expired",
            )

    async def _commit_context(
        self,
        request: ActionRequest,
        lease: ActivationLease,
        handle: KVActivation,
    ) -> _VLAContext:
        assert request.context_id is not None
        metadata = _ActionContextMetadata(request.domain)
        if request.context_op is PrefixContextOp.CREATE:
            self._validate_new_context(request)
            version = self._context_versions.get(request.context_id, 0) + 1
        elif request.context_op is PrefixContextOp.REPLACE:
            old = self._validate_context_identity(request)
            version = old.version + 1
            old.closing = True
            self._contexts.pop(old.context_id, None)
            self._draining_contexts[(old.context_id, old.version)] = old
            await self._release_context_if_drained(old)
        else:
            raise ValueError("cold request does not create a VLA context")

        context = _VLAContext(
            request.context_id,
            version,
            lease,
            handle,
            metadata,
        )
        self._contexts[context.context_id] = context
        self._context_versions[context.context_id] = version
        return context

    async def _release_context_if_drained(self, context: _VLAContext) -> None:
        if (
            not context.closing
            or context.active_tasks
            or context.released
            or context.poisoned
            or context.release_error is not None
        ):
            return
        try:
            context.lease.release()
            context.released = True
            self._draining_contexts.pop(
                (context.context_id, context.version),
                None,
            )
        except BaseException as exc:
            context.release_error = exc
            self.restart_required = True
        finally:
            context.closed.set()

    def _validate_new_context(self, request: ActionRequest) -> None:
        if not request.context_id:
            raise ValueError("context creation requires context_id")
        if request.context_id in self._contexts or any(
            key[0] == request.context_id for key in self._draining_contexts
        ):
            raise RuntimeError(f"VLA context already exists: {request.context_id}")
        if len(self._contexts) + len(self._draining_contexts) >= self.capacity:
            raise RuntimeError("VLA context capacity exceeded")

    def _validate_context_identity(self, request: ActionRequest) -> _VLAContext:
        identity = request.context_identity
        if identity is None:
            raise ValueError("context operation requires an exact identity")
        if identity.server_epoch != self.server_epoch:
            raise RuntimeError(f"VLA context belongs to another server: {identity}")
        context = self._contexts.get(identity.context_id)
        if (
            context is None
            or context.version != identity.version
            or context.closing
        ):
            raise RuntimeError(f"unknown or stale VLA context {identity}")
        return context

    def _find_context_for_close(
        self,
        request: ActionRequest,
    ) -> _VLAContext | None:
        identity = request.context_identity
        if identity is None:
            raise ValueError("context close requires an exact identity")
        if identity.server_epoch != self.server_epoch:
            raise RuntimeError(f"VLA context belongs to another server: {identity}")
        context = self._contexts.get(identity.context_id)
        if context is None:
            context = self._draining_contexts.get(
                (identity.context_id, identity.version)
            )
        if context is not None:
            if context.version != identity.version:
                raise RuntimeError(f"unknown or stale VLA context {identity}")
            return context
        if self._context_versions.get(identity.context_id) == identity.version:
            return None
        raise RuntimeError(f"unknown or stale VLA context {identity}")

    @staticmethod
    def _apply_context_domain(
        request: ActionRequest,
        context: _VLAContext,
    ) -> None:
        domain = context.metadata.domain
        if request.domain is None:
            request.domain = domain
        elif request.domain != domain:
            raise ValueError(
                f"VLA context domain is {domain!r}, got {request.domain!r}"
            )

    def _public_context(self, context: _VLAContext) -> PrefixContextIdentity:
        return PrefixContextIdentity(
            context.context_id,
            context.version,
            self.server_epoch,
        )

    @staticmethod
    def _generation_task(
        correlation_id: str,
        identity: ActionTaskIdentity,
        request: ActionRequest,
        request_config: dict,
        seed: int,
    ) -> ActionGenerationTask:
        return ActionGenerationTask(
            task_id=correlation_id,
            action_horizon=request_config["action_horizon"],
            action_dimension=request_config["action_dim"],
            identity=identity,
            action_request=request,
            request_config=request_config,
            seed=seed,
        )

    def _remember_and_relay(self, submission: _ActionSubmission, execution) -> None:
        self._submissions[submission.correlation_id] = submission
        relay = asyncio.create_task(self._relay_result(submission, execution))
        self._relay_tasks.add(relay)
        relay.add_done_callback(self._relay_tasks.discard)

    async def _relay_result(self, submission: _ActionSubmission, execution) -> None:
        result = None
        try:
            execution_result = await execution
            generation = execution_result.generation
            if generation is None:
                result = ActionResult(
                    correlation_id=submission.correlation_id,
                    request_id=submission.action_request.request_id,
                    outcome=ActionOutcome.SUCCESS,
                    prefix_context=execution_result.context,
                )
            else:
                total_ms = (
                    time.perf_counter() - submission.task.submitted_at
                ) * 1000.0
                result = ActionResult(
                    correlation_id=submission.correlation_id,
                    request_id=submission.action_request.request_id,
                    outcome=ActionOutcome.SUCCESS,
                    actions=generation.actions,
                    action_horizon=submission.request_config["action_horizon"],
                    action_dim=submission.request_config["action_dim"],
                    policy_timing={
                        "queue_ms": max(0.0, total_ms - generation.backend_ms),
                        "action_expert_ms": generation.action_expert_ms,
                        "action_backend_ms": generation.backend_ms,
                        "action_overhead_ms": max(
                            0.0,
                            generation.backend_ms - generation.action_expert_ms,
                        ),
                        "total_ms": total_ms,
                    },
                    prefix_context=execution_result.context
                    or submission.action_request.context_identity,
                    committed_new_context=execution_result.committed_new_context,
                )
        except _VLAExecutionStopped as exc:
            result = self._error_result(submission, exc.outcome, str(exc))
        except GenerationBackendError as exc:
            result = self._error_result(submission, ActionOutcome.ERROR, str(exc))
        except Exception as exc:
            result = self._error_result(submission, ActionOutcome.ERROR, repr(exc))

        try:
            await self._send_result(submission.route, result)
        except Exception:
            logger.exception(
                "failed to send VLA result correlation_id=%s",
                submission.correlation_id,
            )
            if result.committed_new_context and result.prefix_context is not None:
                context = self._contexts.get(result.prefix_context.context_id)
                if context is not None and context.version == result.prefix_context.version:
                    try:
                        await self._close_context(context)
                    except Exception:
                        logger.exception("failed to close an undelivered VLA context")
        finally:
            self._submissions.pop(submission.correlation_id, None)

    async def _close_context(self, context: _VLAContext) -> None:
        if not context.closing:
            context.closing = True
            self._contexts.pop(context.context_id, None)
            self._draining_contexts[(context.context_id, context.version)] = context
        await self._release_context_if_drained(context)
        if context.poisoned:
            raise RuntimeError("VLA context is pinned until worker restart")
        await context.closed.wait()
        if context.release_error is not None:
            raise context.release_error

    def _error_result(
        self,
        submission: _ActionSubmission,
        outcome: ActionOutcome,
        error: str,
    ) -> ActionResult:
        return ActionResult(
            correlation_id=submission.correlation_id,
            request_id=submission.action_request.request_id,
            outcome=outcome,
            action_horizon=(
                submission.request_config["action_horizon"]
                if submission.action_request.produces_action
                else 0
            ),
            action_dim=(
                submission.request_config["action_dim"]
                if submission.action_request.produces_action
                else 0
            ),
            error_info=error,
            prefix_context=submission.action_request.context_identity,
            restart_required=outcome is ActionOutcome.RESTART_REQUIRED,
        )

    def _check_accepting(self, correlation_id: str) -> None:
        if self.restart_required:
            raise RuntimeError("VLA worker restart is required")
        if correlation_id in self._submissions:
            raise ValueError(f"duplicate VLA correlation_id: {correlation_id}")
        if len(self._submissions) >= self.capacity:
            raise RuntimeError("VLA server is overloaded")

    @staticmethod
    def _cold_seed(request: ActionRequest, default_seed: int) -> int:
        return int(default_seed if request.seed is None else request.seed)

    @staticmethod
    def _context_seed(request: ActionRequest, task_id: int) -> int:
        return int(task_id % (1 << 63) if request.seed is None else request.seed)

    async def _reply_rejected(
        self,
        task: ActionTask | ActionContextTask,
        route: tuple[bytes, ...],
        exc: Exception,
    ) -> None:
        await self._send_result(
            route,
            ActionResult(
                correlation_id=task.correlation_id,
                request_id=task.action_request.request_id,
                outcome=ActionOutcome.ERROR,
                error_info=repr(exc),
            ),
        )

    async def _send_result(
        self,
        route: tuple[bytes, ...],
        result: ActionResult,
    ) -> None:
        payload = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
        async with self._send_lock:
            await self.recv_socket.send_multipart([*route, payload])


# Keep the old public name as a compatibility alias; startup and documentation
# now describe this process by its capability, VLAServer.
ActionGenerationServer = VLAServer


def start_vla_process(args, pipe_writer):
    graceful_registry(inspect.currentframe().f_code.co_name)
    setproctitle.setproctitle(
        f"lightllm::{get_unique_server_name()}::vla_server"
    )
    start_parent_check_thread()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server = None
    try:
        try:
            server = VLAServer(args)
            loop.run_until_complete(server.wait_to_model_ready())
        except Exception as exc:
            logger.exception("VLA server startup failed")
            pipe_writer.send(str(exc))
            raise
        pipe_writer.send("init ok")
        loop.run_until_complete(server.run())
    finally:
        if server is not None:
            try:
                loop.run_until_complete(server.close())
            except Exception:
                logger.exception("failed to close VLA server resources")
        pipe_writer.close()
        loop.close()


start_action_generation_process = start_vla_process


__all__ = [
    "ActionGenerationServer",
    "VLAServer",
    "start_action_generation_process",
    "start_vla_process",
]
