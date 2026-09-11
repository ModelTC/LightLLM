"""Autonomous image/video generation server.

The server owns pairing, deadlines, cancellation and model scheduling.  Its
only shared component with VLA is ActivationTransfer's readiness/lease
boundary.
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

from lightllm.server.core.objs import (
    ShmReqManager,
)
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.utils.log_utils import init_logger
from lightllm.utils.process_check import start_parent_check_thread
from lightllm.utils.shm_port_args import get_shm_port_args

from .backend import (
    ImageBackend,
    ImageGenerationResult,
)
from lightllm.server.activation_transfer import (
    ConsumerExecutionError as GenerationBackendError,
)
from lightllm.server.activation_transfer.channel import (
    ActivationChannel,
    ActivationLease,
    ActivationLeaseState,
)
from .model_rpc import start_model_process
from lightllm.server.generationserver.image.protocol import (
    ImageCancel,
    ImageActivationTask,
    ImageGenerationInput,
    ImageOutcome,
    ImageResult,
)

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
logger = init_logger(__name__)


class _ImageExecutionStopped(RuntimeError):
    def __init__(self, outcome: ImageOutcome, message: str) -> None:
        super().__init__(message)
        self.outcome = outcome


@dataclass
class _PrefixRecord:
    task: ImageActivationTask
    reply_route: tuple[bytes, ...]
    lease: ActivationLease


@dataclass
class _GenerationGroup:
    generation: ImageGenerationInput
    prefixes: dict[str, _PrefixRecord] = field(default_factory=dict)
    dispatched: bool = False
    deadline_task: asyncio.Task | None = None
    cancel_reason: str | None = None
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def deadline_at(self) -> float:
        submitted_at = min(
            (record.task.submitted_at for record in self.prefixes.values()),
            default=time.perf_counter(),
        )
        return submitted_at + self.generation.timeout


class ImageGenerationServer:
    """Pair understanding prefixes and execute one image backend."""

    PREFIX_POLL_SECONDS = 0.001

    def __init__(self, args):
        self.args = args
        if int(args.tp) != 1:
            raise ValueError(
                "the current image generation worker requires target tp=1"
            )
        generation_port = get_shm_port_args().generation_port
        self.expected_ready_mask = 1
        self.model_rpcs = []
        self.shm_req_manager = ShmReqManager()
        self.transfer = ActivationChannel.for_shm_kv(
            self.shm_req_manager,
            expected_ready_mask=self.expected_ready_mask,
        )
        self.groups: dict[str, _GenerationGroup] = {}
        self.by_correlation: dict[str, _GenerationGroup] = {}

        self.context = zmq.asyncio.Context(2)
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(f"{args.zmq_mode}127.0.0.1:{generation_port}")
        self._send_lock = asyncio.Lock()
        self._relay_tasks: set[asyncio.Task] = set()
        self._closed = False

        self.capacity = max(1, int(args.generation_max_concurrency))
        self._concurrency = asyncio.Semaphore(self.capacity)
        self.backend = ImageBackend(self.model_rpcs)
        self.restart_required = False

    async def wait_to_model_ready(self):
        model_rpc = await start_model_process()
        self.model_rpcs.append(model_rpc)
        await model_rpc.init_model({"args": self.args})

    async def run(self):
        try:
            await self._request_loop()
        finally:
            await self.close()

    async def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            for group in tuple(self.groups.values()):
                group.cancel_reason = "generation server is closing"
                group.cancel_event.set()
                if not group.dispatched:
                    self._forget_group(group)
            self.transfer.release_idle()
            await asyncio.gather(*self._relay_tasks, return_exceptions=True)
            close_results = await asyncio.gather(
                *(rpc.close() for rpc in self.model_rpcs),
                return_exceptions=True,
            )
            for result in close_results:
                if isinstance(result, BaseException):
                    logger.error(
                        "failed to close generation model worker",
                        exc_info=(type(result), result, result.__traceback__),
                    )
            self.model_rpcs.clear()
            # Unsafe consumer-owned leases stay pinned until model workers have
            # stopped. They are safe to retire at this process boundary.
            self.transfer.release_all()
        finally:
            self.socket.close(linger=0)
            self.context.term()

    async def _request_loop(self):
        while True:
            frames = await self.socket.recv_multipart()
            if len(frames) < 2:
                logger.error("discarding malformed generation message")
                continue
            route = tuple(frames[:-1])
            message = None
            try:
                message = pickle.loads(frames[-1])
                if isinstance(message, ImageActivationTask):
                    await self._accept_prefix(message, route)
                elif isinstance(message, ImageCancel):
                    await self._accept_cancel(message)
                else:
                    raise TypeError(f"invalid generation message: {type(message)!r}")
            except Exception as exc:
                logger.exception("failed to accept generation message")
                if isinstance(message, ImageActivationTask):
                    await self._reject_prefix(message, route, exc)

    async def _accept_prefix(
        self,
        task: ImageActivationTask,
        route: tuple[bytes, ...],
    ) -> None:
        if task.activation in self.transfer:
            raise ValueError("duplicate generation prefix owner")
        lease = self.transfer.acquire(task.activation)
        accepted = False
        try:
            if task.correlation_id in self.by_correlation:
                raise ValueError("duplicate generation correlation_id")
            if self.restart_required:
                raise RuntimeError("generation worker restart is required")

            group = self.groups.get(task.generation.generation_id)
            if group is None:
                if len(self.groups) >= self.capacity:
                    raise RuntimeError("generation server is overloaded")
                group = _GenerationGroup(task.generation)
                self.groups[task.generation.generation_id] = group
            elif group.generation != task.generation:
                raise ValueError("generation prefixes disagree on parameters")
            if task.role in group.prefixes:
                raise ValueError(f"duplicate generation prefix role: {task.role}")

            record = _PrefixRecord(task, route, lease)
            group.prefixes[task.role] = record
            self.by_correlation[task.correlation_id] = group
            accepted = True
            if group.deadline_task is None:
                group.deadline_task = asyncio.create_task(self._expire_group(group))
            if set(group.prefixes) == set(group.generation.prefix_roles):
                await self._dispatch_group(group)
        finally:
            if not accepted:
                lease.release()

    async def _dispatch_group(self, group: _GenerationGroup) -> None:
        if group.dispatched:
            return
        remaining = group.deadline_at - time.perf_counter()
        if remaining <= 0:
            await self._finish_incomplete(
                group,
                ImageOutcome.TIMEOUT,
                "generation timed out while pairing prefixes",
            )
            return

        leases = [record.lease for record in group.prefixes.values()]
        for lease in leases:
            lease.mark_in_use()
        group.dispatched = True
        if group.deadline_task is not None:
            group.deadline_task.cancel()
        relay = asyncio.create_task(self._run_group(group))
        self._relay_tasks.add(relay)
        relay.add_done_callback(self._relay_tasks.discard)

    async def _run_group(
        self,
        group: _GenerationGroup,
    ) -> None:
        release_leases = True
        try:
            activations = await self._wait_for_activations(group)
            async with self._concurrency:
                self._raise_if_stopped(group)
                backend_task = asyncio.create_task(
                    self.backend.generate(group.generation, activations)
                )
                cancel_wait = asyncio.create_task(group.cancel_event.wait())
                remaining = max(0.0, group.deadline_at - time.perf_counter())
                done, _ = await asyncio.wait(
                    (backend_task, cancel_wait),
                    timeout=remaining,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if backend_task in done:
                    cancel_wait.cancel()
                    await asyncio.gather(cancel_wait, return_exceptions=True)
                    result = await backend_task
                else:
                    # RPyC/CUDA execution is not interruptible. Retire it to a
                    # CUDA-safe ACK before releasing producer-owned KV.
                    cancel_wait.cancel()
                    await asyncio.gather(cancel_wait, return_exceptions=True)
                    await backend_task
                    self._raise_if_stopped(group)
            if not isinstance(result, ImageGenerationResult):
                raise TypeError(f"invalid image backend result: {type(result)!r}")
            await self._reply_group(
                group,
                ImageOutcome.SUCCESS,
                artifacts=result.image,
                generation_ms=result.generation_ms,
            )
        except _ImageExecutionStopped as exc:
            await self._reply_group(group, exc.outcome, error=str(exc))
        except GenerationBackendError as exc:
            if not exc.safe_to_release:
                release_leases = False
                self.restart_required = True
                outcome = ImageOutcome.RESTART_REQUIRED
            else:
                outcome = ImageOutcome.ERROR
            await self._reply_group(group, outcome, error=str(exc))
        except Exception as exc:
            await self._reply_group(group, ImageOutcome.ERROR, error=repr(exc))
        finally:
            if release_leases:
                try:
                    for record in group.prefixes.values():
                        record.lease.release()
                except Exception:
                    self.restart_required = True
                    logger.exception("failed to release image understanding lease")
            self._forget_group(group)

    async def _wait_for_activations(self, group: _GenerationGroup):
        while True:
            self._raise_if_stopped(group)
            resolved = {
                role: record.lease.resolve()
                for role, record in group.prefixes.items()
            }
            if all(handle is not None for handle in resolved.values()):
                return resolved
            try:
                await asyncio.wait_for(
                    group.cancel_event.wait(),
                    timeout=min(
                        self.PREFIX_POLL_SECONDS,
                        max(0.0, group.deadline_at - time.perf_counter()),
                    ),
                )
            except asyncio.TimeoutError:
                pass

    def _raise_if_stopped(self, group: _GenerationGroup) -> None:
        if self.restart_required:
            raise _ImageExecutionStopped(
                ImageOutcome.RESTART_REQUIRED,
                "generation worker restart is required",
            )
        if group.cancel_reason is not None:
            raise _ImageExecutionStopped(ImageOutcome.ABORTED, group.cancel_reason)
        if time.perf_counter() >= group.deadline_at:
            raise _ImageExecutionStopped(
                ImageOutcome.TIMEOUT,
                "generation deadline expired",
            )

    async def _accept_cancel(self, cancel: ImageCancel) -> None:
        group = self.by_correlation.get(cancel.correlation_id)
        if group is None:
            return
        if group.dispatched:
            group.cancel_reason = cancel.reason
            group.cancel_event.set()
        else:
            await self._finish_incomplete(
                group,
                ImageOutcome.ABORTED,
                cancel.reason,
            )

    async def _expire_group(self, group: _GenerationGroup) -> None:
        try:
            await asyncio.sleep(max(0.0, group.deadline_at - time.perf_counter()))
            if not group.dispatched and group.generation.generation_id in self.groups:
                await self._finish_incomplete(
                    group,
                    ImageOutcome.TIMEOUT,
                    "generation timed out while pairing prefixes",
                )
        except asyncio.CancelledError:
            raise

    async def _finish_incomplete(
        self,
        group: _GenerationGroup,
        outcome: ImageOutcome,
        error: str,
    ) -> None:
        try:
            for record in tuple(group.prefixes.values()):
                if record.lease.state is ActivationLeaseState.HELD:
                    record.lease.release()
        except Exception as exc:
            outcome = ImageOutcome.RESTART_REQUIRED
            error = f"{error}; failed to release Prefix KV: {exc!r}"
        await self._reply_group(group, outcome, error=error)
        self._forget_group(group)

    async def _reply_group(
        self,
        group: _GenerationGroup,
        outcome: ImageOutcome,
        *,
        artifacts=None,
        generation_ms: float = 0.0,
        error: str | None = None,
    ) -> None:
        for record in tuple(group.prefixes.values()):
            result = ImageResult(
                correlation_id=record.task.correlation_id,
                generation_id=group.generation.generation_id,
                request_id=group.generation.request_id,
                outcome=outcome,
                artifacts=list(artifacts or ()),
                generation_ms=generation_ms,
                error_info=error,
                restart_required=outcome is ImageOutcome.RESTART_REQUIRED,
            )
            try:
                await self._send(record.reply_route, result)
            except Exception:
                logger.exception(
                    "failed to send generation result correlation_id=%s",
                    record.task.correlation_id,
                )

    async def _reject_prefix(self, task, route, error) -> None:
        group = self.groups.get(task.generation.generation_id)
        record = None if group is None else group.prefixes.get(task.role)
        replied_by_group = (
            record is not None and record.task.correlation_id == task.correlation_id
        )
        if group is not None and not group.dispatched:
            await self._finish_incomplete(group, ImageOutcome.ERROR, repr(error))
        if replied_by_group:
            return
        await self._send(
            route,
            ImageResult(
                correlation_id=task.correlation_id,
                generation_id=task.generation.generation_id,
                request_id=task.generation.request_id,
                outcome=ImageOutcome.ERROR,
                error_info=repr(error),
            ),
        )

    def _forget_group(self, group: _GenerationGroup) -> None:
        self.groups.pop(group.generation.generation_id, None)
        for record in group.prefixes.values():
            self.by_correlation.pop(record.task.correlation_id, None)
        if (
            group.deadline_task is not None
            and group.deadline_task is not asyncio.current_task()
        ):
            group.deadline_task.cancel()

    async def _send(self, route, result) -> None:
        payload = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
        async with self._send_lock:
            await self.socket.send_multipart([*route, payload])


def start_image_generation_process(args, pipe_writer):
    graceful_registry(inspect.currentframe().f_code.co_name)
    setproctitle.setproctitle(
        f"lightllm::{get_unique_server_name()}::generation_server"
    )
    start_parent_check_thread()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server = None
    try:
        try:
            server = ImageGenerationServer(args)
            loop.run_until_complete(server.wait_to_model_ready())
        except Exception as exc:
            logger.exception("generation server startup failed")
            pipe_writer.send(str(exc))
            raise
        pipe_writer.send("init ok")
        loop.run_until_complete(server.run())
    finally:
        if server is not None:
            try:
                loop.run_until_complete(server.close())
            except Exception:
                logger.exception("failed to close generation server resources")
        pipe_writer.close()
        loop.close()
