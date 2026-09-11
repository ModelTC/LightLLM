"""Shared asynchronous transport for GenerationServer output branches."""

from __future__ import annotations

import asyncio
import os
import pickle
import uuid

import zmq
import zmq.asyncio

from lightllm.utils.log_utils import init_logger
from lightllm.utils.shm_port_args import get_shm_port_args


logger = init_logger(__name__)


class GenerationClient:
    result_type = object
    dispatch_error = RuntimeError
    label = "generation"

    def __init__(self, args):
        context = zmq.asyncio.Context.instance()
        self.socket = context.socket(zmq.DEALER)
        identity = f"generation-http-{os.getpid()}-{uuid.uuid4().hex}"
        self.socket.setsockopt(zmq.IDENTITY, identity.encode("ascii"))
        self.socket.setsockopt(zmq.LINGER, 0)
        generation_port = get_shm_port_args().generation_port
        self.socket.connect(f"{args.zmq_mode}127.0.0.1:{generation_port}")
        self._pending: dict[str, asyncio.Future] = {}
        self._send_lock = asyncio.Lock()
        self._receiver_task: asyncio.Task | None = None

    def _ensure_receiver(self) -> None:
        if self._receiver_task is not None and self._receiver_task.done():
            try:
                self._receiver_task.result()
            except BaseException:
                logger.exception("restarting failed %s receive loop", self.label)
            self._receiver_task = None
        if self._receiver_task is None:
            self._receiver_task = asyncio.create_task(self._receive_loop())

    async def dispatch(self, message, *, timeout: float) -> asyncio.Task:
        """Complete the local send before transferring ownership to the caller."""

        correlation_id = getattr(message, "correlation_id", None)
        if not isinstance(correlation_id, str) or not correlation_id:
            raise self.dispatch_error(f"{self.label} message requires a correlation_id")
        if correlation_id in self._pending:
            raise self.dispatch_error(
                f"duplicate {self.label} correlation_id: {correlation_id}"
            )

        self._ensure_receiver()
        future = asyncio.get_running_loop().create_future()
        self._pending[correlation_id] = future

        async def send_message():
            async with self._send_lock:
                await self.socket.send(
                    pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
                )

        send_task = asyncio.create_task(send_message())
        try:
            await asyncio.shield(send_task)
        except asyncio.CancelledError:
            try:
                await send_task
            except Exception as exc:
                self._pending.pop(correlation_id, None)
                future.cancel()
                raise self.dispatch_error(
                    f"{self.label} message was not sent to GenerationServer"
                ) from exc
            await self._best_effort_cancel(
                correlation_id,
                reason=f"HTTP {self.label} dispatch was cancelled",
            )
            self._pending.pop(correlation_id, None)
            future.cancel()
            raise
        except Exception as exc:
            self._pending.pop(correlation_id, None)
            future.cancel()
            raise self.dispatch_error(
                f"{self.label} message was not sent to GenerationServer"
            ) from exc

        return asyncio.create_task(
            self._wait_for_result(correlation_id, future, timeout=timeout)
        )

    async def submit(self, message, *, timeout: float):
        return await (await self.dispatch(message, timeout=timeout))

    async def cancel(self, correlation_id: str, *, reason: str) -> None:
        await self._best_effort_cancel(correlation_id, reason=reason)

    async def _wait_for_result(
        self,
        correlation_id: str,
        future: asyncio.Future,
        *,
        timeout: float,
    ):
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            await self._best_effort_cancel(
                correlation_id,
                reason=f"HTTP {self.label} wait timed out",
            )
            raise
        except asyncio.CancelledError:
            if future.done() and not future.cancelled():
                await self._release_undelivered_result(future.result())
            await self._best_effort_cancel(
                correlation_id,
                reason=f"HTTP {self.label} wait was cancelled",
            )
            raise
        finally:
            self._pending.pop(correlation_id, None)

    def _make_cancel(self, correlation_id: str, reason: str):
        raise NotImplementedError

    async def _release_undelivered_result(self, result) -> None:
        return None

    def _handle_stale_result(self, result) -> None:
        return None

    async def _best_effort_cancel(
        self,
        correlation_id: str,
        *,
        reason: str,
    ) -> None:
        if not correlation_id:
            return
        try:
            message = self._make_cancel(correlation_id, reason)
            async with self._send_lock:
                await self.socket.send(
                    pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
                )
        except BaseException:
            logger.exception("failed to cancel %s task %s", self.label, correlation_id)

    async def _receive_loop(self) -> None:
        try:
            while True:
                result = pickle.loads(await self.socket.recv())
                if not isinstance(result, self.result_type):
                    logger.error(
                        "discarding invalid %s response: %r",
                        self.label,
                        type(result),
                    )
                    continue
                future = self._pending.get(result.correlation_id)
                if future is None or future.done():
                    logger.warning(
                        "discarding stale %s reply correlation_id=%s",
                        self.label,
                        result.correlation_id,
                    )
                    self._handle_stale_result(result)
                    continue
                future.set_result(result)
        except asyncio.CancelledError:
            raise
        except BaseException:
            logger.exception("%s receive loop failed", self.label)
            for future in tuple(self._pending.values()):
                if not future.done():
                    future.set_exception(
                        RuntimeError(f"{self.label} receive loop failed")
                    )
            raise
