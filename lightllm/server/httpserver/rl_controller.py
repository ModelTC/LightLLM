import asyncio
import socket
import time
from contextlib import asynccontextmanager
from typing import Tuple

import rpyc
from rpyc.utils.classic import obtain

from lightllm.server.io_struct import (
    AbortReq,
    FlushCacheReq,
    GeneralHttpToModelRpcReq,
    GeneralModelToHttpRpcRsp,
    InitWeightsUpdateGroupReq,
    DestroyWeightsUpdateGroupReq,
    ReleaseMemoryReq,
    ResumeMemoryReq,
    UpdateWeightsFromDistributedReq,
    UpdateWeightsFromIPCReq,
    UpdateWeightsFromTensorReq,
)
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class _GenerationPauseGate:
    """Generation pause gate.

    Requests are tracked from admission until req_id_to_out_inf takes over.
    """

    _RUNNING = 0
    _PAUSED = 1

    def __init__(self) -> None:
        self._state = self._RUNNING
        self._pending_request_abort_events = {}
        self._lock = asyncio.Lock()
        self._resume_event = asyncio.Event()
        self._resume_event.set()

    @asynccontextmanager
    async def pause_and_abort_context(self):
        """Enter pause mode once and tell the caller whether it owns the abort pass."""
        async with self._lock:
            if self._state != self._RUNNING:
                do_abort = False
            else:
                # New requests after this point should wait at the gate until resume.
                # abort_all below is responsible only for requests already present
                # when that abort_all call takes its snapshot.
                self._state = self._PAUSED
                self._resume_event.clear()
                do_abort = True
        yield do_abort

    async def unregister_pending_request(self, request_id: int):
        async with self._lock:
            self._pending_request_abort_events.pop(request_id, None)

    async def wait_until_resumed_or_aborted(self, request_id: int) -> bool:
        """Enter admission and return True if this request should abort."""
        async with self._lock:
            abort_event = asyncio.Event()
            self._pending_request_abort_events[request_id] = abort_event
            if self._state == self._RUNNING:
                return False

            resume_event = self._resume_event

        resume_task = asyncio.create_task(resume_event.wait())
        abort_task = asyncio.create_task(abort_event.wait())
        try:
            done, pending = await asyncio.wait({resume_task, abort_task}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            resume_task.cancel()
            abort_task.cancel()
            await asyncio.gather(resume_task, abort_task, return_exceptions=True)
            await self.unregister_pending_request(request_id)
            raise

        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if abort_task in done and abort_event.is_set():
            await self.unregister_pending_request(request_id)
            return True
        return False

    async def abort_pending_requests(self, request_ids) -> None:
        async with self._lock:
            for request_id in request_ids:
                abort_event = self._pending_request_abort_events.get(request_id)
                if abort_event is not None:
                    abort_event.set()

    async def snapshot_and_abort_pending_requests(self):
        async with self._lock:
            request_ids = list(self._pending_request_abort_events.keys())
            for request_id in request_ids:
                self._pending_request_abort_events[request_id].set()
            return request_ids

    def has_pending_request(self, request_id: int) -> bool:
        return request_id in self._pending_request_abort_events

    async def resume(self) -> None:
        async with self._lock:
            self._state = self._RUNNING
            self._resume_event.set()


class HttpRlController:
    def __init__(self, manager) -> None:
        self.manager = manager
        self.args = manager.args
        self._generation_gate = _GenerationPauseGate()

    async def wait_if_generation_paused(self, request_id: int) -> bool:
        """Enter generation admission and return True if this request should abort."""
        return await self._generation_gate.wait_until_resumed_or_aborted(request_id)

    async def unregister_generation_admission(self, request_id: int) -> None:
        await self._generation_gate.unregister_pending_request(request_id)

    async def _wait_for_abort_released(
        self,
        request_ids,
        abort_when_running_request_ids=(),
        timeout: float = 60.0,
    ) -> Tuple[bool, str]:
        """Wait until aborted work has left both the pause gate and running map.

        request_ids is the fixed set of requests selected by abort_request().
        Requests entering the pause gate later are not part of this wait.
        """
        start_time = time.time()
        request_ids = set(request_ids)
        abort_when_running_request_ids = set(abort_when_running_request_ids)
        while True:
            has_unreleased_req = False
            for group_req_id in request_ids:
                if self._generation_gate.has_pending_request(group_req_id):
                    has_unreleased_req = True
                    break

                if group_req_id in self.manager.req_id_to_out_inf:
                    if group_req_id in abort_when_running_request_ids:
                        await self.manager.abort(group_req_id)
                        abort_when_running_request_ids.remove(group_req_id)
                    has_unreleased_req = True
                    break

            if not has_unreleased_req:
                return True, ""

            if time.time() - start_time > timeout:
                error_msg = (
                    f"abort request wait release timeout, request_ids_count={len(request_ids)}, timeout={timeout}s"
                )
                logger.error(error_msg)
                return False, error_msg

            await asyncio.sleep(0.02)
        return True, ""

    async def abort_request(self, request: AbortReq) -> Tuple[bool, str]:
        """Abort one request, or snapshot and abort all requests present now."""
        request_id = request.request_id
        if request.abort_all:
            # Snapshot before issuing aborts: this abort_all clears current
            # pending/running work, while future paused requests keep waiting.
            pending_request_ids = set(await self._generation_gate.snapshot_and_abort_pending_requests())
            running_request_ids = set(self.manager.req_id_to_out_inf.keys())
            abort_request_ids = pending_request_ids | running_request_ids
            for group_req_id in running_request_ids:
                await self.manager.abort(group_req_id)
            return await self._wait_for_abort_released(
                request_ids=abort_request_ids, abort_when_running_request_ids=pending_request_ids
            )

        if request_id is None:
            return True, ""

        await self._generation_gate.abort_pending_requests((request_id,))
        await self.manager.abort(request_id)
        return await self._wait_for_abort_released(request_ids=(request_id,))

    async def pause_generation(self):
        """Pause future generations and drain only the requests already present."""
        async with self._generation_gate.pause_and_abort_context() as do_abort:
            if not do_abort:
                return
            while True:
                success, msg = await self.abort_request(AbortReq(request_id=None, abort_all=True))
                if success:
                    break
                logger.warning(f"pause_generation abort_all still waiting: {msg}")
                await asyncio.sleep(1.0)

    async def continue_generation(self):
        """Release requests waiting at the pause gate."""
        await self._generation_gate.resume()

    def _call_router_rl_sync(self, req: GeneralHttpToModelRpcReq) -> GeneralModelToHttpRpcRsp:
        from lightllm.utils.retry_utils import retry

        conn = retry(max_attempts=20, wait_time=0.5)(rpyc.connect)(
            "localhost",
            self.args.rl_rpyc_port,
            config={"allow_pickle": True, "sync_request_timeout": 600},
        )
        try:
            conn._channel.stream.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return obtain(conn.root.rl_op(req))
        finally:
            try:
                conn.close()
            except BaseException:
                pass

    async def _call_router_rl(self, func_name: str, func_args=None) -> GeneralModelToHttpRpcRsp:
        req = GeneralHttpToModelRpcReq(func_name=func_name, func_args=func_args)
        try:
            return await asyncio.to_thread(self._call_router_rl_sync, req)
        except BaseException as e:
            logger.exception(f"rl rpyc call {func_name} failed: {e}")
            return GeneralModelToHttpRpcRsp(
                success=False,
                msg=f"rl rpyc call {func_name} error: {e}",
                func_name=func_name,
            )

    async def flush_cache(self, request: FlushCacheReq):
        return await self._call_router_rl("flush_cache", request)

    async def release_memory_occupation(self, request: ReleaseMemoryReq):
        assert (
            len(self.manager.req_id_to_out_inf) == 0
        ), "there are still requests running, cannot release memory occupation"
        return await self._call_router_rl("release_memory_occupation", request.tags)

    async def resume_memory_occupation(self, request: ResumeMemoryReq):
        return await self._call_router_rl("resume_memory_occupation", request.tags)

    async def init_weights_update_group(self, request: InitWeightsUpdateGroupReq):
        return await self._call_router_rl("init_weights_update_group", request)

    async def destroy_weights_update_group(self, request: DestroyWeightsUpdateGroupReq):
        return await self._call_router_rl("destroy_weights_update_group", request)

    async def update_weights_from_distributed(self, request: UpdateWeightsFromDistributedReq):
        if request.abort_all_requests:
            success, msg = await self.abort_request(AbortReq(abort_all=True))
            if not success:
                return GeneralModelToHttpRpcRsp(success=False, msg=msg, func_name="update_weights_from_distributed")
        if request.flush_cache:
            ret = await self.flush_cache(FlushCacheReq())
            if not ret.success:
                return ret
        return await self._call_router_rl("update_weights_from_distributed", request)

    async def update_weights_from_tensor(self, request: UpdateWeightsFromTensorReq) -> GeneralModelToHttpRpcRsp:
        if request.abort_all_requests:
            success, msg = await self.abort_request(AbortReq(abort_all=True))
            if not success:
                return GeneralModelToHttpRpcRsp(success=False, msg=msg, func_name="update_weights_from_tensor")
        if request.flush_cache:
            ret = await self.flush_cache(FlushCacheReq())
            if not ret.success:
                return ret
        return await self._call_router_rl("update_weights_from_tensor", request)

    async def update_weights_from_ipc(self, request: UpdateWeightsFromIPCReq) -> GeneralModelToHttpRpcRsp:
        if request.abort_all_requests:
            success, msg = await self.abort_request(AbortReq(abort_all=True))
            if not success:
                return GeneralModelToHttpRpcRsp(success=False, msg=msg, func_name="update_weights_from_ipc")
        if request.flush_cache:
            ret = await self.flush_cache(FlushCacheReq())
            if not ret.success:
                return ret
        return await self._call_router_rl("update_weights_from_ipc", request)
