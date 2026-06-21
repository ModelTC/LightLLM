import asyncio
import socket
import time
from contextlib import asynccontextmanager
from typing import Optional, Tuple

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

    Pending requests are tracked only while they are waiting at the pause gate.
    Once generation is allowed, the request leaves this lightweight path and
    continues as a normal request managed by req_id_to_out_inf.
    """

    _RUNNING = 0
    _PAUSED = 1

    def __init__(self) -> None:
        self._state = self._RUNNING
        self._pending_request_abort_events = {}
        self._cond = asyncio.Condition()

    @asynccontextmanager
    async def pause_and_abort_context(self):
        async with self._cond:
            if self._state != self._RUNNING:
                do_abort = False
            else:
                self._state = self._PAUSED
                do_abort = True
        yield do_abort

    async def register_pending_request(self, request_id: int):
        async with self._cond:
            self._pending_request_abort_events[request_id] = asyncio.Event()

    async def unregister_pending_request(self, request_id: int):
        async with self._cond:
            self._pending_request_abort_events.pop(request_id, None)

    async def wait_until_running_or_aborted(self, request_id: int) -> bool:
        """Returns True when the pending request should finish as aborted."""
        async with self._cond:
            abort_event = self._pending_request_abort_events.get(request_id)
            if abort_event is None:
                return False
            if abort_event.is_set():
                return True
            if self._state == self._RUNNING:
                return False

        resume_task = asyncio.create_task(self._wait_until_running())
        abort_task = asyncio.create_task(abort_event.wait())
        done, pending = await asyncio.wait({resume_task, abort_task}, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        return abort_task in done and abort_event.is_set()

    async def abort_pending_request(self, request_id: int) -> bool:
        async with self._cond:
            abort_event = self._pending_request_abort_events.get(request_id)
            if abort_event is None:
                return False
            abort_event.set()
            return True

    async def abort_all_pending_requests(self):
        async with self._cond:
            for abort_event in self._pending_request_abort_events.values():
                abort_event.set()

    async def is_pending_request(self, request_id: int) -> bool:
        async with self._cond:
            return request_id in self._pending_request_abort_events

    async def get_pending_request_count(self) -> int:
        async with self._cond:
            return len(self._pending_request_abort_events)

    async def _wait_until_running(self) -> None:
        async with self._cond:
            while self._state != self._RUNNING:
                await self._cond.wait()

    async def resume(self) -> None:
        async with self._cond:
            self._state = self._RUNNING
            self._cond.notify_all()


class HttpRlController:
    def __init__(self, manager) -> None:
        self.manager = manager
        self.args = manager.args
        self._generation_gate = _GenerationPauseGate()

    async def wait_until_generation_allowed(self, request_id: int) -> bool:
        await self._generation_gate.register_pending_request(request_id)
        try:
            return await self._generation_gate.wait_until_running_or_aborted(request_id)
        finally:
            await self._generation_gate.unregister_pending_request(request_id)

    async def wait_until_can_released_mark(self, req, timeout: float = 60.0) -> bool:
        start_time = time.time()
        while not req.can_released_mark:
            if time.time() - start_time > timeout:
                logger.warning(f"wait req can_released_mark timeout, req_id={req.request_id}, timeout={timeout}s")
                return False
            await asyncio.sleep(0.005)
        return True

    async def _wait_for_abort_released(
        self, request_id: Optional[int], abort_all: bool, timeout: float = 60.0
    ) -> Tuple[bool, str]:
        start_time = time.time()
        empty_since = None
        while True:
            if abort_all:
                pending_request_count = await self._generation_gate.get_pending_request_count()
                if len(self.manager.req_id_to_out_inf) == 0 and pending_request_count == 0:
                    empty_since = empty_since or time.time()
                    if time.time() - empty_since >= 1.0:
                        return True, ""
                else:
                    empty_since = None
                    await self._generation_gate.abort_all_pending_requests()
                    for group_req_id, req_status in list(self.manager.req_id_to_out_inf.items()):
                        if req_status is not None and any(
                            not req.is_aborted for req in req_status.group_req_objs.shm_req_objs
                        ):
                            await self.manager.abort(group_req_id)
            else:
                req_status = self.manager.req_id_to_out_inf.get(request_id, None)
                if req_status is not None:
                    if any(not req.is_aborted for req in req_status.group_req_objs.shm_req_objs):
                        await self.manager.abort(request_id)
                elif not await self._generation_gate.is_pending_request(request_id):
                    return True, ""

            if time.time() - start_time > timeout:
                error_msg = (
                    f"abort request wait release timeout, request_id={request_id}, abort_all={abort_all}, "
                    f"timeout={timeout}s"
                )
                logger.error(error_msg)
                return False, error_msg

            await asyncio.sleep(0.02)
        return True, ""

    async def abort_request(self, request: AbortReq) -> Tuple[bool, str]:
        request_id = request.request_id
        if request.abort_all:
            await self._generation_gate.abort_all_pending_requests()
            for group_req_id in list(self.manager.req_id_to_out_inf.keys()):
                await self.manager.abort(group_req_id)
            return await self._wait_for_abort_released(request_id=None, abort_all=True)

        if request_id is None:
            return True, ""

        await self._generation_gate.abort_pending_request(request_id)
        await self.manager.abort(request_id)
        return await self._wait_for_abort_released(request_id=request_id, abort_all=False)

    async def pause_generation(self):
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
