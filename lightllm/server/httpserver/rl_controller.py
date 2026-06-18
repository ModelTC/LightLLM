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
from lightllm.utils.error_utils import ServerBusyError
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class _GenerationPauseGate:
    """Generation pause gate: RUNNING / PAUSED(queue new requests) / PAUSED_REJECT(reject new requests)."""

    _RUNNING = 0
    _PAUSED = 1
    _PAUSED_REJECT = 2

    def __init__(self) -> None:
        self._state = self._RUNNING
        self._cond = asyncio.Condition()

    @asynccontextmanager
    async def pause_and_abort_context(self, reject_new: bool):
        async with self._cond:
            if self._state != self._RUNNING:
                if reject_new and self._state != self._PAUSED_REJECT:
                    self._state = self._PAUSED_REJECT
                    self._cond.notify_all()
                yield False
                return
            self._state = self._PAUSED_REJECT if reject_new else self._PAUSED
            if reject_new:
                self._cond.notify_all()
            yield True

    async def wait_until_running(self) -> None:
        async with self._cond:
            while self._state != self._RUNNING:
                if self._state == self._PAUSED_REJECT:
                    raise ServerBusyError("Generation is paused, new requests are rejected")
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

    async def wait_until_generation_allowed(self) -> None:
        await self._generation_gate.wait_until_running()

    async def _wait_for_abort_released(
        self, request_id: Optional[int], abort_all: bool, timeout: float = 60.0
    ) -> Tuple[bool, str]:
        start_time = time.time()
        empty_since = None
        while True:
            if abort_all:
                if len(self.manager.req_id_to_out_inf) == 0:
                    empty_since = empty_since or time.time()
                    if time.time() - empty_since >= 1.0:
                        return True, ""
                else:
                    empty_since = None
                    for group_req_id, req_status in list(self.manager.req_id_to_out_inf.items()):
                        if req_status is not None and any(
                            not req.is_aborted for req in req_status.group_req_objs.shm_req_objs
                        ):
                            await self.manager.abort(group_req_id)
            elif request_id not in self.manager.req_id_to_out_inf:
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
            for group_req_id in list(self.manager.req_id_to_out_inf.keys()):
                await self.manager.abort(group_req_id)
            return await self._wait_for_abort_released(request_id=None, abort_all=True)

        if request_id is None:
            return True, ""

        await self.manager.abort(request_id)
        return await self._wait_for_abort_released(request_id=request_id, abort_all=False)

    async def pause_generation(self, reject_new: bool = False):
        # In multinode TP mode, the master HTTP node gates incoming requests before they reach slave nodes.
        async with self._generation_gate.pause_and_abort_context(reject_new) as do_abort:
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
