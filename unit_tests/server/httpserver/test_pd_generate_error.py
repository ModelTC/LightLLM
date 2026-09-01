import asyncio
import pickle
from contextlib import suppress
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from lightllm.server.core.objs import FinishStatus, SamplingParams
from lightllm.server.httpserver.decode_admission import DecodeAdmissionController
from lightllm.server.httpserver.pd_loop import _pd_process_generate, _reserve_decode_slots
from lightllm.server.httpserver_for_pd_master.manager import (
    HttpServerManagerForPDMaster,
    ReqStatus,
)
from lightllm.server.pd_io_struct import ObjType
from lightllm.utils.error_utils import PDPrefillNodeStopGenToken, ServerBusyError


class _FailingManager:
    args = SimpleNamespace(run_mode="prefill")

    async def generate(self, **_kwargs):
        raise RuntimeError("prefill failed")
        yield


class _CancelledManager:
    args = SimpleNamespace(run_mode="prefill")

    async def generate(self, **_kwargs):
        raise asyncio.CancelledError()
        yield


class _BusyManager:
    args = SimpleNamespace(run_mode="decode")

    async def generate(self, **_kwargs):
        raise ServerBusyError("decode queue is full")
        yield


class _SuccessfulManager:
    args = SimpleNamespace(run_mode="prefill")

    async def generate(self, **_kwargs):
        yield 123, "token", {}, FinishStatus(FinishStatus.FINISHED_STOP)


class _StopPrefillManager:
    args = SimpleNamespace(run_mode="prefill")

    async def generate(self, **_kwargs):
        raise PDPrefillNodeStopGenToken(group_request_id=123)
        yield


class _FatalGenerateError(BaseException):
    pass


class _FatalManager:
    args = SimpleNamespace(run_mode="decode")

    async def generate(self, **_kwargs):
        raise _FatalGenerateError("fatal generate failure")
        yield


def test_pd_node_reports_generate_error_to_master():
    async def run():
        sampling_params = SamplingParams()
        sampling_params.group_request_id = 123
        websocket = AsyncMock()

        await _pd_process_generate(
            manager=_FailingManager(),
            prompt="prompt",
            sampling_params=sampling_params,
            multimodal_params=MagicMock(),
            forwarding_queue=MagicMock(),
            pd_upload_websocket=websocket,
            pd_event=asyncio.Event(),
        )

        websocket.send.assert_awaited_once()
        obj = pickle.loads(websocket.send.await_args.args[0])
        assert obj == (
            ObjType.PD_UPLOAD_GENERATE_ERROR,
            123,
            "RuntimeError: prefill failed",
        )

    asyncio.run(run())


def test_pd_node_cancellation_finishes_without_reporting_generate_error():
    async def run():
        sampling_params = SamplingParams()
        sampling_params.group_request_id = 123
        websocket = AsyncMock()

        await _pd_process_generate(
            manager=_CancelledManager(),
            prompt="prompt",
            sampling_params=sampling_params,
            multimodal_params=MagicMock(),
            forwarding_queue=MagicMock(),
            pd_upload_websocket=websocket,
            pd_event=asyncio.Event(),
        )

        websocket.send.assert_not_awaited()

    asyncio.run(run())


def test_pd_node_reports_server_busy_separately():
    async def run():
        sampling_params = SamplingParams()
        sampling_params.group_request_id = 123
        websocket = AsyncMock()

        await _pd_process_generate(
            manager=_BusyManager(),
            prompt="prompt",
            sampling_params=sampling_params,
            multimodal_params=MagicMock(),
            forwarding_queue=MagicMock(),
            pd_upload_websocket=websocket,
            pd_event=asyncio.Event(),
        )

        obj = pickle.loads(websocket.send.await_args.args[0])
        assert obj == (ObjType.PD_UPLOAD_SERVER_BUSY, 123, "decode queue is full")

    asyncio.run(run())


def test_pd_node_success_forwards_token_without_error_report():
    async def run():
        sampling_params = SamplingParams()
        sampling_params.group_request_id = 123
        websocket = AsyncMock()
        forwarding_queue = AsyncMock()

        await _pd_process_generate(
            manager=_SuccessfulManager(),
            prompt="prompt",
            sampling_params=sampling_params,
            multimodal_params=MagicMock(),
            forwarding_queue=forwarding_queue,
            pd_upload_websocket=websocket,
            pd_event=asyncio.Event(),
        )

        forwarding_queue.put.assert_awaited_once()
        forwarded = forwarding_queue.put.await_args.args[0]
        assert forwarded[0] == 123
        assert forwarded[1] == "token"
        assert forwarded[2]["node_mode"] == "prefill"
        assert forwarded[3].is_finished()
        websocket.send.assert_not_awaited()

    asyncio.run(run())


def test_pd_prefill_full_cache_stop_is_not_reported_as_error():
    async def run():
        sampling_params = SamplingParams()
        sampling_params.group_request_id = 123
        websocket = AsyncMock()

        await _pd_process_generate(
            manager=_StopPrefillManager(),
            prompt="prompt",
            sampling_params=sampling_params,
            multimodal_params=MagicMock(),
            forwarding_queue=AsyncMock(),
            pd_upload_websocket=websocket,
            pd_event=asyncio.Event(),
        )

        websocket.send.assert_not_awaited()

    asyncio.run(run())


def test_pd_node_reports_non_exception_base_exception():
    async def run():
        sampling_params = SamplingParams()
        sampling_params.group_request_id = 123
        websocket = AsyncMock()

        await _pd_process_generate(
            manager=_FatalManager(),
            prompt="prompt",
            sampling_params=sampling_params,
            multimodal_params=MagicMock(),
            forwarding_queue=AsyncMock(),
            pd_upload_websocket=websocket,
            pd_event=asyncio.Event(),
        )

        obj = pickle.loads(websocket.send.await_args.args[0])
        assert obj == (
            ObjType.PD_UPLOAD_GENERATE_ERROR,
            123,
            "_FatalGenerateError: fatal generate failure",
        )

    asyncio.run(run())


def test_pd_node_error_report_send_failure_is_contained():
    async def run():
        sampling_params = SamplingParams()
        sampling_params.group_request_id = 123
        websocket = AsyncMock()
        websocket.send.side_effect = ConnectionError("websocket closed")

        await _pd_process_generate(
            manager=_FailingManager(),
            prompt="prompt",
            sampling_params=sampling_params,
            multimodal_params=MagicMock(),
            forwarding_queue=AsyncMock(),
            pd_upload_websocket=websocket,
            pd_event=asyncio.Event(),
        )

        websocket.send.assert_awaited_once()

    asyncio.run(run())


def test_pd_decode_node_reserves_n_slots_and_hands_out_independent_leases():
    async def run():
        manager = SimpleNamespace(
            decode_admission_controller=DecodeAdmissionController(capacity=3, max_queued_slots=0, timeout_seconds=1),
            metric_client=MagicMock(),
        )
        websocket = AsyncMock()
        handles = {}

        await _reserve_decode_slots(manager, 100, (11, 22, 33), handles, websocket)

        assert manager.decode_admission_controller.active_slots == 3
        assert set(handles) == {11, 22, 33}
        assert pickle.loads(websocket.send.await_args.args[0]) == (ObjType.PD_DECODE_SLOTS_RESERVED, 100)

        first = handles.pop(11).take()
        first.release()
        handles.pop(22).release()
        handles.pop(33).release()
        assert manager.decode_admission_controller.active_slots == 0

    asyncio.run(run())


def test_pd_decode_node_cancels_queued_atomic_reservation_without_leaking():
    async def run():
        controller = DecodeAdmissionController(capacity=2, max_queued_slots=2, timeout_seconds=10)
        active = await controller.acquire(2)
        manager = SimpleNamespace(decode_admission_controller=controller, metric_client=MagicMock())
        websocket = AsyncMock()
        handles = {}
        task = asyncio.create_task(_reserve_decode_slots(manager, 100, (11, 22), handles, websocket))
        await asyncio.sleep(0)

        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

        assert controller.queued_slots == 0
        assert handles == {}
        websocket.send.assert_not_awaited()
        active.release()
        assert controller.active_slots == 0

    asyncio.run(run())


def test_pd_master_routes_decode_reservation_ack_and_busy_response():
    async def run():
        manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
        manager.args = SimpleNamespace(config_server_host=None, config_server_port=None)
        manager.pd_manager = MagicMock()
        manager.timer_log = AsyncMock()
        manager.infos_queues = None
        manager.req_id_to_out_inf = {}
        manager.decode_reservation_statuses = {}
        d_node = SimpleNamespace(
            start_args={"pd_node_decode_admission_timeout": 1},
            websocket=SimpleNamespace(send_bytes=AsyncMock()),
        )
        request = SimpleNamespace(is_disconnected=AsyncMock(return_value=False))

        handle_task = asyncio.create_task(manager.handle_loop())
        try:
            while manager.infos_queues is None:
                await asyncio.sleep(0)

            accepted = asyncio.create_task(manager.reserve_decode_slots(d_node, 100, (11, 22), request))
            while 100 not in manager.decode_reservation_statuses:
                await asyncio.sleep(0)
            await manager.put_to_handle_queue((ObjType.PD_DECODE_SLOTS_RESERVED, 100))
            await asyncio.wait_for(accepted, timeout=1)

            rejected = asyncio.create_task(manager.reserve_decode_slots(d_node, 200, (33, 44), request))
            while 200 not in manager.decode_reservation_statuses:
                await asyncio.sleep(0)
            await manager.put_to_handle_queue((ObjType.PD_UPLOAD_SERVER_BUSY, 200, "decode queue is full"))
            with pytest.raises(ServerBusyError, match="decode queue is full"):
                await asyncio.wait_for(rejected, timeout=1)

            sent = [pickle.loads(call.args[0]) for call in d_node.websocket.send_bytes.await_args_list]
            assert sent == [
                (ObjType.PD_RESERVE_DECODE_SLOTS, 100, (11, 22)),
                (ObjType.PD_RESERVE_DECODE_SLOTS, 200, (33, 44)),
            ]
            assert manager.decode_reservation_statuses == {}
        finally:
            handle_task.cancel()
            with suppress(asyncio.CancelledError):
                await handle_task

    asyncio.run(run())


def test_pd_master_generate_error_marks_request_and_wakes_all_waiters():
    async def run():
        manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
        manager.args = SimpleNamespace(config_server_host=None)
        manager.pd_manager = MagicMock()
        manager.timer_log = AsyncMock()
        manager.infos_queues = None
        manager.decode_reservation_statuses = {}

        p_node = SimpleNamespace(websocket=SimpleNamespace(send_bytes=AsyncMock()))
        d_node = SimpleNamespace(websocket=SimpleNamespace(send_bytes=AsyncMock()))
        req_status = ReqStatus(123, p_node, d_node)
        manager.req_id_to_out_inf = {123: req_status}

        handle_task = asyncio.create_task(manager.handle_loop())
        try:
            while manager.infos_queues is None:
                await asyncio.sleep(0)
            await manager.put_to_handle_queue((ObjType.PD_UPLOAD_GENERATE_ERROR, 123, "RuntimeError: prefill failed"))
            await asyncio.wait_for(req_status.event.wait(), timeout=1)

            assert req_status.error_info == "RuntimeError: prefill failed"
            assert req_status.prefill_prompt_ids_event.is_set()
            assert req_status.up_status_event.is_set()
            assert manager.req_id_to_out_inf[123] is req_status
            p_node.websocket.send_bytes.assert_not_awaited()
            d_node.websocket.send_bytes.assert_not_awaited()

            with pytest.raises(
                RuntimeError,
                match="PD node generate failed: RuntimeError: prefill failed",
            ):
                req_status.raise_if_error()
        finally:
            handle_task.cancel()
            with suppress(asyncio.CancelledError):
                await handle_task

    asyncio.run(run())


def test_pd_master_server_busy_wakes_waiters_and_preserves_429():
    async def run():
        manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
        manager.args = SimpleNamespace(config_server_host=None)
        manager.pd_manager = MagicMock()
        manager.timer_log = AsyncMock()
        manager.infos_queues = None

        req_status = ReqStatus(123, MagicMock(), MagicMock())
        manager.req_id_to_out_inf = {123: req_status}
        handle_task = asyncio.create_task(manager.handle_loop())
        try:
            while manager.infos_queues is None:
                await asyncio.sleep(0)
            await manager.put_to_handle_queue((ObjType.PD_UPLOAD_SERVER_BUSY, 123, "decode queue is full"))
            await asyncio.wait_for(req_status.event.wait(), timeout=1)

            with pytest.raises(ServerBusyError, match="decode queue is full"):
                req_status.raise_if_error()
        finally:
            handle_task.cancel()
            with suppress(asyncio.CancelledError):
                await handle_task

    asyncio.run(run())


@pytest.mark.parametrize("event_name", ["prefill_prompt_ids_event", "up_status_event"])
def test_pd_master_generate_error_wakes_resource_wait(event_name):
    async def run():
        manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
        req_status = ReqStatus(123, MagicMock(), MagicMock())
        request = SimpleNamespace(is_disconnected=AsyncMock(return_value=False))
        wait_task = asyncio.create_task(
            manager._wait_for_event_or_disconnect(
                getattr(req_status, event_name),
                request,
                timeout=60,
                group_request_id=123,
                stage="test",
            )
        )

        await req_status.set_error("RuntimeError: node failed")
        await asyncio.wait_for(wait_task, timeout=1)

        with pytest.raises(
            RuntimeError,
            match="PD node generate failed: RuntimeError: node failed",
        ):
            req_status.raise_if_error()

    asyncio.run(run())


def test_pd_master_abort_removes_request_even_when_node_notifications_fail():
    async def run():
        manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
        manager.req_id_to_out_inf = {}
        p_node = SimpleNamespace(websocket=SimpleNamespace(send_bytes=AsyncMock(side_effect=ConnectionError("p down"))))
        d_node = SimpleNamespace(websocket=SimpleNamespace(send_bytes=AsyncMock(side_effect=ConnectionError("d down"))))
        manager.req_id_to_out_inf[123] = ReqStatus(123, p_node, d_node)

        await manager.abort(123)

        assert 123 not in manager.req_id_to_out_inf

    asyncio.run(run())


def test_pd_master_abort_uses_explicit_nodes_when_request_status_is_missing():
    async def run():
        manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
        manager.req_id_to_out_inf = {}
        p_node = SimpleNamespace(websocket=SimpleNamespace(send_bytes=AsyncMock()))
        d_node = SimpleNamespace(websocket=SimpleNamespace(send_bytes=AsyncMock()))

        await manager.abort(123, p_node=p_node, d_node=d_node)

        p_node.websocket.send_bytes.assert_awaited_once_with(pickle.dumps((ObjType.ABORT, 123)))
        d_node.websocket.send_bytes.assert_awaited_once_with(pickle.dumps((ObjType.ABORT, 123)))

    asyncio.run(run())
