import asyncio
import pickle
from contextlib import suppress
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from lightllm.server.core.objs import FinishStatus, SamplingParams
from lightllm.server.multimodal_params import MultimodalParams
from lightllm.server.httpserver.pd_loop import _pd_process_generate
from lightllm.server.httpserver_for_pd_master.manager import (
    HttpServerManagerForPDMaster,
    ReqStatus,
)
from lightllm.server.pd_io_struct import ObjType, PD_Client_Obj
from lightllm.utils.error_utils import InvalidRequestError, PDPrefillNodeStopGenToken, ServerBusyError


class _PDManagerStub:
    def cancel_pd_request_registration(self, _group_request_id):
        pass


class _FailingManager(_PDManagerStub):
    args = SimpleNamespace(run_mode="prefill")

    async def generate(self, **_kwargs):
        raise RuntimeError("prefill failed")
        yield


class _CancelledManager(_PDManagerStub):
    args = SimpleNamespace(run_mode="prefill")

    async def generate(self, **_kwargs):
        raise asyncio.CancelledError()
        yield


class _SuccessfulManager(_PDManagerStub):
    args = SimpleNamespace(run_mode="prefill")

    async def generate(self, **_kwargs):
        yield 123, "token", {"count_output_tokens": 1}, FinishStatus(FinishStatus.FINISHED_STOP)


class _StopPrefillManager(_PDManagerStub):
    args = SimpleNamespace(run_mode="prefill")

    async def generate(self, **_kwargs):
        raise PDPrefillNodeStopGenToken(group_request_id=123)
        yield


class _FatalGenerateError(BaseException):
    pass


class _FatalManager(_PDManagerStub):
    args = SimpleNamespace(run_mode="decode")

    async def generate(self, **_kwargs):
        raise _FatalGenerateError("fatal generate failure")
        yield


class _BusyManager(_PDManagerStub):
    args = SimpleNamespace(run_mode="decode")

    async def generate(self, **_kwargs):
        raise ServerBusyError("decode node could not allocate a shm_req object within 20 seconds")
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
            heartbeat_reporter=None,
        )

        websocket.send.assert_awaited_once()
        obj = pickle.loads(websocket.send.await_args.args[0])
        assert obj == (
            ObjType.PD_UPLOAD_GENERATE_ERROR,
            123,
            "RuntimeError: prefill failed",
        )

    asyncio.run(run())


def test_pd_node_reports_local_request_rejection_to_master():
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
            heartbeat_reporter=None,
        )

        websocket.send.assert_awaited_once()
        obj = pickle.loads(websocket.send.await_args.args[0])
        assert obj == (
            ObjType.PD_UPLOAD_SERVER_BUSY,
            123,
            "decode node could not allocate a shm_req object within 20 seconds",
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
            heartbeat_reporter=None,
        )

        websocket.send.assert_not_awaited()

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
            heartbeat_reporter=None,
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
            heartbeat_reporter=None,
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
            heartbeat_reporter=None,
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
            heartbeat_reporter=None,
        )

        websocket.send.assert_awaited_once()

    asyncio.run(run())


def test_pd_master_generate_error_marks_request_and_wakes_all_waiters():
    async def run():
        manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
        manager._abort_notify_tasks = set()
        manager.args = SimpleNamespace(config_server_host=None)
        manager.pd_manager = MagicMock()
        manager.timer_log = AsyncMock()
        manager.metric_client = MagicMock()
        manager.infos_queues = None

        p_node = SimpleNamespace(send_control_message=AsyncMock())
        d_node = SimpleNamespace(send_control_message=AsyncMock())
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
            p_node.send_control_message.assert_not_awaited()
            d_node.send_control_message.assert_not_awaited()

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


@pytest.mark.parametrize(
    "terminal_status, output_index",
    [
        (FinishStatus.FINISHED_ERROR, 1),
        (FinishStatus.FINISHED_ABORTED, 1),
        (FinishStatus.FINISHED_LENGTH, 2),
        (FinishStatus.FINISHED_STOP, 2),
    ],
)
def test_pd_master_first_token_dedup_preserves_terminal_status(terminal_status, output_index):
    async def run():
        manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
        manager.args = SimpleNamespace(pd_node_id=1)
        manager.req_id_to_out_inf = {}
        manager.next_request_queue_metric_time = 0.0
        manager.metric_client = MagicMock()
        p_node = SimpleNamespace(send_control_message=AsyncMock())
        d_node = SimpleNamespace(send_control_message=AsyncMock())
        sampling_params = SamplingParams()
        sampling_params.group_request_id = 123
        sampling_params.max_new_tokens = 4

        async def handshake(event, _request, **kwargs):
            if kwargs["stage"] == "prefill":
                event.prompt_ids = [1, 2]
            else:
                event.upkv_status = SimpleNamespace(pd_kv_trans_params=pickle.dumps(SimpleNamespace(ready_kv_len=0)))
                status = manager.req_id_to_out_inf[123]
                status.out_token_info_list.extend(
                    [
                        (
                            123,
                            "first",
                            {"count_output_tokens": 1, "node_mode": "prefill"},
                            FinishStatus(FinishStatus.FINISHED_LENGTH),
                        ),
                        (
                            123,
                            "duplicate",
                            {"count_output_tokens": 1, "node_mode": "decode"},
                            FinishStatus(FinishStatus.NO_FINISH),
                        ),
                        (
                            123,
                            "terminal",
                            {"count_output_tokens": output_index, "node_mode": "decode"},
                            FinishStatus(terminal_status),
                        ),
                    ]
                )
                status.event.set()

        manager._wait_for_event_or_disconnect = AsyncMock(side_effect=handshake)
        stream = manager.fetch_pd_stream(
            p_node,
            d_node,
            "prompt",
            sampling_params,
            MultimodalParams(),
            SimpleNamespace(is_disconnected=AsyncMock(return_value=False)),
        )
        try:
            first = await asyncio.wait_for(anext(stream), timeout=1)
            assert first[1] == "first"
            assert not first[3].is_finished()
            terminal = await asyncio.wait_for(anext(stream), timeout=1)
            assert terminal[1] == "terminal"
            assert terminal[3].status == terminal_status
        finally:
            await stream.aclose()

    asyncio.run(run())


def test_pd_master_request_rejection_becomes_server_busy_error():
    async def run():
        manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
        manager._abort_notify_tasks = set()
        manager.args = SimpleNamespace(config_server_host=None)
        manager.pd_manager = MagicMock()
        manager.timer_log = AsyncMock()
        manager.metric_client = MagicMock()
        manager.infos_queues = None

        req_status = ReqStatus(123, MagicMock(), MagicMock())
        manager.req_id_to_out_inf = {123: req_status}

        handle_task = asyncio.create_task(manager.handle_loop())
        try:
            while manager.infos_queues is None:
                await asyncio.sleep(0)
            await manager.put_to_handle_queue(
                (
                    ObjType.PD_UPLOAD_SERVER_BUSY,
                    123,
                    "prefill node could not allocate a shm_req object within 20 seconds",
                )
            )
            await asyncio.wait_for(req_status.event.wait(), timeout=1)

            assert req_status.is_server_busy is True
            with pytest.raises(ServerBusyError, match="prefill node could not allocate a shm_req object"):
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
        manager._abort_notify_tasks = set()
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
        manager._abort_notify_tasks = set()
        manager.req_id_to_out_inf = {}
        p_node = SimpleNamespace(send_control_message=AsyncMock(side_effect=ConnectionError("p down")))
        d_node = SimpleNamespace(send_control_message=AsyncMock(side_effect=ConnectionError("d down")))
        manager.req_id_to_out_inf[123] = ReqStatus(123, p_node, d_node)

        await manager.abort(123)

        assert 123 not in manager.req_id_to_out_inf

    asyncio.run(run())


def test_pd_master_abort_uses_explicit_nodes_when_request_status_is_missing():
    async def run():
        manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
        manager._abort_notify_tasks = set()
        manager.req_id_to_out_inf = {}
        p_node = SimpleNamespace(send_control_message=AsyncMock())
        d_node = SimpleNamespace(send_control_message=AsyncMock())

        await manager.abort(123, p_node=p_node, d_node=d_node)

        p_node.send_control_message.assert_awaited_once_with(pickle.dumps((ObjType.ABORT, 123)))
        d_node.send_control_message.assert_awaited_once_with(pickle.dumps((ObjType.ABORT, 123)))

    asyncio.run(run())


def test_pd_master_abort_does_not_wait_for_disconnected_node_inflight_send():
    async def run():
        send_started = asyncio.Event()
        release_send = asyncio.Event()

        class _BlockingWebSocket:
            async def send_bytes(self, _payload):
                send_started.set()
                await release_send.wait()

        p_node = PD_Client_Obj(1, "prefill:8000", "prefill", {}, websocket=_BlockingWebSocket())
        d_node = SimpleNamespace(send_control_message=AsyncMock())
        inflight_send = asyncio.create_task(p_node.send_control_message(b"request"))
        await send_started.wait()

        manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
        manager._abort_notify_tasks = set()
        manager.req_id_to_out_inf = {123: ReqStatus(123, p_node, d_node)}
        p_node.websocket = None

        try:
            await asyncio.wait_for(manager.abort(123), timeout=1)
            d_node.send_control_message.assert_awaited_once_with(pickle.dumps((ObjType.ABORT, 123)))
        finally:
            release_send.set()
            await inflight_send

    asyncio.run(run())


def test_pd_invalid_request_error_round_trip():
    class InvalidManager(_PDManagerStub):
        args = SimpleNamespace(run_mode="prefill")

        async def generate(self, **_kwargs):
            raise InvalidRequestError("invalid image tag num: 0 images vs 1 tags")
            yield

    async def run():
        sampling_params = SamplingParams()
        sampling_params.group_request_id = 123
        websocket = AsyncMock()
        await _pd_process_generate(
            manager=InvalidManager(),
            prompt="prompt",
            sampling_params=sampling_params,
            multimodal_params=MagicMock(),
            forwarding_queue=AsyncMock(),
            pd_upload_websocket=websocket,
            pd_event=asyncio.Event(),
            heartbeat_reporter=None,
        )
        kind, req_id, error_info = pickle.loads(websocket.send.await_args.args[0])
        assert kind == ObjType.PD_UPLOAD_GENERATE_ERROR
        assert req_id == 123
        status = ReqStatus(req_id, MagicMock(), MagicMock())
        await status.set_error(error_info)
        with pytest.raises(InvalidRequestError, match="invalid image tag num: 0 images vs 1 tags"):
            status.raise_if_error()

    asyncio.run(run())


def test_pd_stream_validation_returns_bad_request(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from lightllm.server import api_stream_obj
    from lightllm.server.api_http import g_objs, invalid_request_exception_handler
    from lightllm.server.api_openai import _safe_stream_wrapper

    monkeypatch.setattr(
        api_stream_obj,
        "get_env_start_args",
        lambda: SimpleNamespace(disable_delay_response_start=False, run_mode="normal"),
    )
    monkeypatch.setattr(g_objs, "metric_client", MagicMock())
    app = FastAPI()
    app.add_exception_handler(InvalidRequestError, invalid_request_exception_handler)

    async def generate():
        status = ReqStatus(123, MagicMock(), MagicMock())
        await status.set_error("InvalidRequestError: invalid image tag num: 0 images vs 1 tags")
        status.raise_if_error()
        yield

    @app.get("/")
    async def stream():
        return api_stream_obj.CustomStreamingResponse(_safe_stream_wrapper(generate()), media_type="text/event-stream")

    response = TestClient(app).get("/")
    assert response.status_code == 400
    assert "invalid image tag num" in response.text
