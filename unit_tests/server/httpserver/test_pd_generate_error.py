import asyncio
import pickle
from contextlib import aclosing, suppress
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lightllm.server.core.objs import FinishStatus, SamplingParams
from lightllm.server.multimodal_params import MultimodalParams
from lightllm.server.httpserver.pd_loop import _pd_process_generate
from lightllm.server.httpserver_for_pd_master.manager import (
    HttpServerManagerForPDMaster,
    ReqStatus,
)
from lightllm.server.pd_io_struct import (
    ObjType,
    PD_Client_Obj,
    PDAllocationReady,
    PDDecodeAllocationPolicy,
    PDUpKVStatus,
)
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
    "node_mode, finish",
    [("prefill", FinishStatus.FINISHED_ABORTED), ("decode", FinishStatus.FINISHED_ERROR)],
)
def test_deferred_terminal_packet_wakes_pending_handshake(node_mode, finish):
    async def run():
        manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
        manager.args = SimpleNamespace(config_server_host=None)
        manager.pd_manager = MagicMock()
        manager.metric_client = MagicMock()
        manager.timer_log = AsyncMock()
        manager.infos_queues = None
        req_id = 128
        status = ReqStatus(req_id, None, None, awaiting_deferred_bind=True)
        manager.req_id_to_out_inf = {req_id: status}
        handle = asyncio.create_task(manager.handle_loop())
        try:
            while manager.infos_queues is None:
                await asyncio.sleep(0)
            if node_mode == "decode":
                status.prefill_prompt_ids_event.set()
                status.allocation_ready_event.set()
            token = (req_id, "", {"node_mode": node_mode}, FinishStatus(finish))
            await manager.put_to_handle_queue((ObjType.TOKEN_PACKS, [token], {}))
            await asyncio.gather(
                status.prefill_prompt_ids_event.wait(),
                status.allocation_ready_event.wait(),
                status.up_status_event.wait(),
            )
            with pytest.raises(RuntimeError, match="before decode binding"):
                status.raise_if_error()
            assert not status.out_token_info_list
        finally:
            handle.cancel()
            with suppress(asyncio.CancelledError):
                await handle

    asyncio.run(asyncio.wait_for(run(), 2))


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
        manager.args = SimpleNamespace(pd_node_id=1, pd_decode_allocation_policy="early")
        manager.req_id_to_out_inf = {}
        manager.next_request_queue_metric_time = 0.0
        manager.next_pd_handshake_metric_time = 0.0
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
            observations = [call.args[0] for call in manager.metric_client.histogram_observe.call_args_list]
            assert observations.count("lightllm_pd_master_bind_first_token_duration") == 1
            assert observations.count("lightllm_pd_master_first_decode_gap_duration") == int(output_index == 2)
        finally:
            await stream.aclose()

    asyncio.run(run())


def test_deferred_old_boot_readiness_and_binding_do_not_wake_new_attempt():
    async def run():
        from lightllm.server.httpserver.req_id_generator_server import _create_req_id_generator_state

        args = SimpleNamespace(
            run_mode="pd_master",
            pd_decode_allocation_policy="last_chunk",
            config_server_host=None,
            config_server_port=None,
        )
        with patch("lightllm.server.httpserver.req_id_generator_server.secrets.randbelow", side_effect=[100, 200]):
            old, _ = _create_req_id_generator_state(args).allocate_id_range(1)
            current, _ = _create_req_id_generator_state(args).allocate_id_range(1)
        manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
        status = ReqStatus(current, None, None)
        manager.req_id_to_out_inf = {current: status}
        await manager.update_req_status(PDAllocationReady(old, 1, float("inf")))
        await manager.update_req_status(PDUpKVStatus(old, 1, b"stale D binding"))
        assert not status.allocation_ready_event.is_set() and not status.up_status_event.is_set()
        await manager.update_req_status(PDAllocationReady(current, 1, float("inf")))
        assert status.allocation_ready_event.is_set()

    asyncio.run(run())


@pytest.mark.parametrize("policy", [PDDecodeAllocationPolicy.PREFILL_COMPLETE, PDDecodeAllocationPolicy.LAST_CHUNK])
def test_deferred_prefill_submits_decode_once_after_ready(policy):
    async def run():
        manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
        manager.args = SimpleNamespace(
            pd_node_id=1, pd_decode_allocation_policy=policy.name.lower(), pd_prefill_ready_timeout=5
        )
        manager.req_id_to_out_inf = {}
        manager.metric_client = MagicMock()
        manager.next_request_queue_metric_time = 0.0
        manager.next_pd_handshake_metric_time = 0.0
        prompt_sent = asyncio.Event()
        req_id = 128

        async def send_p(payload):
            typ, *body = pickle.loads(payload)
            status = manager.req_id_to_out_inf[req_id]
            if typ == ObjType.REQ:
                _, params, _ = body[0]
                assert params.pd_decode_allocation_policy == policy
                status.prefill_prompt_ids_event.prompt_ids = [1, 2]
                status.prefill_prompt_ids_event.set()
                prompt_sent.set()
            else:
                assert typ == ObjType.PD_BIND_DECODE_NODE_INFO
                assert body[1].ready_kv_len == 1  # Late full hit: P still owns the sampled token.
                status.append_token(
                    (
                        req_id,
                        "P",
                        {"count_output_tokens": 1, "node_mode": "decode"},
                        FinishStatus(FinishStatus.NO_FINISH),
                    )
                )
                status.append_token(
                    (
                        req_id,
                        "P",
                        {"count_output_tokens": 1, "node_mode": "prefill", "prompt_cache_len": 0},
                        FinishStatus(FinishStatus.FINISHED_LENGTH),
                    )
                )

        async def send_d(payload):
            typ, (prompt, params, _) = pickle.loads(payload)
            assert typ == ObjType.REQ and prompt == [1, 2]
            assert params.pd_decode_allocation_policy == policy
            await manager.update_req_status(
                PDUpKVStatus(
                    group_request_id=req_id,
                    pd_master_node_id=1,
                    pd_kv_trans_params=pickle.dumps(SimpleNamespace(ready_kv_len=1)),
                )
            )

        p_node = SimpleNamespace(send_control_message=AsyncMock(side_effect=send_p))
        d_node = SimpleNamespace(send_control_message=AsyncMock(side_effect=send_d))
        params = SamplingParams()
        params.group_request_id = req_id
        params.max_new_tokens = 4
        stream = manager.fetch_pd_stream(
            p_node,
            d_node,
            "prompt",
            params,
            MultimodalParams(),
            SimpleNamespace(is_disconnected=AsyncMock(return_value=False)),
        )
        async with aclosing(stream):
            first = asyncio.create_task(anext(stream))
            await asyncio.wait_for(prompt_sent.wait(), timeout=1)
            await asyncio.sleep(0)
            d_node.send_control_message.assert_not_awaited()
            for _ in range(2):
                await manager.update_req_status(PDAllocationReady(req_id, 1, asyncio.get_running_loop().time() + 5))
            result = await asyncio.wait_for(first, timeout=1)
            assert result[1] == "P" and result[2]["prompt_cache_len"] == 0
            assert not result[3].is_finished()
            d_node.send_control_message.assert_awaited_once()
        await manager.remove_req(req_id)
        await manager.update_req_status(PDAllocationReady(req_id, 1, asyncio.get_running_loop().time() + 5))
        assert not manager.req_id_to_out_inf

    asyncio.run(run())


@pytest.mark.parametrize("d_first", [False, True])
@pytest.mark.parametrize("finish", [FinishStatus.FINISHED_LENGTH, FinishStatus.FINISHED_STOP, FinishStatus.NO_FINISH])
def test_deferred_first_output_waits_for_delivery_and_uses_decode_stop_status(d_first, finish):
    async def run():
        manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
        status = ReqStatus(128, None, None)
        p = (
            128,
            "first",
            {
                "node_mode": "prefill",
                "count_output_tokens": 1,
                "prompt_cache_len": 10,
                "input_usage": {"input_image_tokens": 4},
            },
            FinishStatus(FinishStatus.FINISHED_LENGTH),
        )
        d = (128, "first", {"node_mode": "decode", "count_output_tokens": 1}, FinishStatus(finish))
        first, second = (d, p) if d_first else (p, d)
        status.append_token(first)
        wait = asyncio.create_task(
            manager._wait_for_prefill_token_if_needed(
                status, SimpleNamespace(is_disconnected=AsyncMock(return_value=False)), 128, True, 127, deferred=True
            )
        )
        while status.out_token_info_list:
            await asyncio.sleep(0)
        assert not wait.done()
        status.append_token(second)
        assert await asyncio.wait_for(wait, 1) == 10
        assert status.pop_all_tokens() == [d]
        assert d[2]["input_usage"] == p[2]["input_usage"]

    asyncio.run(asyncio.wait_for(run(), 2))


def test_deferred_source_failure_preempts_buffered_decode_success():
    async def run():
        manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
        status = ReqStatus(128, None, None)
        status.append_token(
            (
                128,
                "first",
                {"node_mode": "decode", "count_output_tokens": 1},
                FinishStatus(FinishStatus.FINISHED_LENGTH),
            )
        )
        error = (128, "", {"node_mode": "prefill", "count_output_tokens": 1}, FinishStatus(FinishStatus.FINISHED_ERROR))
        status.append_token(error)
        await manager._wait_for_prefill_token_if_needed(
            status, SimpleNamespace(is_disconnected=AsyncMock(return_value=False)), 128, True, 127, deferred=True
        )
        assert status.pop_all_tokens() == [error]

    asyncio.run(asyncio.wait_for(run(), 2))


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


@pytest.mark.parametrize("event_name", ["prefill_prompt_ids_event", "up_status_event", "allocation_ready_event"])
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
