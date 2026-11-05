import asyncio
import pickle
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lightllm.server.core.objs import StopStrMatchMode, FinishStatus, SamplingParams
from lightllm.server.httpserver.manager import HttpServerManager, ReqStatus
from lightllm.server.pd_io_struct import NodeRole, ObjType
from lightllm.utils.error_utils import ClientDisconnected, PDPrefillNodeStopGenToken, ServerBusyError


class _ValueMark:
    def __init__(self, value=0):
        self.value = value

    def get_value(self):
        return self.value

    def set_value(self, value):
        self.value = value


def _make_manager(mode: NodeRole):
    manager = HttpServerManager.__new__(HttpServerManager)
    manager.args = SimpleNamespace(
        run_mode=mode.value,
        running_max_req_size=2,
        reasoning_parser=None,
        enable_rl=False,
        enable_return_routed_experts=False,
    )
    manager.pd_mode = mode
    manager.is_multinode_tp_slave = False
    manager.alloc_req_id = MagicMock(return_value=123)
    manager.is_multinode_tp_master = False
    manager.rl_controller = None
    manager.output_grammar_compiler = None
    manager.req_id_to_out_inf = {}
    manager._log_stage_timing = MagicMock()
    manager._log_req_header = AsyncMock()
    manager._encode = AsyncMock(return_value=[10, 11, 12])
    manager._check_and_repair_length = AsyncMock(side_effect=lambda prompt_ids, _params: prompt_ids)
    manager._release_multimodal_resources = AsyncMock()
    manager.abort = AsyncMock()
    manager._register_running_request = AsyncMock()
    manager._unregister_running_request = AsyncMock()
    manager.metric_client = MagicMock()
    manager._run_reqs_count_lock = asyncio.Lock()
    manager.run_reqs_count_mark = _ValueMark()
    manager.shm_req_manager = SimpleNamespace(
        async_alloc_req_indexes=AsyncMock(side_effect=RuntimeError("alloc failed"))
    )
    return manager


def _sampling_params():
    sampling_params = SamplingParams()
    sampling_params.group_request_id = 123
    sampling_params.n = 1
    sampling_params.max_new_tokens = 1
    sampling_params.pd_node_resource_wait_timeout_seconds = 20
    return sampling_params


def _multimodal_params():
    return SimpleNamespace(audios=[], images=[], verify_and_preload=AsyncMock())


@pytest.mark.parametrize("mode", [NodeRole.NORMAL, NodeRole.P, NodeRole.D])
@pytest.mark.parametrize("warm_cache", [False, True])
def test_http_publishes_grammar_for_all_choices_before_forwarding(compiler, mode, warm_cache):
    import xgrammar as xgr

    async def run():
        manager = _make_manager(mode)
        manager.output_grammar_compiler = compiler
        manager.tokenizer = None
        manager.enable_multimodal = False
        manager.args.chunked_prefill_size = 1
        manager._alloc_shm_req_indexes = AsyncMock(return_value=[0, 1])
        reqs = [MagicMock(), MagicMock()]
        manager.shm_req_manager.async_get_req_obj_by_index = AsyncMock(side_effect=reqs)

        async def forward(*args):
            payloads = []
            for req in reqs:
                req.init.assert_called_once()
                req.set_compiled_grammar.assert_called_once()
                payload = req.set_compiled_grammar.call_args.args[0]
                payloads.append(payload)
                grammar = compiler.output_constraint_state_factory.get_grammar(payload)
                matcher = xgr.GrammarMatcher(grammar)
                assert matcher.accept_token(ord("a")) and matcher.accept_token(ord("b"))
                assert matcher.accept_token(256) and matcher.is_terminated()
            assert payloads[0] is payloads[1]

        manager.transfer_to_next_module_or_node = AsyncMock(side_effect=forward)

        async def results(*args):
            for req_id in (123, 124):
                yield req_id, "", {"id": 256}, FinishStatus(FinishStatus.FINISHED_STOP)

        manager._wait_to_token_package = results
        params = SamplingParams()
        params.init(None, regular_constraint="ab", n=2, best_of=2, max_new_tokens=1, group_request_id=123)
        with patch.object(compiler, "_compile", wraps=compiler._compile) as compile_grammar:
            if warm_cache:
                await params.verify_cpu_async(compiler)
            params = pickle.loads(pickle.dumps(SamplingParams.from_buffer_copy(params)))
            outputs = [result async for result in manager.generate("prompt", params, _multimodal_params(), None)]
            assert len(outputs) == 2
            compile_grammar.assert_called_once_with("regex", "ab")
        manager.transfer_to_next_module_or_node.assert_awaited_once()
        assert not hasattr(params, "compiled_grammar")

    asyncio.run(run())


@pytest.mark.parametrize("mode", [NodeRole.NORMAL, NodeRole.P, NodeRole.D])
def test_grammar_get_failure_precedes_request_admission_and_pd_handshake(mode):
    async def run():
        manager = _make_manager(mode)
        manager.output_grammar_compiler = SimpleNamespace(compile=AsyncMock(side_effect=ValueError("compile failed")))
        manager._alloc_shm_req_indexes = AsyncMock()
        params = SamplingParams()
        params.init(None, regular_constraint="ab", max_new_tokens=1, group_request_id=123)
        multimodal_params = _multimodal_params()
        websocket = AsyncMock() if mode == NodeRole.P else None
        pd_event = asyncio.Event() if mode == NodeRole.P else None
        with pytest.raises(ValueError, match="compile failed"):
            await _drain_generate(manager, params, multimodal_params, websocket, pd_event)
        manager.alloc_req_id.assert_not_called()
        manager._encode.assert_not_awaited()
        multimodal_params.verify_and_preload.assert_not_awaited()
        manager._alloc_shm_req_indexes.assert_not_awaited()
        manager._register_running_request.assert_not_awaited()
        manager._unregister_running_request.assert_not_awaited()
        manager.abort.assert_not_awaited()
        if websocket is not None:
            websocket.send.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.parametrize("mode", [NodeRole.NORMAL, NodeRole.D, NodeRole.P])
@pytest.mark.parametrize("include", [False, True])
@pytest.mark.parametrize("parser", [None, "qwen3"])
@pytest.mark.parametrize("stop_mode", [None, StopStrMatchMode.ALL_TEXT, StopStrMatchMode.CONTENT_ONLY])
def test_http_forwards_packaged_tokens_and_preserves_stop_policy(mode, include, parser, stop_mode):
    async def run():
        manager = _make_manager(mode)
        manager.tokenizer = SimpleNamespace(encode=lambda text, **kwargs: [ord(c) for c in text])
        manager.enable_multimodal = False
        manager.args.chunked_prefill_size = 1
        manager.args.reasoning_parser = parser
        manager._alloc_shm_req_indexes = AsyncMock(return_value=[0])
        manager.shm_req_manager.async_get_req_obj_by_index = AsyncMock(return_value=MagicMock())
        manager.transfer_to_next_module_or_node = AsyncMock()

        async def results(*args):
            yield 123, "hello", {"id": 1}, FinishStatus()
            yield 123, "END" if include else "", {"id": 2}, FinishStatus(FinishStatus.FINISHED_STOP)

        manager._wait_to_token_package = results
        params = _sampling_params()
        params.init(
            manager.tokenizer,
            stop_sequences="END",
            include_stop_str_in_output=include,
            max_new_tokens=1,
            group_request_id=123,
            pd_node_resource_wait_timeout_seconds=20,
        )
        # P/D nodes keep the matching policy supplied by Master.
        if stop_mode is not None:
            params.stop_str_match_mode.set_mode(stop_mode)
        websocket = AsyncMock() if mode == NodeRole.P else None
        pd_event = None
        if mode == NodeRole.P:
            pd_event = asyncio.Event()
            pd_event.decode_node_info = SimpleNamespace(ready_kv_len=1)
            pd_event.set()

        results = [
            result
            async for result in manager.generate("prompt", params, _multimodal_params(), None, websocket, pd_event)
        ]
        expected_mode = StopStrMatchMode.ALL_TEXT if stop_mode is None else stop_mode
        assert params.stop_str_match_mode.get_mode() == expected_mode
        expected = "helloEND" if include else "hello"
        assert "".join(result[1] for result in results) == expected
        assert [result[2]["id"] for result in results] == [1, 2]
        assert results[0][2]["input_usage"]["input_text_tokens"] == 3
        assert results[-1][3].get_finish_reason() == "stop"
        manager.shm_req_manager.async_get_req_obj_by_index.return_value.set_compiled_grammar.assert_not_called()
        manager._unregister_running_request.assert_awaited_once()

    asyncio.run(run())


def _req_status(reqs):
    return ReqStatus(123, None, reqs, 0)


async def _drain_generate(manager, sampling_params, multimodal_params, websocket=None, pd_event=None):
    async for _ in manager.generate(
        prompt="prompt",
        sampling_params=sampling_params,
        multimodal_params=multimodal_params,
        request=None,
        pd_upload_websocket=websocket,
        pd_event=pd_event,
    ):
        pass


@pytest.mark.parametrize("mode", [NodeRole.NORMAL, NodeRole.D])
def test_non_prefill_request_is_counted_before_early_failure_and_always_unregistered(
    mode,
):
    async def run():
        manager = _make_manager(mode)
        manager._encode.side_effect = RuntimeError("encode failed")

        with pytest.raises(RuntimeError, match="encode failed"):
            await _drain_generate(manager, _sampling_params(), _multimodal_params())

        manager._register_running_request.assert_awaited_once()
        manager._unregister_running_request.assert_awaited_once()

    asyncio.run(run())


def test_prefill_encode_failure_is_outside_running_request_count():
    async def run():
        manager = _make_manager(NodeRole.P)
        manager._encode.side_effect = RuntimeError("encode failed")

        with pytest.raises(RuntimeError, match="encode failed"):
            await _drain_generate(
                manager,
                _sampling_params(),
                _multimodal_params(),
                AsyncMock(),
                asyncio.Event(),
            )

        manager._register_running_request.assert_not_awaited()
        manager._unregister_running_request.assert_not_awaited()

    asyncio.run(run())


def test_prefill_full_cache_hit_never_enters_running_request_count():
    async def run():
        manager = _make_manager(NodeRole.P)
        websocket = AsyncMock()
        pd_event = asyncio.Event()
        pd_event.decode_node_info = SimpleNamespace(ready_kv_len=2)
        pd_event.set()

        with pytest.raises(PDPrefillNodeStopGenToken):
            await _drain_generate(manager, _sampling_params(), _multimodal_params(), websocket, pd_event)

        manager._register_running_request.assert_not_awaited()
        manager._unregister_running_request.assert_not_awaited()
        websocket.send.assert_awaited_once()

    asyncio.run(run())


def test_prefill_upload_preserves_int64_multimodal_prompt_ids():
    async def run():
        manager = _make_manager(NodeRole.P)
        multimodal_token_id = 2 ** 32
        manager._encode.return_value = [10, multimodal_token_id, 12]
        websocket = AsyncMock()
        pd_event = asyncio.Event()
        pd_event.decode_node_info = SimpleNamespace(ready_kv_len=2)
        pd_event.set()

        with pytest.raises(PDPrefillNodeStopGenToken):
            await _drain_generate(manager, _sampling_params(), _multimodal_params(), websocket, pd_event)

        obj_type, group_request_id, prompt_ids = pickle.loads(websocket.send.await_args.args[0])
        assert obj_type == ObjType.PD_UPLOAD_PREFILL_PROMPT_IDS
        assert group_request_id == 123
        assert prompt_ids.itemsize == 8
        assert prompt_ids.tolist() == [10, multimodal_token_id, 12]

    asyncio.run(run())


def test_prefill_waiting_for_decode_assignment_can_be_cancelled_without_touching_count():
    async def run():
        manager = _make_manager(NodeRole.P)
        websocket = AsyncMock()
        pd_event = asyncio.Event()
        task = asyncio.create_task(
            _drain_generate(manager, _sampling_params(), _multimodal_params(), websocket, pd_event)
        )

        for _ in range(100):
            if websocket.send.await_count:
                break
            await asyncio.sleep(0)
        assert websocket.send.await_count == 1
        assert not task.done()
        manager._register_running_request.assert_not_awaited()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        manager._register_running_request.assert_not_awaited()
        manager._unregister_running_request.assert_not_awaited()

    asyncio.run(run())


def test_prefill_is_counted_after_decode_assignment_and_unregistered_on_following_failure():
    async def run():
        manager = _make_manager(NodeRole.P)
        websocket = AsyncMock()
        pd_event = asyncio.Event()
        pd_event.decode_node_info = SimpleNamespace(ready_kv_len=1)
        pd_event.set()

        with pytest.raises(RuntimeError, match="alloc failed"):
            await _drain_generate(manager, _sampling_params(), _multimodal_params(), websocket, pd_event)

        manager._register_running_request.assert_awaited_once()
        manager._unregister_running_request.assert_awaited_once()

    asyncio.run(run())


@pytest.mark.parametrize("mode", [NodeRole.P, NodeRole.D])
def test_pd_node_returns_busy_when_shm_req_allocation_times_out(mode):
    async def run():
        manager = _make_manager(mode)
        manager.shm_req_manager = SimpleNamespace(
            async_alloc_req_indexes=AsyncMock(return_value=None),
            async_release_req_index=AsyncMock(),
        )

        websocket = AsyncMock() if mode == NodeRole.P else None
        pd_event = None
        if mode == NodeRole.P:
            pd_event = asyncio.Event()
            pd_event.decode_node_info = SimpleNamespace(ready_kv_len=1)
            pd_event.set()

        with patch("lightllm.server.httpserver.manager.time.monotonic", side_effect=[0, 21]):
            with pytest.raises(ServerBusyError, match=f"PD {mode.value} node is busy"):
                await _drain_generate(
                    manager,
                    _sampling_params(),
                    _multimodal_params(),
                    websocket,
                    pd_event,
                )

        manager.shm_req_manager.async_alloc_req_indexes.assert_awaited_once_with(1)
        manager.shm_req_manager.async_release_req_index.assert_not_awaited()
        manager._register_running_request.assert_awaited_once()
        manager._unregister_running_request.assert_awaited_once()

    asyncio.run(run())


@pytest.mark.parametrize("mode", [NodeRole.P, NodeRole.D])
def test_pd_node_returns_busy_while_first_token_request_waits_in_router(mode):
    async def run():
        manager = _make_manager(mode)
        req = SimpleNamespace(
            request_id=123,
            is_aborted=False,
            router_arrival_time=0,
            infer_start_time=0,
            sample_params=SimpleNamespace(
                high_priority_request=False,
                pd_node_resource_wait_timeout_seconds=20,
            ),
        )
        req_status = _req_status([req])
        req_status.event.set()
        sampling_params = _sampling_params()

        with patch("lightllm.server.httpserver.manager.time.monotonic", return_value=21):
            req.router_arrival_time = 1.0
            output_generator = manager._wait_to_token_package(
                start_time=0,
                prompt_ids=[],
                group_request_id=123,
                sampling_params=sampling_params,
                req_status=req_status,
                request=None,
            )
            with pytest.raises(ServerBusyError, match="request did not enter inference"):
                await output_generator.__anext__()

    asyncio.run(run())


def test_multinode_tp_slave_does_not_apply_router_wait_timeout():
    async def run():
        manager = _make_manager(NodeRole.D)
        manager.is_multinode_tp_slave = True
        req = SimpleNamespace(
            router_arrival_time=1.0,
            infer_start_time=0.0,
            sample_params=SimpleNamespace(pd_node_resource_wait_timeout_seconds=0),
        )
        req_status = _req_status([req])
        req_status.event.set()
        request = SimpleNamespace(is_disconnected=AsyncMock(return_value=True))

        with patch("lightllm.server.httpserver.manager.time.monotonic", return_value=2):
            # ReqStatus 只判断时间条件；是否允许 slave 执行超时策略由 manager 在调用处决定。
            assert req_status.has_timed_out_waiting_for_inference() is True
            output_generator = manager._wait_to_token_package(
                start_time=0,
                prompt_ids=[],
                group_request_id=123,
                sampling_params=_sampling_params(),
                req_status=req_status,
                request=request,
            )
            with pytest.raises(ClientDisconnected):
                await output_generator.__anext__()

    asyncio.run(run())


def test_httpserver_keeps_started_requests_and_requests_with_remaining_master_timeout():
    waiting_req = SimpleNamespace(
        router_arrival_time=1.0,
        infer_start_time=0.0,
        sample_params=SimpleNamespace(
            high_priority_request=False,
            pd_node_resource_wait_timeout_seconds=60,
        ),
    )
    started_req = SimpleNamespace(
        router_arrival_time=1.0,
        infer_start_time=2.0,
        sample_params=SimpleNamespace(high_priority_request=False),
    )
    req_status = _req_status([waiting_req, started_req])

    with patch("lightllm.server.httpserver.manager.time.monotonic", return_value=30):
        assert req_status.has_timed_out_waiting_for_inference() is False

        started_req.infer_start_time = 0.0
        assert req_status.has_timed_out_waiting_for_inference() is False


def test_pd_node_allocation_failure_does_not_hold_partial_slots():
    async def run():
        manager = _make_manager(NodeRole.D)
        manager.shm_req_manager = SimpleNamespace(
            async_alloc_req_indexes=AsyncMock(side_effect=RuntimeError("alloc failed")),
            async_release_req_index=AsyncMock(),
        )
        sampling_params = _sampling_params()
        sampling_params.n = 2

        with pytest.raises(RuntimeError, match="alloc failed"):
            await _drain_generate(manager, sampling_params, _multimodal_params())

        manager.shm_req_manager.async_alloc_req_indexes.assert_awaited_once_with(2)
        manager.shm_req_manager.async_release_req_index.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.parametrize("high_priority", [False, True])
def test_batch_allocation_retries_with_priority_backoff(high_priority):
    async def run():
        manager = _make_manager(NodeRole.D)
        manager.shm_req_manager.async_alloc_req_indexes = AsyncMock(side_effect=[None, None, [0, 1]])
        with patch("lightllm.server.httpserver.manager.asyncio.sleep", new_callable=AsyncMock) as sleep:
            assert await manager._alloc_shm_req_indexes(2, high_priority_request=high_priority) == [0, 1]
        factor = 0.2 if high_priority else 1
        assert [call.args[0] for call in sleep.await_args_list] == pytest.approx([0.1 * factor, 0.11 * factor])
        assert [call.args for call in manager.shm_req_manager.async_alloc_req_indexes.await_args_list] == [(2,)] * 3

    asyncio.run(run())


def test_batch_allocation_rejects_requests_larger_than_capacity():
    async def run():
        manager = _make_manager(NodeRole.NORMAL)
        with pytest.raises(ValueError, match="must not exceed running_max_req_size"):
            await manager._alloc_shm_req_indexes(3)
        manager.shm_req_manager.async_alloc_req_indexes.assert_not_awaited()

    asyncio.run(run())


def test_multinode_tp_slave_does_not_apply_allocation_wait_timeout():
    async def run():
        manager = _make_manager(NodeRole.D)
        manager.is_multinode_tp_slave = True
        manager.shm_req_manager.async_alloc_req_indexes = AsyncMock(side_effect=[None, [0, 1]])
        with patch("lightllm.server.httpserver.manager.asyncio.sleep", new_callable=AsyncMock):
            with patch("lightllm.server.httpserver.manager.time.monotonic") as monotonic:
                assert await manager._alloc_shm_req_indexes(2, pd_node_resource_wait_timeout_seconds=0) == [0, 1]
                monotonic.assert_not_called()

    asyncio.run(run())


def test_cancelled_batch_allocation_does_not_hold_slots():
    async def run():
        manager = _make_manager(NodeRole.NORMAL)
        manager.shm_req_manager = SimpleNamespace(
            async_alloc_req_indexes=AsyncMock(return_value=None),
            async_release_req_index=AsyncMock(),
        )
        task = asyncio.create_task(manager._alloc_shm_req_indexes(2))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        manager.shm_req_manager.async_alloc_req_indexes.assert_awaited_once_with(2)
        manager.shm_req_manager.async_release_req_index.assert_not_awaited()

    asyncio.run(run())


def test_running_request_helpers_are_atomic_and_refresh_timestamp_only_on_idle_to_running_transition():
    async def run():
        manager = HttpServerManager.__new__(HttpServerManager)
        manager._run_reqs_count_lock = asyncio.Lock()
        manager.run_reqs_count_mark = _ValueMark()
        manager.latest_success_infer_time_mark = _ValueMark(7)

        with patch("lightllm.server.httpserver.manager.time.time", side_effect=[1000, 2000]):
            await asyncio.gather(*(manager._register_running_request() for _ in range(100)))
            assert manager.run_reqs_count_mark.get_value() == 100
            assert manager.latest_success_infer_time_mark.get_value() == 1000

            await asyncio.gather(*(manager._unregister_running_request() for _ in range(100)))
            assert manager.run_reqs_count_mark.get_value() == 0

            await manager._register_running_request()
            assert manager.run_reqs_count_mark.get_value() == 1
            assert manager.latest_success_infer_time_mark.get_value() == 2000
            await manager._unregister_running_request()
            assert manager.run_reqs_count_mark.get_value() == 0

    asyncio.run(run())
