import asyncio
import pickle
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lightllm.server.core.objs import SamplingParams
from lightllm.server.httpserver.manager import HttpServerManager, ReqStatus
from lightllm.server.pd_io_struct import NodeRole, ObjType
from lightllm.utils.error_utils import PDPrefillNodeStopGenToken, ServerBusyError


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
        enable_pd_node_self_request_limit=False,
        run_mode=mode.value,
        running_max_req_size=2,
    )
    manager.pd_mode = mode
    manager.is_multinode_tp_slave = False
    manager.pd_node_request_limit_enabled = False
    manager.pd_node_shm_req_alloc_timeout_seconds = 20
    manager.pd_node_router_wait_timeout_seconds = 20
    manager.alloc_req_id = MagicMock(return_value=123)
    manager.is_multinode_tp_master = False
    manager.rl_controller = None
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
    manager.shm_req_manager = SimpleNamespace(async_alloc_req_index=AsyncMock(side_effect=RuntimeError("alloc failed")))
    return manager


def _sampling_params():
    sampling_params = SamplingParams()
    sampling_params.group_request_id = 123
    sampling_params.n = 1
    sampling_params.max_new_tokens = 1
    return sampling_params


def _multimodal_params():
    return SimpleNamespace(audios=[], images=[], verify_and_preload=AsyncMock())


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
        manager.pd_node_request_limit_enabled = True
        manager.shm_req_manager = SimpleNamespace(
            async_alloc_req_index=AsyncMock(return_value=None),
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

        manager.shm_req_manager.async_alloc_req_index.assert_awaited_once()
        manager.shm_req_manager.async_release_req_index.assert_not_awaited()
        manager._register_running_request.assert_awaited_once()
        manager._unregister_running_request.assert_awaited_once()

    asyncio.run(run())


@pytest.mark.parametrize("mode", [NodeRole.P, NodeRole.D])
def test_pd_node_returns_busy_while_first_token_request_waits_in_router(mode):
    async def run():
        manager = _make_manager(mode)
        manager.pd_node_request_limit_enabled = True
        req = SimpleNamespace(
            request_id=123,
            is_aborted=False,
            router_arrival_time=0,
            infer_start_time=0,
            sample_params=SimpleNamespace(pd_high_priority_request=False),
        )
        req_status = ReqStatus(123, None, [req], 0)
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


def test_httpserver_keeps_started_and_high_priority_request_groups():
    waiting_req = SimpleNamespace(
        router_arrival_time=1.0,
        infer_start_time=0.0,
        sample_params=SimpleNamespace(
            pd_high_priority_request=False,
            pd_high_priority_request_time_out_seconds=60,
        ),
    )
    started_req = SimpleNamespace(
        router_arrival_time=1.0,
        infer_start_time=2.0,
        sample_params=SimpleNamespace(pd_high_priority_request=False),
    )
    req_status = ReqStatus(123, None, [waiting_req, started_req], 0)

    with patch("lightllm.server.httpserver.manager.time.monotonic", return_value=30):
        assert req_status.has_timed_out_waiting_for_inference(20) is False

        started_req.infer_start_time = 0.0
        waiting_req.sample_params.pd_high_priority_request = True
        assert req_status.has_timed_out_waiting_for_inference(20) is False


def test_pd_node_self_request_limit_releases_partially_allocated_shm_reqs():
    async def run():
        manager = _make_manager(NodeRole.D)
        manager.shm_req_manager = SimpleNamespace(
            async_alloc_req_index=AsyncMock(side_effect=[7, RuntimeError("alloc failed")]),
            async_release_req_index=AsyncMock(),
        )
        sampling_params = _sampling_params()
        sampling_params.n = 2

        with pytest.raises(RuntimeError, match="alloc failed"):
            await _drain_generate(manager, sampling_params, _multimodal_params())

        manager.shm_req_manager.async_release_req_index.assert_awaited_once_with(7)

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
