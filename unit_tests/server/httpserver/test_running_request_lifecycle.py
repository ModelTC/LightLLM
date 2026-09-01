import asyncio
import pickle
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lightllm.server.core.objs import SamplingParams
from lightllm.server.httpserver.decode_admission import DecodeAdmissionController
from lightllm.server.httpserver.manager import HttpServerManager
from lightllm.server.pd_io_struct import NodeRole, ObjType
from lightllm.utils.error_utils import PDPrefillNodeStopGenToken


class _ValueMark:
    def __init__(self, value=0):
        self.value = value

    def get_value(self):
        return self.value

    def set_value(self, value):
        self.value = value


def _make_manager(mode: NodeRole):
    manager = HttpServerManager.__new__(HttpServerManager)
    manager.pd_mode = mode
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
    manager.shm_req_manager = SimpleNamespace(
        async_alloc_req_indexes=AsyncMock(side_effect=RuntimeError("alloc failed"))
    )
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


def test_decode_admission_lease_lives_until_request_resources_are_recycled():
    async def run():
        manager = _make_manager(NodeRole.D)
        manager.decode_admission_controller = DecodeAdmissionController(
            capacity=1, max_queued_slots=1, timeout_seconds=1
        )
        manager.args = SimpleNamespace(chunked_prefill_size=None)
        manager.tokenizer = object()
        manager.enable_multimodal = False
        manager.transfer_to_next_module_or_node = AsyncMock()
        manager._count_multimodal_tokens = MagicMock(return_value=(0, 0))

        req_obj = MagicMock(request_id=123, index_in_shm_mem=7)
        manager.shm_req_manager = SimpleNamespace(
            async_alloc_req_indexes=AsyncMock(return_value=[7]),
            async_get_req_obj_by_index=AsyncMock(return_value=req_obj),
            async_put_back_req_obj=AsyncMock(),
            async_release_req_index=AsyncMock(),
        )

        async def wait_to_token_package(*_args, **_kwargs):
            yield 123, "token", {}, MagicMock()

        manager._wait_to_token_package = wait_to_token_package
        generator = manager.generate(
            prompt=[10, 11, 12],
            sampling_params=_sampling_params(),
            multimodal_params=_multimodal_params(),
            request=None,
        )

        assert (await generator.__anext__())[1] == "token"
        req_status = manager.req_id_to_out_inf[123]
        assert manager.decode_admission_controller.active_slots == 1

        await generator.aclose()
        assert manager.decode_admission_controller.active_slots == 1

        req_status.release_decode_admission()
        assert manager.decode_admission_controller.active_slots == 0

    asyncio.run(run())


def test_multinode_tp_slave_does_not_register_as_independent_pd_node():
    async def run():
        manager = HttpServerManager.__new__(HttpServerManager)
        manager.recycle_resource_loop = AsyncMock()
        manager.loop_for_request = AsyncMock()
        manager.is_multinode_tp_slave = True
        manager.pd_mode = NodeRole.D
        manager.zmq_recv_socket = SimpleNamespace(recv_pyobj=AsyncMock(side_effect=asyncio.CancelledError))

        with patch("lightllm.server.httpserver.pd_loop.pd_handle_loop") as pd_handle_loop:
            with pytest.raises(asyncio.CancelledError):
                await manager.handle_loop()

        pd_handle_loop.assert_not_called()

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
