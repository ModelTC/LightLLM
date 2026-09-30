import asyncio
import pickle
import queue
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from lightllm.server.core.objs import FinishStatus, StartArgs
from lightllm.server.httpserver.manager import HttpServerManager
from lightllm.server.httpserver_for_pd_master.manager import HttpServerManagerForPDMaster, PDManager, ReqStatus
from lightllm.server.pd_io_struct import ObjType, PD_Client_Obj
from lightllm.utils.error_utils import ServerBusyError


def node(node_id=1):
    return PD_Client_Obj(node_id=node_id, client_ip_port="10.0.0.1:8000", mode="prefill", start_args={})


def test_abort_before_request_registration_is_applied_and_released():
    async def run():
        manager = HttpServerManager.__new__(HttpServerManager)
        manager.req_id_to_out_inf = {}
        manager._pd_registration_abort_flags = {}
        manager.begin_pd_request_registration(123)
        assert await manager.abort(123)
        reqs = [SimpleNamespace(is_aborted=False) for _ in range(2)]
        status = SimpleNamespace(group_req_objs=SimpleNamespace(shm_req_objs=reqs))
        manager._register_req_status(123, status)
        assert all(req.is_aborted for req in reqs)
        assert not manager._pd_registration_abort_flags
        manager.begin_pd_request_registration(124)
        manager.cancel_pd_request_registration(124)
        assert not manager._pd_registration_abort_flags
        assert not await manager.abort(125)

    asyncio.run(run())


def test_old_disconnect_does_not_remove_new_registration():
    args = StartArgs()
    manager = PDManager(args)
    info = dict(node_id=1, client_ip_port="10.0.0.1:8000", mode="prefill", start_args=vars(args))
    old = manager.register_pd(info, Mock())
    new = manager.register_pd({**info, "node_id": 2}, Mock())
    manager.remove_pd(old)
    assert old.websocket is None
    assert manager.url_to_pd_nodes[new.client_ip_port] is new
    assert manager.prefill_nodes == [new]
    manager.remove_pd(new)
    assert not manager.prefill_nodes


def test_disconnect_wakes_all_request_stages_only_for_affected_connection():
    async def run():
        old, new = node(1), node(2)
        affected, unaffected = ReqStatus(123, old, None), ReqStatus(124, new, None)
        manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
        manager.pd_manager = Mock()
        manager.req_id_to_out_inf = {123: affected, 124: unaffected}
        await manager.remove_pd(old)
        assert affected.event.is_set() and affected.up_status_event.is_set()
        assert affected.prefill_prompt_ids_event.is_set()
        with pytest.raises(RuntimeError, match="disconnected"):
            affected.raise_if_error()
        assert unaffected.error_info is None

    asyncio.run(run())


def test_cancelled_control_send_finishes_frame_before_next_send():
    async def run():
        client = node()
        entered, release = asyncio.Event(), asyncio.Event()
        frames = []

        async def send(payload):
            frames.append(payload)
            if payload == b"first":
                entered.set()
                await release.wait()

        client.websocket = SimpleNamespace(send_bytes=send)
        first = asyncio.create_task(client.send_control_message(b"first"))
        await entered.wait()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(client.send_control_message(b"second"))
        await asyncio.sleep(0)
        assert frames == [b"first"]
        release.set()
        await asyncio.wait_for(second, 1)
        assert frames == [b"first", b"second"]
        client.websocket = None
        with pytest.raises(ConnectionError):
            await client.send_control_message(b"third")

    asyncio.run(run())


def test_abort_notifications_survive_cancelled_http_request():
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        frames = []

        async def send(payload):
            entered.set()
            await release.wait()
            frames.append(pickle.loads(payload))

        client = node()
        client.send_control_message = send
        manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
        manager.req_id_to_out_inf = {}
        manager._abort_notify_tasks = set()
        abort = asyncio.create_task(manager.abort(123, p_node=client))
        await entered.wait()
        abort.cancel()
        with pytest.raises(asyncio.CancelledError):
            await abort
        pending = list(manager._abort_notify_tasks)
        assert pending
        release.set()
        await asyncio.wait_for(asyncio.gather(*pending), 1)
        assert frames == [(ObjType.ABORT, 123)]
        assert not manager._abort_notify_tasks

    asyncio.run(run())


def test_prefill_wait_preserves_terminal_error_token():
    async def run():
        manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
        status = ReqStatus(123, None, None)
        token = (123, "", {}, FinishStatus(FinishStatus.FINISHED_ERROR))
        await status.put_tokens_to_front([token])
        manager.req_id_to_out_inf = {123: status}
        result = await asyncio.wait_for(
            manager._wait_for_prefill_token_if_needed(
                status, SimpleNamespace(is_disconnected=AsyncMock(return_value=False)), 123, True, 100
            ),
            1,
        )
        assert result == 100
        assert await status.pop_all_tokens() == [token]

    asyncio.run(run())


def test_missing_pd_nodes_returns_service_unavailable():
    manager = PDManager(StartArgs())
    with pytest.raises(ServerBusyError) as exc:
        manager.select_p_d_node("prompt", None, None)
    assert exc.value.status_code == 503


@pytest.mark.parametrize("terminal_status", [FinishStatus.FINISHED_ERROR, FinishStatus.FINISHED_ABORTED])
def test_terminal_decode_first_token_is_not_filtered_as_duplicate(terminal_status):
    async def run():
        from lightllm.server.core.objs import SamplingParams

        manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
        manager.args = StartArgs(pd_node_id=1)
        manager.metric_client = Mock()
        manager.req_id_to_out_inf = {}
        manager._wait_for_prefill_token_if_needed = AsyncMock(return_value=0)
        nodes = [node(1), node(2)]
        for client in nodes:
            client.websocket = SimpleNamespace(send_bytes=AsyncMock())

        async def wait(event, request, **kwargs):
            if kwargs["stage"] == "prefill":
                event.prompt_ids = [1, 2, 3]
            else:
                event.upkv_status = SimpleNamespace(pd_kv_trans_params=pickle.dumps(SimpleNamespace(ready_kv_len=2)))
                status = manager.req_id_to_out_inf[0]
                status.out_token_info_list = [
                    (0, "first", {"count_output_tokens": 1, "node_mode": "prefill"}, FinishStatus()),
                    (0, "", {"count_output_tokens": 1, "node_mode": "decode"}, FinishStatus(terminal_status)),
                ]
                status.event.set()

        manager._wait_for_event_or_disconnect = wait
        params = SamplingParams()
        params.group_request_id = 0
        params.max_new_tokens = 5
        stream = manager.fetch_pd_stream(
            *nodes, "prompt", params, SimpleNamespace(), SimpleNamespace(is_disconnected=AsyncMock(return_value=False))
        )
        assert (await asyncio.wait_for(stream.__anext__(), 1))[1] == "first"
        assert (await asyncio.wait_for(stream.__anext__(), 1))[3].status == terminal_status
        await stream.aclose()

    asyncio.run(run())


def test_kv_group_is_registered_before_following_abort_is_dispatched():
    from lightllm.server.pd_io_struct import PDAbortReq, PDChunckedTransTaskGroup
    from lightllm.server.router.model_infer.mode_backend.pd.decode_node_impl.decode_trans_process import (
        _DecodeTransModule,
    )

    class StopLoop(BaseException):
        pass

    task = SimpleNamespace(
        request_id=123,
        pd_master_node_id=1,
        start_kv_index=0,
        dst_page_index=None,
        need_transfer_page=lambda: True,
        get_key=lambda: "page",
    )
    group, abort = PDChunckedTransTaskGroup([task]), PDAbortReq(request_id=123, device_id=0)
    module = _DecodeTransModule.__new__(_DecodeTransModule)
    module.task_in_queue = SimpleNamespace(get=Mock(side_effect=[group, abort, StopLoop()]))
    module.recv_task_group_queue = queue.Queue()
    module.waiting_dict_lock = threading.Lock()
    module.waiting_dict = {}
    module.failed_queue = queue.Queue()
    module.success_queue = queue.Queue()
    module.up_status_in_queue = queue.Queue()
    module.args = StartArgs(pd_node_id=2)
    module.transporter = SimpleNamespace(agent_name="test", agent_metadata=b"", num_pages=1, local_page_mem_desc=b"")
    with pytest.raises(StopLoop):
        module.recv_task_loop()
    assert module.recv_task_group_queue.get_nowait() is group
    assert module.recv_task_group_queue.get_nowait() is abort
    assert not module.waiting_dict and module.failed_queue.empty()

    module.recv_task_group_queue = SimpleNamespace(get=Mock(side_effect=[group, abort, StopLoop()]))
    with pytest.raises(StopLoop):
        module.dispatch_task_loop()
    assert not module.waiting_dict
    assert module.failed_queue.get_nowait() is task
    assert task.error_info == "aborted req"
    assert module.up_status_in_queue.get_nowait().group_request_id == 123
