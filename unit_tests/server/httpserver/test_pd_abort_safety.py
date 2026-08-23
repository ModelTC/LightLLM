import asyncio
import copy
from types import SimpleNamespace

import pytest

from lightllm.server.core.objs import FinishStatus, SamplingParams
from lightllm.server.httpserver.manager import HttpServerManager
from lightllm.server.httpserver_for_pd_master.manager import HttpServerManagerForPDMaster
from lightllm.utils.error_utils import ClientDisconnected


def test_pd_worker_preserves_abort_received_during_request_registration():
    manager = object.__new__(HttpServerManager)
    manager.req_id_to_out_inf = {}
    manager._pd_registration_abort_flags = {}
    request_id = 417

    manager.begin_pd_request_registration(request_id)
    assert asyncio.run(manager.abort(request_id)) is True
    assert manager._pd_registration_abort_flags == {request_id: True}

    requests = [SimpleNamespace(is_aborted=False), SimpleNamespace(is_aborted=False)]
    request_status = SimpleNamespace(group_req_objs=SimpleNamespace(group_req_id=request_id, shm_req_objs=requests))
    manager._register_req_status(request_id, request_status)

    assert manager.req_id_to_out_inf[request_id] is request_status
    assert manager._pd_registration_abort_flags == {}
    assert all(request.is_aborted for request in requests)


def test_pd_worker_cancelled_registration_does_not_poison_reused_request_id():
    manager = object.__new__(HttpServerManager)
    manager.req_id_to_out_inf = {}
    manager._pd_registration_abort_flags = {}
    request_id = 418

    manager.begin_pd_request_registration(request_id)
    assert asyncio.run(manager.abort(request_id)) is True
    manager.cancel_pd_request_registration(request_id)

    request = SimpleNamespace(is_aborted=False)
    manager.begin_pd_request_registration(request_id)
    manager._register_req_status(
        request_id,
        SimpleNamespace(group_req_objs=SimpleNamespace(group_req_id=request_id, shm_req_objs=[request])),
    )

    assert manager._pd_registration_abort_flags == {}
    assert request.is_aborted is False


def test_pd_master_client_disconnect_aborts_nodes_and_releases_load(monkeypatch):
    monkeypatch.setattr(
        SamplingParams,
        "from_buffer_copy",
        classmethod(lambda cls, other: copy.copy(other)),
    )
    manager = object.__new__(HttpServerManagerForPDMaster)
    manager._split_max_new_tokens = lambda max_new_tokens: [max_new_tokens]
    manager.id_gen = SimpleNamespace(generate_id=lambda: 9001)
    manager.pd_manager = SimpleNamespace(selector=SimpleNamespace(record_prompt_cache_hit_rate=lambda _rate: None))
    p_node = SimpleNamespace(dispatched_prompt_chars=0, dispatched_req_num=0)
    d_node = SimpleNamespace()

    async def select_nodes(*_args):
        return p_node, d_node

    manager.select_p_d_node = select_nodes
    aborts = []
    removals = []

    async def abort(request_id, p_node=None, d_node=None):
        aborts.append((request_id, p_node, d_node))

    async def remove_req(request_id=None, group_request_id=None):
        request_id = request_id if request_id is not None else group_request_id
        removals.append(request_id)

    manager.abort = abort
    manager.remove_req = remove_req

    async def disconnected_stream(*_args):
        if False:
            yield None
        raise ClientDisconnected(group_request_id=9001, reason="test client cancelled")

    manager._wait_to_token_package = disconnected_stream
    sampling_params = SamplingParams()
    sampling_params.max_new_tokens = 1
    sampling_params.return_output_logprobs = True

    async def consume_disconnected_request():
        async for _ in manager._generate_one(
            "hello",
            sampling_params,
            SimpleNamespace(),
            SimpleNamespace(),
            1.0,
            8001,
        ):
            pass

    with pytest.raises(ClientDisconnected, match="test client cancelled"):
        asyncio.run(consume_disconnected_request())

    assert aborts == [(9001, p_node, d_node)]
    assert removals == [9001]
    assert p_node.dispatched_prompt_chars == 0
    assert p_node.dispatched_req_num == 0

    async def healthy_stream(*_args):
        yield 9001, "ok", {"prompt_tokens": 1, "prompt_cache_len": 0}, FinishStatus(FinishStatus.FINISHED_STOP)

    manager._wait_to_token_package = healthy_stream

    async def consume_healthy_request():
        return [
            item
            async for item in manager._generate_one(
                "hello",
                sampling_params,
                SimpleNamespace(),
                SimpleNamespace(),
                1.0,
                8002,
            )
        ]

    assert asyncio.run(consume_healthy_request())[0][1] == "ok"
    assert p_node.dispatched_prompt_chars == 0
    assert p_node.dispatched_req_num == 0
