import asyncio
from types import SimpleNamespace

from lightllm.server.httpserver.manager import HttpServerManager
from lightllm.server.httpserver.pd_loop import _abort_requests_from_disconnected_master


def _manager_with_requests(*request_ids):
    manager = HttpServerManager.__new__(HttpServerManager)
    manager.req_id_to_out_inf = {}
    requests = {}
    for request_id in request_ids:
        shm_reqs = [SimpleNamespace(is_aborted=False) for _ in range(2)]
        requests[request_id] = shm_reqs
        manager.req_id_to_out_inf[request_id] = SimpleNamespace(
            group_req_objs=SimpleNamespace(group_req_id=request_id, shm_req_objs=shm_reqs)
        )
    return manager, requests


def test_disconnect_aborts_only_requests_from_that_master():
    async def run():
        manager, requests = _manager_with_requests(11, 22)
        pending = await _abort_requests_from_disconnected_master(manager, [11])
        assert pending == []
        assert all(req.is_aborted for req in requests[11])
        assert not any(req.is_aborted for req in requests[22])

    asyncio.run(run())


def test_unregistered_request_is_aborted_once_it_is_registered():
    async def run():
        manager, requests = _manager_with_requests(22)
        pending = await _abort_requests_from_disconnected_master(manager, [11, 22])
        assert pending == [11]
        assert all(req.is_aborted for req in requests[22])

        requests[11] = [SimpleNamespace(is_aborted=False)]
        manager.req_id_to_out_inf[11] = SimpleNamespace(
            group_req_objs=SimpleNamespace(group_req_id=11, shm_req_objs=requests[11])
        )
        pending = await _abort_requests_from_disconnected_master(manager, pending)
        assert pending == []
        assert all(req.is_aborted for req in requests[11])

    asyncio.run(run())
