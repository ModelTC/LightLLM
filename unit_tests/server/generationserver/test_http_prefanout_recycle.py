import asyncio
from types import SimpleNamespace

from lightllm.server.core.objs import KV_CONSUMER_GENERATION, Req
from lightllm.server.generationserver.action.protocol import PrefixContextIdentity
from lightllm.server.httpserver.manager import HttpServerManager, ReqStatus


def _req(*, consumer_mask: int = 0) -> Req:
    req = Req()
    req.request_id = 17
    req.index_in_shm_mem = 3
    req.ref_count = 1
    req.kv_consumer_mask = consumer_mask
    req.is_aborted = False
    req.can_released_mark = False
    return req


def _manager_with(req_status: ReqStatus) -> HttpServerManager:
    manager = HttpServerManager.__new__(HttpServerManager)
    manager.req_id_to_out_inf = {
        req_status.group_req_objs.group_req_id: req_status,
    }
    return manager


def test_abort_marks_unforwarded_request_for_normal_recycle():
    req = _req(consumer_mask=KV_CONSUMER_GENERATION)
    req_status = ReqStatus(17, None, [req], 0.0)
    manager = _manager_with(req_status)

    assert asyncio.run(manager.abort(17))
    assert req.is_aborted
    assert req.can_released_mark

    # The local finish mark does not bypass action ownership.  Recycling only
    # becomes legal after GenerationServer releases its Prefix KV lease.
    assert not req_status.can_release()
    req.kv_consumer_mask = 0
    assert req_status.can_release()


def test_abort_does_not_locally_finish_forwarded_request():
    req = _req()
    req_status = ReqStatus(17, None, [req], 0.0)
    req_status.request_forwarded = True
    manager = _manager_with(req_status)

    assert asyncio.run(manager.abort(17))
    assert req.is_aborted
    assert not req.can_released_mark
    assert not req_status.can_release()


def test_failed_persistent_response_delivery_closes_undelivered_context():
    async def scenario():
        req = _req(consumer_mask=KV_CONSUMER_GENERATION)
        req_status = ReqStatus(17, None, [req], 0.0)
        req_status.request_forwarded = True
        identity = PrefixContextIdentity("robot-7", 1, "test-epoch")

        async def action_result():
            return SimpleNamespace(
                ok=True,
                committed_new_context=True,
                prefix_context=identity,
                request_id="request-17",
            )

        class ActionClient:
            def __init__(self):
                self.closed = []

            async def close_context(self, prefix_context, *, request_id):
                self.closed.append((prefix_context, request_id))

        manager = _manager_with(req_status)
        manager.output_client = ActionClient()
        req_status.output_modality = "action"
        req_status.output_result_task = asyncio.create_task(action_result())
        await req_status.output_result_task

        assert await manager.settle_action_response_delivery(17, delivered=False)
        assert manager.output_client.closed == [(identity, "request-17")]
        assert req.is_aborted
        assert not req_status.action_result_delivered

    asyncio.run(scenario())
