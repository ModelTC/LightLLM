import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lightllm.server.core.objs import SamplingParams
from lightllm.server.httpserver_for_pd_master.manager import HttpServerManagerForPDMaster, ReqStatus
from lightllm.server.multimodal_params import MultimodalParams
from lightllm.server.pd_io_struct import PD_Client_Obj
from lightllm.utils.error_utils import ServerBusyError


@pytest.mark.parametrize("resource_wait,expected", [(0, 180), (60, 180), (200, 415), (450, 915), (-1, 180)])
def test_master_uses_the_current_segments_resource_budget(resource_wait: int, expected: int) -> None:
    async def run() -> None:
        manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
        manager.args = SimpleNamespace(pd_node_id=1)
        manager.req_id_to_out_inf = {}
        params = SamplingParams()
        params.group_request_id, params.max_new_tokens = 123, 4
        params.pd_node_resource_wait_timeout_seconds = resource_wait

        async def handshake(
            event: asyncio.Event, request: object, timeout: float, group_request_id: int, stage: str
        ) -> None:
            if stage == "prefill":
                assert timeout == 60
                event.prompt_ids = [1, 2]
            else:
                assert timeout == expected
                raise RuntimeError("decode budget checked")

        manager._wait_for_event_or_disconnect = handshake
        node = PD_Client_Obj(1, "node:8000", "prefill", {})
        node.sub_connections["test"] = SimpleNamespace(send_bytes=AsyncMock())
        stream = manager.fetch_pd_stream(
            p_node=node,
            d_node=node,
            prompt="prompt",
            sampling_params=params,
            multimodal_params=MultimodalParams(),
            request=SimpleNamespace(is_disconnected=AsyncMock(return_value=False)),
        )
        try:
            with pytest.raises(expected_exception=RuntimeError, match="decode budget checked"):
                await anext(stream)
        finally:
            await stream.aclose()

    asyncio.run(main=run())


def test_node_error_wakes_handshake_without_retry() -> None:
    async def run() -> None:
        manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
        status = ReqStatus(req_id=123, p_node=None, d_node=None, p_websocket=object(), d_websocket=object())
        request = SimpleNamespace(is_disconnected=AsyncMock(return_value=False))
        task = asyncio.create_task(
            coro=manager._wait_for_event_or_disconnect(
                event=status.up_status_event, request=request, timeout=415, group_request_id=123, stage="decode"
            )
        )
        await status.set_error(error_info="allocation failed")
        await asyncio.wait_for(fut=task, timeout=1)
        with pytest.raises(expected_exception=RuntimeError, match="allocation failed"):
            status.raise_if_error()

    asyncio.run(main=run())


def test_finite_handshake_still_expires() -> None:
    async def run() -> None:
        manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
        with pytest.raises(expected_exception=ServerBusyError):
            await manager._wait_for_event_or_disconnect(
                event=asyncio.Event(),
                request=SimpleNamespace(is_disconnected=AsyncMock(return_value=False)),
                timeout=0.001,
                group_request_id=123,
                stage="decode",
            )

    asyncio.run(main=run())
