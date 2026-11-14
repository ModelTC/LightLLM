import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lightllm.server.core.objs import SamplingParams
from lightllm.server.httpserver_for_pd_master.manager import HttpServerManagerForPDMaster, ReqStatus
from lightllm.server.multimodal_params import MultimodalParams
from lightllm.utils.error_utils import ClientDisconnected, ServerBusyError
from lightllm.utils.pd_timeout_utils import get_pd_handshake_timeout_seconds


@pytest.mark.parametrize(
    "resource_wait,master,prefill", [(-1, None, None), (0, 180, 195), (60, 180, 195), (200, 415, 430), (450, 915, 930)]
)
def test_handshake_encloses_both_resource_waits(resource_wait: int, master: int | None, prefill: int | None) -> None:
    assert get_pd_handshake_timeout_seconds(resource_wait_seconds=resource_wait) == master
    assert get_pd_handshake_timeout_seconds(resource_wait_seconds=resource_wait, prefill=True) == prefill


@pytest.mark.parametrize("resource_wait,expected", [(60, 180), (200, 415), (-1, None)])
def test_master_uses_the_current_segments_resource_budget(resource_wait: int, expected: int | None) -> None:
    async def run() -> None:
        manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
        manager.args = SimpleNamespace(pd_node_id=1)
        manager.req_id_to_out_inf = {}
        params = SamplingParams()
        params.group_request_id, params.max_new_tokens = 123, 4
        params.pd_node_resource_wait_timeout_seconds = resource_wait

        async def handshake(
            event: asyncio.Event, request: object, timeout: float | None, group_request_id: int, stage: str
        ) -> None:
            if stage == "prefill":
                assert timeout == 60
                event.prompt_ids = [1, 2]
            else:
                assert timeout == expected
                raise RuntimeError("decode budget checked")

        manager._wait_for_event_or_disconnect = handshake
        node = SimpleNamespace(websocket=SimpleNamespace(send_bytes=AsyncMock()))
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


@pytest.mark.parametrize("timeout", [None, 415])
def test_node_error_wakes_handshake_without_retry(timeout: float | None) -> None:
    async def run() -> None:
        manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
        status = ReqStatus(req_id=123, p_node=None, d_node=None)
        request = SimpleNamespace(is_disconnected=AsyncMock(return_value=False))
        task = asyncio.create_task(
            coro=manager._wait_for_event_or_disconnect(
                event=status.up_status_event, request=request, timeout=timeout, group_request_id=123, stage="decode"
            )
        )
        await status.set_error(error_info="allocation failed")
        await asyncio.wait_for(fut=task, timeout=1)
        with pytest.raises(expected_exception=RuntimeError, match="allocation failed"):
            status.raise_if_error()

    asyncio.run(main=run())


def test_unbounded_handshake_still_checks_disconnect_and_cancellation() -> None:
    async def run() -> None:
        manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
        request = SimpleNamespace(is_disconnected=AsyncMock(return_value=True))
        with pytest.raises(expected_exception=ClientDisconnected):
            await manager._wait_for_event_or_disconnect(
                event=asyncio.Event(), request=request, timeout=None, group_request_id=123, stage="decode"
            )
        request.is_disconnected.return_value = False
        task = asyncio.create_task(
            coro=manager._wait_for_event_or_disconnect(
                event=asyncio.Event(), request=request, timeout=None, group_request_id=123, stage="decode"
            )
        )
        await asyncio.sleep(delay=0)
        task.cancel()
        with pytest.raises(expected_exception=asyncio.CancelledError):
            await task

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
