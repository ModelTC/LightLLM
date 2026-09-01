from __future__ import annotations

import asyncio
import pickle
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from lightllm.server.core.objs import FinishStatus, SamplingParams
from lightllm.server.httpserver.manager import HttpServerManager
from lightllm.server.httpserver_for_pd_master.manager import HttpServerManagerForPDMaster
from lightllm.server.multimodal_params import MultimodalParams
from lightllm.server.pd_io_struct import ObjType


def _manager() -> HttpServerManagerForPDMaster:
    manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
    manager.pd_manager = MagicMock()
    manager.args = SimpleNamespace(
        disable_pd_node_decode_admission=False,
        disable_pd_master_decode_capacity_limit=False,
        pd_node_id=7,
    )
    manager.id_gen = MagicMock()
    manager.id_gen.generate_id.return_value = 800
    manager.metric_client = MagicMock()
    manager._log_req_header = AsyncMock()
    manager.tokens = MagicMock(return_value=2)
    manager.req_id_to_out_inf = {}
    manager.decode_reservation_statuses = {}
    return manager


def test_pd_master_reserves_n_then_sends_independent_choices_to_one_decode_node():
    async def run():
        manager = _manager()
        manager.id_gen.generate_id.side_effect = [800, 808, 816, 824]
        sampling_params = SamplingParams()
        sampling_params.n = 3
        sampling_params.best_of = 3
        sampling_params.max_new_tokens = 4

        multimodal_params = MagicMock()
        multimodal_params.verify_and_preload = AsyncMock()
        request = MagicMock()
        p_node = MagicMock(dispatched_prompt_chars=0, dispatched_req_num=0)
        d_node = MagicMock()
        manager.select_p_d_node = AsyncMock(return_value=(p_node, d_node))
        manager._split_max_new_tokens = MagicMock()
        manager.remove_req = AsyncMock()
        manager.reserve_decode_slots = AsyncMock()
        captured_params = []
        captured_nodes = []

        async def wait_to_token_package(
            selected_p_node,
            selected_d_node,
            start_time,
            prompt,
            node_sampling_params,
            child_multimodal_params,
            child_request,
            decode_reservation=None,
        ):
            captured_params.append(node_sampling_params)
            captured_nodes.append((selected_p_node, selected_d_node))
            await decode_reservation.wait(node_sampling_params.group_request_id)
            yield (
                node_sampling_params.group_request_id,
                f"choice-{node_sampling_params.group_request_id}",
                {"prompt_tokens": 2},
                FinishStatus(FinishStatus.FINISHED_STOP),
            )

        manager._wait_to_token_package = wait_to_token_package

        with patch.object(HttpServerManager, "_check_and_repair_length", new=AsyncMock()):
            results = []
            async for result in manager._generate("prompt", sampling_params, multimodal_params, request):
                results.append(result)

        assert {result[0] for result in results} == {800, 801, 802}
        assert {result[1] for result in results} == {"choice-808", "choice-816", "choice-824"}
        assert len(captured_params) == 3
        assert all(params.n == 1 for params in captured_params)
        assert all(params.best_of == 1 for params in captured_params)
        assert {params.group_request_id for params in captured_params} == {808, 816, 824}
        assert captured_nodes == [(p_node, d_node)] * 3
        manager.select_p_d_node.assert_awaited_once()
        manager._split_max_new_tokens.assert_not_called()
        manager.reserve_decode_slots.assert_awaited_once()
        reserve_args = manager.reserve_decode_slots.await_args.args
        assert reserve_args[0] is d_node
        assert reserve_args[1] == 800
        assert set(reserve_args[2]) == {808, 816, 824}
        assert reserve_args[3] is request
        multimodal_params.verify_and_preload.assert_awaited_once_with(request)
        manager._log_req_header.assert_awaited_once_with(request, 800)
        assert manager.metric_client.counter_inc.call_args_list == [
            call("lightllm_request_count"),
            call("lightllm_request_success"),
        ]
        assert p_node.dispatched_prompt_chars == 0
        assert p_node.dispatched_req_num == 0

    asyncio.run(asyncio.wait_for(run(), timeout=2))


def test_pd_master_waits_for_prefill_and_atomic_reservation_before_decode_request():
    async def run():
        manager = _manager()
        sampling_params = SamplingParams()
        sampling_params.group_request_id = 808
        sampling_params.max_new_tokens = 4
        p_node = SimpleNamespace(websocket=SimpleNamespace(send_bytes=AsyncMock()))
        d_node = SimpleNamespace(websocket=SimpleNamespace(send_bytes=AsyncMock()))
        request = SimpleNamespace(is_disconnected=AsyncMock(return_value=False))
        reservation_entered = asyncio.Event()
        reservation_release = asyncio.Event()

        async def wait_for_reservation(group_request_id):
            assert group_request_id == 808
            reservation_entered.set()
            await reservation_release.wait()

        decode_reservation = SimpleNamespace(wait=wait_for_reservation)
        generator = manager.fetch_pd_stream(
            p_node,
            d_node,
            "prompt",
            sampling_params,
            MultimodalParams(),
            request,
            decode_reservation=decode_reservation,
        )
        next_result = asyncio.create_task(generator.__anext__())
        try:
            while 808 not in manager.req_id_to_out_inf:
                await asyncio.sleep(0)
            req_status = manager.req_id_to_out_inf[808]
            req_status.prefill_prompt_ids_event.prompt_ids = [1, 2, 3]
            req_status.prefill_prompt_ids_event.set()

            await asyncio.wait_for(reservation_entered.wait(), timeout=1)
            d_node.websocket.send_bytes.assert_not_awaited()

            reservation_release.set()
            while d_node.websocket.send_bytes.await_count == 0:
                await asyncio.sleep(0)
            decode_message = pickle.loads(d_node.websocket.send_bytes.await_args.args[0])
            assert decode_message[0] == ObjType.REQ
            assert list(decode_message[1][0]) == [1, 2, 3]
        finally:
            next_result.cancel()
            await asyncio.gather(next_result, return_exceptions=True)

    asyncio.run(asyncio.wait_for(run(), timeout=2))


def test_pd_master_n_one_uses_the_same_choice_merge_path():
    async def run():
        manager = _manager()
        sampling_params = SamplingParams()
        sampling_params.n = 1
        sampling_params.best_of = 1
        sampling_params.max_new_tokens = 4

        multimodal_params = MagicMock()
        multimodal_params.verify_and_preload = AsyncMock()
        request = MagicMock()
        p_node = MagicMock()
        d_node = MagicMock()
        manager.select_p_d_node = AsyncMock(return_value=(p_node, d_node))
        merge_choice_generators = manager._merge_choice_generators
        manager._merge_choice_generators = MagicMock(side_effect=merge_choice_generators)

        async def generate_one(
            prompt,
            choice_sampling_params,
            child_multimodal_params,
            child_request,
            start_time,
            origin_request_id,
            **_kwargs,
        ):
            yield (
                origin_request_id,
                "choice-0",
                {"prompt_tokens": 2},
                FinishStatus(FinishStatus.FINISHED_STOP),
            )

        manager._generate_one = generate_one

        with patch.object(HttpServerManager, "_check_and_repair_length", new=AsyncMock()):
            results = []
            async for result in manager._generate("prompt", sampling_params, multimodal_params, request):
                results.append(result)

        assert [result[0] for result in results] == [800]
        manager._merge_choice_generators.assert_called_once()
        assert len(manager._merge_choice_generators.call_args.args[0]) == 1

    asyncio.run(asyncio.wait_for(run(), timeout=2))


def test_pd_master_multi_choice_failure_closes_other_generators():
    async def run():
        manager = _manager()
        sibling_started = asyncio.Event()
        sibling_closed = asyncio.Event()

        async def sibling():
            try:
                sibling_started.set()
                while True:
                    await asyncio.sleep(1)
                    yield None
            finally:
                sibling_closed.set()

        async def failing():
            await sibling_started.wait()
            raise RuntimeError("choice failed")
            yield None

        with pytest.raises(RuntimeError, match="choice failed"):
            async for _ in manager._merge_choice_generators([sibling(), failing()]):
                pass

        assert sibling_closed.is_set()

    asyncio.run(asyncio.wait_for(run(), timeout=2))


def test_pd_master_multi_choice_cancellation_is_not_treated_as_success():
    async def run():
        manager = _manager()

        async def cancelled():
            raise asyncio.CancelledError("choice cancelled")
            yield None

        with pytest.raises(asyncio.CancelledError, match="choice cancelled"):
            async for _ in manager._merge_choice_generators([cancelled()]):
                pass

    asyncio.run(asyncio.wait_for(run(), timeout=2))


def test_pd_master_closing_merged_stream_closes_choice_generators():
    async def run():
        manager = _manager()
        choice_closed = asyncio.Event()

        async def choice():
            try:
                yield "first"
                await asyncio.sleep(10)
            finally:
                choice_closed.set()

        merged = manager._merge_choice_generators([choice()])
        assert await merged.__anext__() == "first"
        await merged.aclose()

        assert choice_closed.is_set()

    asyncio.run(asyncio.wait_for(run(), timeout=2))


def test_pd_master_releases_prefill_load_when_generation_fails():
    async def run():
        manager = _manager()
        manager._split_max_new_tokens = MagicMock(return_value=[4])
        manager.id_gen.generate_id.return_value = 808
        manager.remove_req = AsyncMock()
        manager.abort = AsyncMock()
        p_node = MagicMock(dispatched_prompt_chars=0, dispatched_req_num=0)
        d_node = MagicMock()

        async def failing_wait_to_token_package(*_args, **_kwargs):
            raise RuntimeError("generation failed")
            yield None

        manager._wait_to_token_package = failing_wait_to_token_package

        with pytest.raises(RuntimeError, match="generation failed"):
            # 空 prompt 的字符负载为 0，但已派发请求数仍必须在异常路径释放。
            async for _ in manager._generate_one(
                "",
                SamplingParams(),
                MagicMock(),
                MagicMock(),
                0,
                800,
                p_node,
                d_node,
            ):
                pass

        assert p_node.dispatched_prompt_chars == 0
        assert p_node.dispatched_req_num == 0

    asyncio.run(asyncio.wait_for(run(), timeout=2))


def test_pd_master_accounts_each_split_prefill_on_the_same_node():
    async def run():
        manager = _manager()
        manager._split_max_new_tokens = MagicMock(return_value=[1, 1])
        manager.id_gen.generate_id.side_effect = [808, 816]
        manager.remove_req = AsyncMock()
        manager.abort = AsyncMock()
        other_request_load = 17
        other_request_count = 3
        p_node = MagicMock(
            dispatched_prompt_chars=other_request_load,
            dispatched_req_num=other_request_count,
        )
        d_node = MagicMock()
        dispatched_nodes = []
        dispatched_prompts = []
        dispatched_loads = []
        dispatched_req_counts = []

        async def wait_to_token_package(
            selected_p_node, _d_node, _start_time, block_prompt, sampling_params, *_args, **_kwargs
        ):
            dispatched_nodes.append(selected_p_node)
            dispatched_prompts.append(block_prompt)
            dispatched_loads.append(selected_p_node.dispatched_prompt_chars)
            dispatched_req_counts.append(selected_p_node.dispatched_req_num)
            yield (
                sampling_params.group_request_id,
                "x",
                {"prompt_tokens": 1},
                FinishStatus(FinishStatus.FINISHED_LENGTH),
            )

        manager._wait_to_token_package = wait_to_token_package

        results = []
        async for result in manager._generate_one(
            "prompt",
            SamplingParams(),
            MagicMock(),
            MagicMock(),
            0,
            800,
            p_node,
            d_node,
        ):
            results.append(result)

        assert dispatched_nodes == [p_node, p_node]
        assert dispatched_prompts == ["prompt", "promptx"]
        assert dispatched_loads == [other_request_load + len("prompt"), other_request_load + len("promptx")]
        assert dispatched_req_counts == [other_request_count + 1, other_request_count + 1]
        assert p_node.dispatched_prompt_chars == other_request_load
        assert p_node.dispatched_req_num == other_request_count
        assert len(results) == 2

    asyncio.run(asyncio.wait_for(run(), timeout=2))


def test_pd_master_releases_prefill_load_when_stream_is_closed():
    async def run():
        manager = _manager()
        manager._split_max_new_tokens = MagicMock(return_value=[4])
        manager.id_gen.generate_id.return_value = 808
        manager.remove_req = AsyncMock()
        manager.abort = AsyncMock()
        other_request_load = 17
        other_request_count = 3
        p_node = MagicMock(
            dispatched_prompt_chars=other_request_load,
            dispatched_req_num=other_request_count,
        )
        d_node = MagicMock()

        async def wait_to_token_package(*_args, **_kwargs):
            yield 808, "first", {"prompt_tokens": 1}, FinishStatus()
            await asyncio.sleep(10)

        manager._wait_to_token_package = wait_to_token_package
        generator = manager._generate_one(
            "prompt",
            SamplingParams(),
            MagicMock(),
            MagicMock(),
            0,
            800,
            p_node,
            d_node,
        )

        assert (await generator.__anext__())[1] == "first"
        assert p_node.dispatched_prompt_chars == other_request_load
        assert p_node.dispatched_req_num == other_request_count
        await generator.aclose()

        assert p_node.dispatched_prompt_chars == other_request_load
        assert p_node.dispatched_req_num == other_request_count
        manager.abort.assert_awaited_once()

    asyncio.run(asyncio.wait_for(run(), timeout=2))
