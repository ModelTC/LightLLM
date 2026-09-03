from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from lightllm.server.core.objs import FinishStatus, SamplingParams
from lightllm.server.httpserver.manager import HttpServerManager
from lightllm.server.httpserver_for_pd_master.manager import HttpServerManagerForPDMaster


def _manager() -> HttpServerManagerForPDMaster:
    manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
    manager.pd_manager = MagicMock()
    manager.id_gen = MagicMock()
    manager.id_gen.generate_id.return_value = 800
    manager.metric_client = MagicMock()
    manager._log_req_header = AsyncMock()
    manager.tokens = MagicMock(return_value=2)
    manager.pd_high_priority_request_time_out_seconds = 60
    return manager


def test_pd_master_expands_n_into_concurrent_single_choice_requests():
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

        started = set()
        all_started = asyncio.Event()
        captured_params = []
        p_node = MagicMock(dispatched_prompt_chars=0, dispatched_req_num=0)
        d_node = MagicMock()
        manager.select_p_d_node = AsyncMock(return_value=(p_node, d_node, 0.0))
        manager.remove_req = AsyncMock()

        async def wait_to_token_package(
            selected_p_node,
            selected_d_node,
            start_time,
            prompt,
            choice_sampling_params,
            child_multimodal_params,
            child_request,
        ):
            captured_params.append(choice_sampling_params)
            started.add(choice_sampling_params.group_request_id)
            if len(started) == 3:
                all_started.set()
            await all_started.wait()
            for token_index in range(2):
                finish_status = FinishStatus() if token_index == 0 else FinishStatus(FinishStatus.FINISHED_STOP)
                yield (
                    choice_sampling_params.group_request_id,
                    f"internal-{choice_sampling_params.group_request_id}-{token_index}",
                    {"prompt_tokens": 2, "count_output_tokens": token_index + 1},
                    finish_status,
                )

        manager._wait_to_token_package = wait_to_token_package

        with patch.object(HttpServerManager, "_check_and_repair_length", new=AsyncMock()):
            results = []
            async for result in manager._generate("prompt", sampling_params, multimodal_params, request):
                results.append(result)

        assert [result[0] for result in results].count(800) == 2
        assert [result[0] for result in results].count(801) == 2
        assert [result[0] for result in results].count(802) == 2
        assert {result[1] for result in results} == {
            "internal-808-0",
            "internal-808-1",
            "internal-816-0",
            "internal-816-1",
            "internal-824-0",
            "internal-824-1",
        }
        assert all(params.n == 1 and params.best_of == 1 for params in captured_params)
        assert {params.group_request_id for params in captured_params} == {808, 816, 824}
        assert sampling_params.n == 3
        assert sampling_params.best_of == 3
        multimodal_params.verify_and_preload.assert_awaited_once_with(request)
        manager._log_req_header.assert_awaited_once_with(request, 800)
        assert manager.metric_client.counter_inc.call_args_list == [
            call("lightllm_request_count"),
            call("lightllm_request_success"),
        ]
        assert p_node.dispatched_prompt_chars == 0
        assert p_node.dispatched_req_num == 0

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
        merge_choice_generators = manager._merge_choice_generators
        manager._merge_choice_generators = MagicMock(side_effect=merge_choice_generators)

        async def generate_one(
            prompt,
            choice_sampling_params,
            child_multimodal_params,
            child_request,
            start_time,
            origin_request_id,
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


@pytest.mark.parametrize(
    "failed_finish_status",
    [FinishStatus.FINISHED_ABORTED, FinishStatus.FINISHED_ERROR],
)
def test_pd_master_does_not_record_aborted_or_error_request_as_success(failed_finish_status):
    async def run():
        manager = _manager()
        sampling_params = SamplingParams()
        sampling_params.n = 1
        sampling_params.best_of = 1
        sampling_params.max_new_tokens = 4

        multimodal_params = MagicMock()
        multimodal_params.verify_and_preload = AsyncMock()
        request = MagicMock()

        async def generate_one(*_args, **_kwargs):
            yield (
                800,
                "",
                {"prompt_tokens": 2},
                FinishStatus(failed_finish_status),
            )

        manager._generate_one = generate_one

        with patch.object(HttpServerManager, "_check_and_repair_length", new=AsyncMock()):
            results = []
            async for result in manager._generate("prompt", sampling_params, multimodal_params, request):
                results.append(result)

        assert len(results) == 1
        assert results[0][3].get_status() == failed_finish_status
        manager.metric_client.counter_inc.assert_called_once_with("lightllm_request_count")

    asyncio.run(asyncio.wait_for(run(), timeout=2))


def test_pd_master_hides_capacity_finish_token_and_continues_next_segment():
    async def run():
        manager = _manager()
        manager.id_gen.generate_id.side_effect = [808, 816]
        manager.remove_req = AsyncMock()
        manager.abort = AsyncMock()
        p_node = MagicMock(dispatched_prompt_chars=0, dispatched_req_num=0)
        d_node = MagicMock()
        manager.select_p_d_node = AsyncMock(return_value=(p_node, d_node, 0.0))
        segment_index = 0

        async def wait_to_token_package(_p_node, _d_node, _start_time, prompt, params, *_args):
            nonlocal segment_index
            segment_index += 1
            if segment_index == 1:
                assert prompt == "prompt"
                assert params.max_new_tokens == 4
                yield (
                    808,
                    "visible",
                    {"prompt_tokens": 1, "id": 10, "logprob": -0.1, "logprobs": {"visible": -0.1}},
                    FinishStatus(),
                )
                yield (
                    808,
                    "simulated-eos",
                    {"prompt_tokens": 1, "id": 11, "logprob": 0.0, "logprobs": {"eos": 0.0}},
                    FinishStatus(FinishStatus.FINISHED_PD_DECODE_CAPACITY),
                )
            else:
                assert prompt == "promptvisible"
                assert params.max_new_tokens == 3
                yield (
                    816,
                    "continued",
                    {"prompt_tokens": 2, "id": 12, "logprob": -0.2, "logprobs": {"continued": -0.2}},
                    FinishStatus(FinishStatus.FINISHED_STOP),
                )

        manager._wait_to_token_package = wait_to_token_package
        sampling_params = SamplingParams()
        sampling_params.max_new_tokens = 4

        results = []
        async for result in manager._generate_one(
            "prompt",
            sampling_params,
            MagicMock(),
            MagicMock(),
            0,
            800,
        ):
            results.append(result)

        assert segment_index == 2
        assert [result[1] for result in results] == ["visible", "continued"]
        assert all(result[3].status != FinishStatus.FINISHED_PD_DECODE_CAPACITY for result in results)
        assert results[-1][3].status == FinishStatus.FINISHED_STOP

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
        manager.id_gen.generate_id.return_value = 808
        manager.remove_req = AsyncMock()
        manager.abort = AsyncMock()
        p_node = MagicMock(dispatched_prompt_chars=0, dispatched_req_num=0)
        d_node = MagicMock()
        manager.select_p_d_node = AsyncMock(return_value=(p_node, d_node, 0.0))

        async def failing_wait_to_token_package(*_args, **_kwargs):
            raise RuntimeError("generation failed")
            yield None

        manager._wait_to_token_package = failing_wait_to_token_package
        sampling_params = SamplingParams()
        sampling_params.max_new_tokens = 4

        with pytest.raises(RuntimeError, match="generation failed"):
            # 空 prompt 的字符负载为 0，但已派发请求数仍必须在异常路径释放。
            async for _ in manager._generate_one(
                "",
                sampling_params,
                MagicMock(),
                MagicMock(),
                0,
                800,
            ):
                pass

        assert p_node.dispatched_prompt_chars == 0
        assert p_node.dispatched_req_num == 0
        manager.pd_manager.selector.insert_prompt_cache.assert_not_called()

    asyncio.run(asyncio.wait_for(run(), timeout=2))


def test_pd_master_dynamic_split_reuses_nodes_with_remaining_length():
    async def run():
        manager = _manager()
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
        manager.select_p_d_node = AsyncMock(return_value=(p_node, d_node, 0.0))
        dispatched_nodes = []
        dispatched_d_nodes = []
        dispatched_prompts = []
        dispatched_loads = []
        dispatched_req_counts = []
        high_priority_request_flags = []
        dispatched_max_new_tokens = []

        async def wait_to_token_package(
            selected_p_node, selected_d_node, _start_time, block_prompt, sampling_params, *_args
        ):
            dispatched_nodes.append(selected_p_node)
            dispatched_d_nodes.append(selected_d_node)
            dispatched_prompts.append(block_prompt)
            dispatched_loads.append(selected_p_node.dispatched_prompt_chars)
            dispatched_req_counts.append(selected_p_node.dispatched_req_num)
            high_priority_request_flags.append(sampling_params.pd_high_priority_request)
            dispatched_max_new_tokens.append(sampling_params.max_new_tokens)
            yield (
                sampling_params.group_request_id,
                "x",
                {
                    "prompt_tokens": 1 if len(dispatched_max_new_tokens) == 1 else 2,
                    "count_output_tokens": 1,
                },
                (FinishStatus() if len(dispatched_max_new_tokens) == 1 else FinishStatus(FinishStatus.FINISHED_LENGTH)),
            )
            if len(dispatched_max_new_tokens) == 1:
                yield (
                    sampling_params.group_request_id,
                    "simulated-eos",
                    {"prompt_tokens": 1, "count_output_tokens": 1},
                    FinishStatus(FinishStatus.FINISHED_PD_DECODE_CAPACITY),
                )

        manager._wait_to_token_package = wait_to_token_package

        results = []
        sampling_params = SamplingParams()
        sampling_params.max_new_tokens = 2
        multimodal_params = MagicMock()
        async for result in manager._generate_one(
            "prompt",
            sampling_params,
            multimodal_params,
            MagicMock(),
            0,
            800,
        ):
            results.append(result)

        manager.select_p_d_node.assert_awaited_once_with("prompt", sampling_params, multimodal_params)
        assert dispatched_nodes == [p_node, p_node]
        assert dispatched_d_nodes == [d_node, d_node]
        assert dispatched_prompts == ["prompt", "promptx"]
        assert dispatched_loads == [other_request_load + len("prompt"), other_request_load + len("promptx")]
        assert dispatched_req_counts == [other_request_count + 1, other_request_count + 1]
        assert high_priority_request_flags == [False, True]
        assert dispatched_max_new_tokens == [2, 1]
        assert p_node.dispatched_prompt_chars == other_request_load
        assert p_node.dispatched_req_num == other_request_count
        assert len(results) == 2
        assert [result[2]["prompt_tokens"] for result in results] == [1, 1]
        assert not results[0][3].is_finished()
        assert results[1][3].is_finished_length()

    asyncio.run(asyncio.wait_for(run(), timeout=2))


def test_pd_master_counts_segment_tokens_without_relying_on_metadata():
    async def run():
        manager = _manager()
        manager.id_gen.generate_id.side_effect = [808, 816]
        manager.remove_req = AsyncMock()
        manager.abort = AsyncMock()
        p_node = MagicMock(dispatched_prompt_chars=0, dispatched_req_num=0)
        d_node = MagicMock()
        manager.select_p_d_node = AsyncMock(return_value=(p_node, d_node, 0.0))
        dispatched_max_new_tokens = []

        async def wait_to_token_package(_p_node, _d_node, _start_time, _prompt, sampling_params, *_args):
            dispatched_max_new_tokens.append(sampling_params.max_new_tokens)
            is_first_segment = len(dispatched_max_new_tokens) == 1
            token_count = 2 if is_first_segment else 1
            for token_index in range(token_count):
                finish_status = (
                    FinishStatus(FinishStatus.FINISHED_LENGTH)
                    if not is_first_segment and token_index == token_count - 1
                    else FinishStatus()
                )
                yield (
                    sampling_params.group_request_id,
                    "x",
                    {"prompt_tokens": 1, "count_output_tokens": 100},
                    finish_status,
                )
            if is_first_segment:
                yield (
                    sampling_params.group_request_id,
                    "simulated-eos",
                    {"prompt_tokens": 1, "count_output_tokens": 100},
                    FinishStatus(FinishStatus.FINISHED_PD_DECODE_CAPACITY),
                )

        manager._wait_to_token_package = wait_to_token_package

        sampling_params = SamplingParams()
        sampling_params.max_new_tokens = 3
        async for _ in manager._generate_one(
            "prompt",
            sampling_params,
            MagicMock(),
            MagicMock(),
            0,
            800,
        ):
            pass

        assert dispatched_max_new_tokens == [3, 1]

    asyncio.run(asyncio.wait_for(run(), timeout=2))


@pytest.mark.parametrize(
    ("estimated_cache_hit_rate", "expected_high_priority"),
    [(0.8, False), (0.81, True)],
)
def test_pd_master_promotes_request_with_high_estimated_cache_hit_rate(
    estimated_cache_hit_rate,
    expected_high_priority,
):
    async def run():
        manager = _manager()
        manager.id_gen.generate_id.return_value = 808
        manager.remove_req = AsyncMock()
        manager.abort = AsyncMock()
        p_node = MagicMock(dispatched_prompt_chars=0, dispatched_req_num=0)
        d_node = MagicMock()
        manager.select_p_d_node = AsyncMock(return_value=(p_node, d_node, estimated_cache_hit_rate))
        high_priority_request_flags = []

        async def wait_to_token_package(_p_node, _d_node, _start_time, _prompt, sampling_params, *_args):
            high_priority_request_flags.append(sampling_params.pd_high_priority_request)
            yield (
                sampling_params.group_request_id,
                "x",
                {"prompt_tokens": 1, "count_output_tokens": 1},
                FinishStatus(FinishStatus.FINISHED_STOP),
            )

        manager._wait_to_token_package = wait_to_token_package

        sampling_params = SamplingParams()
        sampling_params.max_new_tokens = 1
        async for _ in manager._generate_one(
            "prompt",
            sampling_params,
            MagicMock(),
            MagicMock(),
            0,
            800,
        ):
            pass

        assert high_priority_request_flags == [expected_high_priority]

    asyncio.run(asyncio.wait_for(run(), timeout=2))


def test_pd_master_sets_high_priority_timeout():
    async def run():
        manager = _manager()
        manager.pd_high_priority_request_time_out_seconds = 90
        manager.id_gen.generate_id.return_value = 808
        manager.remove_req = AsyncMock()
        manager.abort = AsyncMock()
        p_node = MagicMock(
            dispatched_prompt_chars=0,
            dispatched_req_num=0,
        )
        d_node = MagicMock()
        manager.select_p_d_node = AsyncMock(return_value=(p_node, d_node, 0.81))
        captured_timeout_seconds = []

        async def wait_to_token_package(_p_node, _d_node, _start_time, _prompt, sampling_params, *_args):
            captured_timeout_seconds.append(sampling_params.pd_high_priority_request_time_out_seconds)
            yield (
                sampling_params.group_request_id,
                "x",
                {"prompt_tokens": 1, "count_output_tokens": 1},
                FinishStatus(FinishStatus.FINISHED_STOP),
            )

        manager._wait_to_token_package = wait_to_token_package

        sampling_params = SamplingParams()
        sampling_params.max_new_tokens = 1
        async for _ in manager._generate_one(
            "prompt",
            sampling_params,
            MagicMock(),
            MagicMock(),
            0,
            800,
        ):
            pass

        assert captured_timeout_seconds == [90]

    asyncio.run(asyncio.wait_for(run(), timeout=2))


def test_pd_master_releases_prefill_load_when_stream_is_closed():
    async def run():
        manager = _manager()
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
        manager.select_p_d_node = AsyncMock(return_value=(p_node, d_node, 0.0))

        async def wait_to_token_package(*_args, **_kwargs):
            yield 808, "first", {"prompt_tokens": 1, "count_output_tokens": 1}, FinishStatus()
            await asyncio.sleep(10)

        manager._wait_to_token_package = wait_to_token_package
        sampling_params = SamplingParams()
        sampling_params.max_new_tokens = 4
        generator = manager._generate_one(
            "prompt",
            sampling_params,
            MagicMock(),
            MagicMock(),
            0,
            800,
        )

        assert (await generator.__anext__())[1] == "first"
        assert p_node.dispatched_prompt_chars == other_request_load
        assert p_node.dispatched_req_num == other_request_count
        await generator.aclose()

        assert p_node.dispatched_prompt_chars == other_request_load
        assert p_node.dispatched_req_num == other_request_count
        manager.abort.assert_awaited_once()

    asyncio.run(asyncio.wait_for(run(), timeout=2))
