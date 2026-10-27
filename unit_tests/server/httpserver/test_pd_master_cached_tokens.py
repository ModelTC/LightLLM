import asyncio
import copy
import pickle
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import numpy as np

from lightllm.server.core.objs import StopStrMatchMode, FinishStatus, SamplingParams
from lightllm.server.router.model_infer.infer_batch import InferReq
from lightllm.server.httpserver_for_pd_master.manager import HttpServerManagerForPDMaster, PDManager
from lightllm.server.httpserver_for_pd_master.pd_selector import PDSelectionExtraInfo
from lightllm.server.pd_io_struct import PD_Client_Obj

pytestmark = pytest.mark.usefixtures("single_http_worker_start_args")


def _make_manager(monkeypatch):
    monkeypatch.setattr(
        "lightllm.server.httpserver.manager.HttpServerManager._check_and_repair_length",
        classmethod(lambda cls, *a, **k: asyncio.sleep(0)),
    )
    monkeypatch.setattr(SamplingParams, "from_buffer_copy", classmethod(lambda cls, other: copy.copy(other)))
    mgr = object.__new__(HttpServerManagerForPDMaster)
    mgr.active_req_ids = set()
    mgr.req_id_to_out_inf = {}
    mgr.args = SimpleNamespace(
        disable_pd_master_decode_capacity_limit=True,
        reasoning_parser=None,
        enable_rl=False,
        enable_return_routed_experts=False,
        select_p_d_node_strategy="cache_aware",
        pd_master_mode="elastic",
    )
    mgr.enable_pd_node_self_request_limit = True
    mgr.pd_node_resource_wait_timeout_seconds = -1
    mgr.pd_node_continuation_resource_wait_timeout_seconds = 60
    mgr.pd_node_busy_retry_timeout_seconds = 120
    mgr.pd_cache_high_priority_max_age_seconds = 60
    mgr.pd_cache_high_priority_min_prompt_tokens = 8192
    mgr.disable_pd_cache_high_priority = False
    mgr.running_request_count = 0
    counter = [0]

    def gen_id():
        counter[0] += 1
        return counter[0]

    mgr.metric_client = SimpleNamespace(counter_inc=lambda *a, **k: None, histogram_observe=lambda *a, **k: None)
    mgr.tokens = lambda prompt, *a, **k: 10 + len(prompt) - len("hello")
    mgr.tokenizer = SimpleNamespace(encode=lambda text, _multimodal=None, **kwargs: [1, 2])
    mgr._log_req_header = lambda *a, **k: asyncio.sleep(0)
    mgr.recorded_cache_hit_rates = []
    mgr.inserted_prompt_caches = []
    mgr.segment_inputs = []
    mgr.pd_manager = PDManager(mgr.args)
    mgr.pd_node_stats = mgr.pd_manager.selector.pd_node_stats
    mgr.id_gen = SimpleNamespace(generate_id=gen_id)
    original_record_hit_rate = mgr.pd_manager.selector.record_prompt_cache_hit_rate
    prompt_cache_tree = mgr.pd_manager.selector.policy.prompt_cache_tree
    original_insert_cache = prompt_cache_tree.insert_sampled

    def record_hit_rate(hit_rate):
        mgr.recorded_cache_hit_rates.append(hit_rate)
        original_record_hit_rate(hit_rate)

    async def insert_cache(prompt_key, connection_key, sample_stride):
        mgr.inserted_prompt_caches.append((prompt_key, connection_key))
        await original_insert_cache(prompt_key, connection_key, sample_stride)

    mgr.pd_manager.selector.record_prompt_cache_hit_rate = record_hit_rate
    prompt_cache_tree.insert_sampled = insert_cache
    p_node = PD_Client_Obj(
        node_id=100,
        client_ip_port="prefill:8000",
        mode="prefill",
        start_args={},
        websocket=SimpleNamespace(send_bytes=AsyncMock()),
    )
    d_node = PD_Client_Obj(
        node_id=200,
        client_ip_port="decode:8000",
        mode="decode",
        start_args={},
        websocket=SimpleNamespace(send_bytes=AsyncMock()),
    )
    mgr.pd_manager.prefill_nodes = [p_node]
    mgr.pd_manager.decode_nodes = [d_node]
    mgr.pd_manager.url_to_pd_nodes = {node.connection_key: node for node in [p_node, d_node]}
    mgr.pd_manager.selector.update_nodes([p_node], [d_node])
    mgr.select_p_d_node = lambda *a, **k: asyncio.sleep(0, result=(p_node, d_node, PDSelectionExtraInfo()))
    mgr.remove_req = lambda *a, **k: asyncio.sleep(0)
    return mgr


def _collect(
    mgr,
    sampling_params,
    monkeypatch,
    segments,
    first_node_mode="prefill",
    needs_prefill_first_token=True,
    first_token_text="",
    prompt="hello",
):
    segment_iter = iter(segments)

    async def fake_wait(p_node, d_node, start_time, prompt_ids, sp, multimodal_params, request):
        mgr.segment_inputs.append(
            pickle.loads(pickle.dumps((prompt_ids, sp.constraint_out_params.pd_previous_output_len)))
        )
        sub_req_id = sp.group_request_id
        token_count, final_status = next(segment_iter)
        hit = sampling_params.max_new_tokens * 10
        for token_index in range(1, token_count + 1):
            finish_status = FinishStatus()
            if token_index == token_count:
                finish_status = FinishStatus(final_status)
            metadata = {
                "id": ord("x"),
                "prompt_tokens": 100,
                "prompt_cache_len": hit if token_index == 1 else 0,
                "count_output_tokens": token_index,
                "node_mode": first_node_mode if token_index == 1 else "decode",
            }
            if token_index == 1 and needs_prefill_first_token is not None:
                metadata["needs_prefill_first_token"] = needs_prefill_first_token
            yield sub_req_id, first_token_text if token_index == 1 else "x", metadata, finish_status

    monkeypatch.setattr(mgr, "_wait_to_token_package", fake_wait)

    async def run():
        out = []
        async for sub_id, out_str, metadata, finish in mgr.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            multimodal_params=SimpleNamespace(images=[], audios=[], verify_and_preload=lambda req: asyncio.sleep(0)),
            request=None,
        ):
            assert "needs_prefill_first_token" not in metadata
            out.append(metadata.get("prompt_cache_len", -1))
        return out

    return asyncio.run(run())


def test_single_block_prefill_hit_persists_past_decode_zeros(monkeypatch):
    mgr = _make_manager(monkeypatch)
    sp = SamplingParams()
    sp.init(None)
    sp.n = 1
    sp.max_new_tokens = 3
    sp.best_of = 1
    sp.group_request_id = 0
    cached = _collect(mgr, sp, monkeypatch, segments=[(3, FinishStatus.FINISHED_STOP)])
    assert cached and all(c == 30 for c in cached), cached
    assert mgr.recorded_cache_hit_rates == [pytest.approx(0.3)]
    assert len(mgr.inserted_prompt_caches) == 1
    assert mgr.inserted_prompt_caches[0][0] == "h"


@pytest.mark.parametrize("mode", [StopStrMatchMode.ALL_TEXT, StopStrMatchMode.CONTENT_ONLY])
@pytest.mark.parametrize("hidden_prefix", [False, True])
@pytest.mark.parametrize("token_chunks", [[[10], [11, 12]], [[10], [11], [12, 13]]])
def test_pd_token_id_stop_matches_across_capacity_segments(monkeypatch, mode, hidden_prefix, token_chunks):
    async def run():
        manager = _make_manager(monkeypatch)
        params = SamplingParams()
        stop_ids = list(range(10, 10 + len(token_chunks)))
        params.init(None, stop_sequences=[stop_ids], stop_str_match_mode=mode, max_new_tokens=6)
        segments = []

        async def raw_segment(p_node, d_node, start_time, prompt, sp, multimodal, request):
            index = len(segments)
            segments.append(sp.group_request_id)
            req = InferReq.__new__(InferReq)
            req.stop_sequences = sp.stop_sequences.to_token_ids_list()
            req.sampling_param = SimpleNamespace(shm_param=sp)
            tokens = token_chunks[index]
            req.shm_req = SimpleNamespace(input_len=1, shm_prompt_ids=SimpleNamespace(arr=np.array([1, *tokens])))
            for output_len, token_id in enumerate(tokens, 1):
                req.finish_status = FinishStatus()
                req.update_finish_status([], output_len)
                if index == len(token_chunks) - 1 and output_len == len(tokens) and not req.finish_status.is_finished():
                    req.finish_status = FinishStatus(FinishStatus.FINISHED_LENGTH)
                text = "" if token_id == 10 and hidden_prefix else str(token_id)
                yield sp.group_request_id, text, {"id": token_id, "prompt_tokens": 2}, req.finish_status
                if req.finish_status.is_finished():
                    return
            yield sp.group_request_id, "", {"id": None}, FinishStatus(FinishStatus.FINISHED_PD_DECODE_CAPACITY)

        manager._wait_to_token_package = raw_segment
        # 分段内仍由真实 InferReq 判断；跨分段匹配由对外生成流的包装层完成。
        from lightllm.server.httpserver.stop_seq_match import wrap_stop_match_stream

        raw_generator = manager._generate_one_attempt("prompt", params, None, None, 0, 80, 2)
        results = [result async for result in wrap_stop_match_stream(manager, raw_generator, params, "prompt")]
        assert [result[2]["id"] for result in results] == stop_ids
        assert results[-1][3].is_stopped()
        assert len(segments) == len(token_chunks)

    asyncio.run(asyncio.wait_for(run(), timeout=2))


def test_dynamic_split_keeps_first_segment_hit(monkeypatch):
    mgr = _make_manager(monkeypatch)
    sp = SamplingParams()
    sp.init(None)
    sp.n = 1
    sp.max_new_tokens = 5
    sp.best_of = 1
    sp.group_request_id = 0
    cached = _collect(
        mgr,
        sp,
        monkeypatch,
        segments=[
            (3, FinishStatus.FINISHED_PD_DECODE_CAPACITY),
            (2, FinishStatus.FINISHED_PD_DECODE_CAPACITY),
            (1, FinishStatus.FINISHED_STOP),
        ],
    )
    assert cached == [50, 50, 50, 50], cached
    assert mgr.recorded_cache_hit_rates == [pytest.approx(0.5)]
    assert len(mgr.inserted_prompt_caches) == 1
    assert mgr.inserted_prompt_caches[0][0] == "h"
    # 无约束请求不需要计算语法状态重放长度。
    assert mgr.segment_inputs == [
        ("hello", 0),
        ("hellox", 0),
        ("hellox", 0),
    ]


@pytest.mark.parametrize("add_special_tokens", [False, True])
@pytest.mark.parametrize("prompt,expected_lengths", [("P", (0, 1)), ("PP", (0, 0))])
@pytest.mark.parametrize("constraint_enabled", [False, True])
def test_continuation_uses_reencoded_length_difference(
    monkeypatch, add_special_tokens, prompt, expected_lengths, constraint_enabled
):
    from tokenizers import Tokenizer, models, processors
    from unittest.mock import Mock, call

    tokenizer = Tokenizer(models.BPE({"P": 0, "x": 1, "Px": 2, "<bos>": 3, "PPx": 4}, [("P", "x"), ("P", "Px")]))
    tokenizer.post_processor = processors.TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", 3)])
    manager = _make_manager(monkeypatch)
    del manager.tokens
    manager.tokenizer = SimpleNamespace(
        encode=Mock(side_effect=lambda text, _multimodal=None, **kwargs: tokenizer.encode(text, **kwargs).ids)
    )
    params = SamplingParams()
    params.init(
        None,
        max_new_tokens=4,
        add_special_tokens=add_special_tokens,
        regular_constraint=".*" if constraint_enabled else "",
    )
    _collect(
        manager,
        params,
        monkeypatch,
        segments=[
            (2, FinishStatus.FINISHED_PD_DECODE_CAPACITY),
            (2, FinishStatus.FINISHED_PD_DECODE_CAPACITY),
            (2, FinishStatus.FINISHED_STOP),
        ],
        first_token_text="x",
        prompt=prompt,
    )
    # 约束输出按拼接文本的编码长度差重放；普通请求跳过后续编码。
    assert manager.segment_inputs == [
        (prompt, 0),
        (prompt + "x", expected_lengths[0] if constraint_enabled else 0),
        (prompt + "xx", expected_lengths[1] if constraint_enabled else 0),
    ]
    expected_encoded_prompts = (prompt, prompt + "x", prompt + "xx") if constraint_enabled else (prompt,)
    assert manager.tokenizer.encode.call_args_list == [
        call(text, None, add_special_tokens=add_special_tokens) for text in expected_encoded_prompts
    ]


def test_error_result_records_hit_rate_without_inserting_prompt_cache(monkeypatch):
    mgr = _make_manager(monkeypatch)
    sampling_params = SamplingParams()
    sampling_params.init(None)
    sampling_params.n = 1
    sampling_params.best_of = 1
    sampling_params.max_new_tokens = 1

    async def failed_wait(_p_node, _d_node, _start_time, _prompt_ids, sp, *_args):
        yield (
            sp.group_request_id,
            "",
            {
                "id": None,
                "prompt_tokens": 100,
                "prompt_cache_len": 20,
                "count_output_tokens": 0,
                "needs_prefill_first_token": True,
            },
            FinishStatus(FinishStatus.FINISHED_ERROR),
        )

    monkeypatch.setattr(mgr, "_wait_to_token_package", failed_wait)

    async def run():
        async for _ in mgr.generate(
            prompt="hello",
            sampling_params=sampling_params,
            multimodal_params=SimpleNamespace(
                images=[],
                audios=[],
                verify_and_preload=lambda req: asyncio.sleep(0),
            ),
            request=None,
        ):
            pass

    asyncio.run(run())

    assert mgr.recorded_cache_hit_rates == [pytest.approx(0.2)]
    assert mgr.inserted_prompt_caches == []


def test_pd_raw_stop_does_not_use_configured_reasoning_parser(monkeypatch):
    manager = _make_manager(monkeypatch)
    manager.args.reasoning_parser = "qwen3"
    manager.tokenizer = SimpleNamespace(encode=lambda text, _multimodal=None, **kwargs: [ord(c) for c in text])
    params = SamplingParams()
    params.init(
        manager.tokenizer,
        stop_sequences="END",
        max_new_tokens=10,
        stop_str_match_mode=StopStrMatchMode.ALL_TEXT,
        _initial_reasoning_state=1,
    )

    async def fake_wait(p_node, d_node, start_time, prompt, sp, multimodal_params, request):
        assert sp.stop_str_match_mode.get_mode() == (StopStrMatchMode.ALL_TEXT)
        metadata = {"prompt_tokens": 10, "prompt_cache_len": 0, "count_output_tokens": 1, "id": 10}
        yield sp.group_request_id, "reasonENDextra", metadata, FinishStatus(FinishStatus.FINISHED_LENGTH)

    monkeypatch.setattr(manager, "_wait_to_token_package", fake_wait)

    async def run():
        return [
            result
            async for result in manager.generate(
                "prompt",
                params,
                SimpleNamespace(images=[], audios=[], verify_and_preload=lambda req: asyncio.sleep(0)),
                None,
            )
        ]

    results = asyncio.run(run())
    assert "".join(result[1] for result in results) == "reason"
    assert results[-1][3].is_stopped()


@pytest.mark.parametrize("in_reasoning", [False, True])
def test_pd_continuation_uses_local_reasoning_state(monkeypatch, in_reasoning):

    manager = _make_manager(monkeypatch)
    manager.args.reasoning_parser = "qwen3"
    manager.tokenizer = SimpleNamespace(
        encode=lambda text, _multimodal=None, **kwargs: {"<think>": [1000], "</think>": [1001]}.get(
            text, [ord(c) for c in text]
        )
    )
    params = SamplingParams()
    params.init(
        manager.tokenizer,
        stop_sequences="END",
        max_new_tokens=3,
        _initial_reasoning_state=1,
        stop_str_match_mode=StopStrMatchMode.CONTENT_ONLY,
    )
    segment_modes = []

    async def fake_wait(p_node, d_node, start_time, prompt, sp, multimodal_params, request):
        segment_modes.append(sp._initial_reasoning_state)
        assert sp.stop_str_match_mode.get_mode() == StopStrMatchMode.CONTENT_ONLY
        metadata = {"prompt_tokens": 10, "prompt_cache_len": 0, "count_output_tokens": 1}
        if len(segment_modes) == 1:
            yield sp.group_request_id, "reason", {**metadata, "id": 10}, FinishStatus()
            if not in_reasoning:
                yield sp.group_request_id, "", {**metadata, "id": 1001}, FinishStatus()
            # A simulated delimiter at the capacity boundary must not change state.
            yield (
                sp.group_request_id,
                "",
                {**metadata, "id": 1001},
                FinishStatus(FinishStatus.FINISHED_PD_DECODE_CAPACITY),
            )
        else:
            yield sp.group_request_id, "END", {**metadata, "id": 11}, FinishStatus(FinishStatus.FINISHED_STOP)

    monkeypatch.setattr(manager, "_wait_to_token_package", fake_wait)

    async def run():
        return [
            result
            async for result in manager.generate(
                "hello",
                params,
                SimpleNamespace(images=[], audios=[], verify_and_preload=lambda req: asyncio.sleep(0)),
                None,
            )
        ]

    results = asyncio.run(run())
    assert segment_modes == [1, 1]  # reasoning 状态只保存在应用侧，不再回写节点参数。
    assert "".join(result[1] for result in results) == "reason" + ("END" if in_reasoning else "")


@pytest.mark.parametrize("node_mode", ["decode", None])
def test_skipped_prefill_does_not_insert_prompt_cache(monkeypatch, node_mode):
    mgr = _make_manager(monkeypatch)
    sampling_params = SamplingParams()
    sampling_params.init(None)
    sampling_params.n = sampling_params.best_of = 1
    sampling_params.max_new_tokens = 3
    cached = _collect(
        mgr,
        sampling_params,
        monkeypatch,
        segments=[(3, FinishStatus.FINISHED_STOP)],
        first_node_mode=node_mode,
        needs_prefill_first_token=False,
    )
    assert cached == [30, 30, 30]
    assert mgr.recorded_cache_hit_rates == [pytest.approx(0.3)]
    assert mgr.inserted_prompt_caches == []


@pytest.mark.parametrize("needs_prefill_first_token", [True, None])
def test_completed_prefill_inserts_cache_when_first_output_is_decode(monkeypatch, needs_prefill_first_token):
    mgr = _make_manager(monkeypatch)
    sampling_params = SamplingParams()
    sampling_params.init(None)
    sampling_params.n = sampling_params.best_of = 1
    sampling_params.max_new_tokens = 3
    cached = _collect(
        mgr,
        sampling_params,
        monkeypatch,
        segments=[(3, FinishStatus.FINISHED_STOP)],
        first_node_mode="decode",
        needs_prefill_first_token=needs_prefill_first_token,
    )
    assert cached == [30, 30, 30]
    assert len(mgr.inserted_prompt_caches) == 1
    assert mgr.inserted_prompt_caches[0][0] == "h"


@pytest.mark.parametrize("include", [False, True])
@pytest.mark.parametrize("parser", [None, "qwen3"])
@pytest.mark.parametrize(
    "continuation,node_finish",
    [
        ("NDextra", FinishStatus.NO_FINISH),
        ("NDextra", FinishStatus.FINISHED_STOP),
        ("NX", FinishStatus.FINISHED_LENGTH),
        ("N", FinishStatus.FINISHED_LENGTH),
    ],
)
def test_pd_stop_prefix_survives_capacity_split(monkeypatch, include, parser, continuation, node_finish):
    manager = _make_manager(monkeypatch)
    manager.args.reasoning_parser = parser
    manager.tokenizer = SimpleNamespace(
        encode=lambda text, _multimodal=None, **kwargs: {"<think>": [1000], "</think>": [1001]}.get(
            text, [ord(c) for c in text]
        )
    )
    manager.abort = AsyncMock()
    stop_request = manager.stop_request
    manager.stop_request = AsyncMock(side_effect=stop_request)
    params = SamplingParams()
    params.init(
        manager.tokenizer,
        stop_sequences="END",
        include_stop_str_in_output=include,
        _initial_reasoning_state=int(parser is not None),
        stop_str_match_mode=StopStrMatchMode.ALL_TEXT if parser is None else StopStrMatchMode.CONTENT_ONLY,
        max_new_tokens=10,
    )
    segments = []

    async def fake_wait(p_node, d_node, start_time, prompt, sp, multimodal_params, request):
        segments.append(prompt)
        metadata = {"prompt_tokens": 10, "prompt_cache_len": 0, "count_output_tokens": 1}
        if len(segments) == 1:
            if parser:
                yield sp.group_request_id, "END", {**metadata, "id": 10}, FinishStatus()
                yield sp.group_request_id, "", {**metadata, "id": 1001}, FinishStatus()
            yield sp.group_request_id, "helloE", {**metadata, "id": 11}, FinishStatus()
            yield sp.group_request_id, "", dict(metadata), FinishStatus(FinishStatus.FINISHED_PD_DECODE_CAPACITY)
        else:
            yield sp.group_request_id, continuation, {**metadata, "id": 12}, FinishStatus(node_finish)
            if continuation == "NDextra" and node_finish == FinishStatus.NO_FINISH:
                yield sp.group_request_id, "", {**metadata, "id": None}, FinishStatus(FinishStatus.FINISHED_ABORTED)

    monkeypatch.setattr(manager, "_wait_to_token_package", fake_wait)

    async def run():
        return [
            result
            async for result in manager.generate(
                "prompt",
                params,
                SimpleNamespace(images=[], audios=[], verify_and_preload=lambda req: asyncio.sleep(0)),
                None,
            )
        ]

    results = asyncio.run(run())
    assert len(segments) == 2
    assert segments == ["prompt", "prompt" + ("END" if parser else "") + "helloE"]
    expected = "hello" + ("END" if include else "") if continuation == "NDextra" else "helloE" + continuation
    assert "".join(result[1] for result in results) == ("END" if parser else "") + expected
    assert [result[2]["id"] for result in results] == ([10, 1001] if parser else []) + [11, 12]
    assert results[-1][3].get_finish_reason() == ("stop" if continuation == "NDextra" else "length")
    assert all(not result[3].is_finished() for result in results[:-1])
    if continuation == "NDextra" and node_finish == FinishStatus.NO_FINISH:
        manager.stop_request.assert_awaited_once()
    else:
        manager.stop_request.assert_not_awaited()


def test_pd_stop_request_marks_only_matching_choice(monkeypatch):
    manager = _make_manager(monkeypatch)
    manager.active_req_ids.update((80, 81))

    async def run():
        await manager.stop_request(80)
        assert manager.active_req_ids == {81}
        await manager.stop_request(80)
        await manager.stop_request(999)
        assert manager.active_req_ids == {81}

    asyncio.run(run())


def test_pd_abort_removes_output_channel_by_default(monkeypatch):
    manager = _make_manager(monkeypatch)
    p_node = SimpleNamespace(websocket=SimpleNamespace(send_bytes=AsyncMock()))
    d_node = SimpleNamespace(websocket=SimpleNamespace(send_bytes=AsyncMock()))
    manager.req_id_to_out_inf = {88: SimpleNamespace(p_node=p_node, d_node=d_node), 96: object()}

    asyncio.run(manager.abort(88))

    assert set(manager.req_id_to_out_inf) == {96}
    for node in (p_node, d_node):
        node.websocket.send_bytes.assert_awaited_once()


def test_application_stop_discards_late_pd_output_and_closes_segment(monkeypatch):
    manager = _make_manager(monkeypatch)
    manager.tokenizer = SimpleNamespace(encode=lambda text, _multimodal=None, **kwargs: [ord(c) for c in text])
    params = SamplingParams()
    params.init(manager.tokenizer, stop_sequences="END", max_new_tokens=10)

    async def run():
        segments = []
        closed = asyncio.Event()
        stopped = asyncio.Event()
        manager.abort = AsyncMock()
        stop_request = manager.stop_request

        async def stop_and_notify(request_id):
            await stop_request(request_id)
            stopped.set()

        manager.stop_request = stop_and_notify

        async def raw_segment(p_node, d_node, start_time, prompt, sp, multimodal_params, request):
            segments.append(prompt)
            metadata = {"prompt_tokens": 10, "prompt_cache_len": 0, "count_output_tokens": 1, "id": 10}
            try:
                yield sp.group_request_id, "helloENDextra", metadata, FinishStatus()
                await stopped.wait()
                yield sp.group_request_id, "late", {**metadata, "id": 11}, FinishStatus()
            finally:
                closed.set()

        manager._wait_to_token_package = raw_segment
        results = [
            result
            async for result in manager.generate(
                "prompt",
                params,
                SimpleNamespace(images=[], audios=[], verify_and_preload=lambda req: asyncio.sleep(0)),
                None,
            )
        ]
        assert segments == ["prompt"]
        assert results[-1][1] == "hello"
        assert results[-1][3].is_stopped()
        assert not manager.active_req_ids
        await asyncio.wait_for(closed.wait(), timeout=1)
        assert closed.is_set()
        manager.abort.assert_awaited_once()

    asyncio.run(asyncio.wait_for(run(), timeout=2))


@pytest.mark.parametrize(
    "exit_mode, token_count", [("finish", 2), ("close", 1), ("error", 1), ("empty", 0), ("empty_prompt", 2)]
)
def test_pd_segment_summary_is_recorded_on_exit(monkeypatch, caplog, exit_mode, token_count):
    import math
    import time
    from unittest.mock import MagicMock

    manager = _make_manager(monkeypatch)
    manager.first_time_costs = MagicMock()
    manager.per_token_costs = MagicMock()
    manager.metric_client = MagicMock()
    params = SamplingParams()
    params.init(SimpleNamespace(encode=lambda *_args, **_kwargs: []), max_new_tokens=4, group_request_id=80)
    request = SimpleNamespace(headers={"X-Request-Id": "summary-test"}, is_disconnected=lambda: asyncio.sleep(0, False))
    caplog.set_level("INFO", logger="lightllm.server.httpserver_for_pd_master.manager")
    # Capture the configured logger, which intentionally disables propagation.
    from lightllm.server.httpserver_for_pd_master.manager import logger

    monkeypatch.setattr(logger, "handlers", [caplog.handler])
    monkeypatch.setattr(logger, "propagate", False)

    async def raw_stream(*_args):
        if exit_mode == "empty":
            return
        metadata = {
            "prompt_tokens": 0 if exit_mode == "empty_prompt" else 10,
            "prompt_cache_len": 0 if exit_mode == "empty_prompt" else 3,
        }
        yield 80, "first", dict(metadata), FinishStatus()
        if exit_mode == "error":
            raise RuntimeError("decode failed")
        yield 80, "last", dict(metadata), FinishStatus(FinishStatus.FINISHED_LENGTH)

    manager.fetch_pd_stream = raw_stream

    async def run():
        generator = manager._wait_to_token_package(None, None, time.time(), "prompt", params, None, request)
        if exit_mode == "close":
            await generator.__anext__()
            await generator.aclose()
        elif exit_mode == "error":
            with pytest.raises(RuntimeError, match="decode failed"):
                async for _ in generator:
                    pass
        else:
            assert len([result async for result in generator]) == token_count

    asyncio.run(run())
    summaries = [record.getMessage() for record in caplog.records if "out_token_counter:" in record.getMessage()]
    if token_count:
        assert len(summaries) == 1
        assert "X-Request-Id:summary-test" in summaries[0]
        assert f"out_token_counter:{token_count}" in summaries[0]
        manager.first_time_costs.add.assert_called_once()
        manager.per_token_costs.add.assert_called_once()
        assert math.isfinite(manager.per_token_costs.add.call_args.args[0])
        generated = [
            args
            for args, _ in manager.metric_client.histogram_observe.call_args_list
            if args[0] == "lightllm_request_generated_tokens"
        ]
        assert generated == [("lightllm_request_generated_tokens", token_count)]
    else:
        assert not summaries
        manager.per_token_costs.add.assert_not_called()
        manager.metric_client.histogram_observe.assert_not_called()


@pytest.mark.parametrize("late_status", [FinishStatus.NO_FINISH, FinishStatus.FINISHED_PD_DECODE_CAPACITY])
def test_pd_stopped_choice_records_segment_summary(monkeypatch, caplog, late_status):
    import time
    from unittest.mock import AsyncMock, MagicMock

    manager = _make_manager(monkeypatch)
    manager.first_time_costs = MagicMock()
    manager.per_token_costs = MagicMock()
    manager.metric_client = MagicMock()
    manager.abort = AsyncMock()
    params = SamplingParams()
    params.init(SimpleNamespace(encode=lambda *_args, **_kwargs: []), max_new_tokens=4)
    request = SimpleNamespace(headers={}, is_disconnected=lambda: asyncio.sleep(0, False))
    caplog.set_level("INFO", logger="lightllm.server.httpserver_for_pd_master.manager")
    # Capture the configured logger, which intentionally disables propagation.
    from lightllm.server.httpserver_for_pd_master.manager import logger

    monkeypatch.setattr(logger, "handlers", [caplog.handler])
    monkeypatch.setattr(logger, "propagate", False)

    async def raw_stream(_p_node, _d_node, _prompt, sampling_params, *_args):
        metadata = {"id": 1, "prompt_tokens": 10, "prompt_cache_len": 3}
        yield sampling_params.group_request_id, "first", dict(metadata), FinishStatus()
        yield sampling_params.group_request_id, "late", dict(metadata), FinishStatus(late_status)

    manager.fetch_pd_stream = raw_stream

    async def run():
        generator = manager._generate_one("prompt", params, None, request, time.time(), 80, 10)
        assert (await generator.__anext__())[1] == "first"
        await manager.stop_request(80)
        # stop_request 只做标记，恢复生成时才停止当前 P/D 分段。
        manager.abort.assert_not_awaited()
        with pytest.raises(StopAsyncIteration):
            await generator.__anext__()
        assert not manager.active_req_ids
        manager.abort.assert_awaited_once()
        summaries = [record.getMessage() for record in caplog.records if "out_token_counter:" in record.getMessage()]
        assert len(summaries) == 1
        assert "lightllm_req_id:1" in summaries[0]
        manager.metric_client.histogram_observe.assert_any_call("lightllm_request_generated_tokens", 2)

    asyncio.run(run())


def test_application_pd_stop_leaves_sibling_choice_running(monkeypatch):
    manager = _make_manager(monkeypatch)
    manager.tokenizer = SimpleNamespace(encode=lambda text, _multimodal=None, **kwargs: [ord(c) for c in text])
    params = SamplingParams()
    params.init(manager.tokenizer, stop_sequences="END", n=2, best_of=2, max_new_tokens=10)

    async def run():
        stopped = asyncio.Event()
        cleaned = asyncio.Event()
        segments = []
        manager.abort = AsyncMock(side_effect=lambda *args, **kwargs: stopped.set())
        original_remove_req = manager.remove_req

        async def remove_req(*args, **kwargs):
            await original_remove_req(*args, **kwargs)
            if not manager.active_req_ids:
                cleaned.set()

        manager.remove_req = remove_req

        async def raw_segment(p_node, d_node, start_time, prompt, sp, multimodal_params, request):
            index = len(segments)
            segments.append(sp.group_request_id)
            metadata = {"prompt_tokens": 10, "prompt_cache_len": 0, "count_output_tokens": 1, "id": 10}
            if index == 0:
                yield sp.group_request_id, "helloENDextra", dict(metadata), FinishStatus()
                await asyncio.sleep(0.01)
                yield sp.group_request_id, "late", dict(metadata), FinishStatus()
            else:
                yield sp.group_request_id, "otherE", dict(metadata), FinishStatus()
                await stopped.wait()
                # 保持 sibling 未结束，验证另一个 choice 停止后仍能继续生成。
                await asyncio.sleep(0.01)
                yield sp.group_request_id, "X", {**metadata, "id": 11}, FinishStatus(FinishStatus.FINISHED_LENGTH)

        manager._wait_to_token_package = raw_segment
        results = [
            result
            async for result in manager.generate(
                "prompt",
                params,
                SimpleNamespace(images=[], audios=[], verify_and_preload=lambda req: asyncio.sleep(0)),
                None,
            )
        ]
        ids = sorted({result[0] for result in results})
        assert len(ids) == 2
        assert "".join(result[1] for result in results if result[0] == ids[0]) == "hello"
        assert "".join(result[1] for result in results if result[0] == ids[1]) == "otherEX"
        assert {result[0]: result[3].get_finish_reason() for result in results if result[3].is_finished()} == {
            ids[0]: "stop",
            ids[1]: "length",
        }
        await asyncio.wait_for(cleaned.wait(), timeout=1)
        assert not manager.active_req_ids

    asyncio.run(asyncio.wait_for(run(), timeout=2))
