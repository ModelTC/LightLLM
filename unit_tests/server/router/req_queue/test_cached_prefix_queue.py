import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import lightllm.server.router.req_queue as module
from lightllm.server.router.req_queue.chunked_prefill.cached_prefix_queue import (
    CachedPrefixChunkedPrefillQueue,
    supports_cached_prefix_reservations,
)
from lightllm.server.router.req_queue.chunked_prefill.impl import ChunkedPrefillQueue
from lightllm.server.router.req_queue.cached_prefix import CachedPrefixReservations


class Load:
    def set_estimated_peak_token_count(self, value, index):
        self.peak = value

    def set_dynamic_max_load(self, value, index):
        self.ratio = value


class Request:
    def __init__(self, identity, base=1800, remaining=350, cached=512):
        self.request_id, self.input_len, self.shm_cur_kv_len = identity, base, base
        self.prompt_cache_len = cached
        self.is_paused = self.is_aborted = False
        self.sample_params = SimpleNamespace(suggested_dp_index=0, disable_prompt_cache=False)
        self.multimodal_params = SimpleNamespace(images=[], audios=[])
        self.shm_prompt_ids = SimpleNamespace(arr=np.arange(base, dtype=np.int64))
        self.linear_att_token_hash_list = SimpleNamespace(get_all=lambda: [17] * (base // 512))
        self.remaining = remaining

    def get_tuple_tokens(self, is_busy, ema):
        return self.input_len, self.remaining

    def get_first_router_need_tokens(self):
        return 512


def arguments(**changes):
    result = SimpleNamespace(
        dp=1,
        nnodes=1,
        run_mode="normal",
        enable_rl=False,
        disable_dynamic_prompt_cache=False,
        enable_cpu_cache=False,
        linear_att_page_block_num=1,
        linear_att_hash_page_size=512,
        short_prefill_token_threshold=None,
        diverse_mode=False,
        output_constraint_mode="none",
        first_token_constraint_mode=False,
        disable_chunked_prefill=False,
        model_dir="unused",
    )
    for name, value in changes.items():
        setattr(result, name, value)
    return result


def queue(requests, cls=CachedPrefixChunkedPrefillQueue):
    result = cls.__new__(cls)
    result.args = arguments()
    result.dp_index = 0
    result.max_total_tokens, result.running_max_req_size, result.batch_max_tokens = 16384, 8, 1024
    result.router = SimpleNamespace(router_statics=SimpleNamespace(ema_req_out_len=2048), shared_token_load=Load())
    result.is_busy = lambda: True
    if cls is CachedPrefixChunkedPrefillQueue:
        result.prefix_reservations = CachedPrefixReservations(result.args, 0)
    result._init_cache_list(SimpleNamespace(reqs=requests), True)
    return result


@pytest.fixture(autouse=True)
def namespace(monkeypatch):
    assert not torch.cuda.is_initialized()
    monkeypatch.setenv("LIGHTLLM_UNIQUE_SERVICE_NAME_ID", "owned-canonical-test")
    yield
    assert not torch.cuda.is_initialized()


def test_known_cached_prefix_changes_actual_admission_at_same_capacity():
    legacy = queue([Request(i) for i in range(7)], ChunkedPrefillQueue)
    candidate = queue([Request(i) for i in range(7)])
    assert not legacy._can_add_new_req(Request(7), True)
    assert candidate._can_add_new_req(Request(7), True) is True
    assert candidate.router.shared_token_load.peak == 13616
    assert candidate.max_total_tokens == 16384 and candidate.running_max_req_size == 8


def test_refactored_legacy_peak_matches_original_formula_for_ties_and_empty_batches():
    rng = random.Random(8245)
    for _ in range(1500):
        entries = [(rng.randrange(1, 3000), rng.randrange(0, 100)) for _ in range(rng.randrange(0, 17))]
        legacy = queue([], ChunkedPrefillQueue)
        legacy.cache_len_list = entries.copy()
        expected = max(
            (
                sum(a for a, b in sorted(entries, key=lambda e: -e[1])[:i]) + i * pair[1]
                for i, pair in enumerate(sorted(entries, key=lambda e: -e[1]), 1)
            ),
            default=0,
        )
        assert legacy._estimate_peak_tokens() == expected


@pytest.mark.parametrize(
    "change",
    [
        "images",
        "audios",
        "paused",
        "aborted",
        "disabled_cache",
        "unresident",
        "missing_hashes",
        "different_hashes",
        "different_tokens",
    ],
)
def test_uncertain_or_distinct_prefixes_preserve_legacy_rejection(change):
    requests = [Request(i) for i in range(8)]
    for request in requests:
        if change in ("images", "audios"):
            setattr(request.multimodal_params, change, [object()])
        elif change in ("paused", "aborted"):
            setattr(request, "is_" + change, True)
        elif change == "disabled_cache":
            request.sample_params.disable_prompt_cache = True
        elif change == "unresident":
            request.shm_cur_kv_len = 128
        elif change == "missing_hashes":
            request.linear_att_token_hash_list = SimpleNamespace(get_all=lambda: [])
        elif change == "different_hashes":
            request.linear_att_token_hash_list = SimpleNamespace(get_all=lambda i=request.request_id: [i])
        elif change == "different_tokens":
            request.shm_prompt_ids.arr += request.request_id * 1800
    candidate = queue(requests[:7])
    assert not candidate._can_add_new_req(requests[7], True)


def test_reset_and_load_preserve_tuple_identity_and_request_lifetime():
    requests = [Request(i, remaining=350 - i * 3) for i in range(7)]
    candidate = queue(requests)
    assert candidate._can_add_new_req(Request(7, remaining=420), True)
    peak, ratio = candidate._calcu_batch_token_load_batch_not_none(SimpleNamespace(reqs=requests))
    assert peak < 16384 and ratio == peak / 16384
    candidate._init_cache_list(SimpleNamespace(reqs=[Request(50)]), True)
    assert set(candidate.prefix_reservations.memo) == {50}
    assert len(candidate.prefix_reservations.by_tuple) == 1
    candidate._init_cache_list(None, True)
    assert not candidate.prefix_reservations.memo and candidate._estimate_peak_tokens() == 0


@pytest.mark.parametrize(
    "change",
    [
        dict(dp=2),
        dict(nnodes=2),
        dict(run_mode="decode"),
        dict(enable_rl=True),
        dict(disable_dynamic_prompt_cache=True),
        dict(enable_cpu_cache=True),
        dict(linear_att_page_block_num=4),
        dict(linear_att_hash_page_size=0),
    ],
)
def test_unverified_cache_geometry_and_modes_are_not_enabled(change):
    assert not supports_cached_prefix_reservations(arguments(**change))


@pytest.mark.parametrize(
    "model_type,expected",
    [("qwen4_exp", True), ("qwen4_exp_text", True), ("qwen3_5", False), ("llama", False), (None, False)],
)
def test_factory_selects_only_verified_models(monkeypatch, model_type, expected):
    monkeypatch.setattr("lightllm.utils.config_utils.get_model_type", lambda path: model_type)
    assert (module._get_req_queue_class(arguments(), None, 1) is CachedPrefixChunkedPrefillQueue) == expected


@pytest.mark.parametrize(
    "change",
    [
        dict(linear_att_page_block_num=4),
        dict(enable_cpu_cache=True),
        dict(dp=2),
        dict(output_constraint_mode="x"),
        dict(first_token_constraint_mode=True),
    ],
)
def test_factory_retains_legacy_queue_for_unsupported_configurations(monkeypatch, change):
    monkeypatch.setattr("lightllm.utils.config_utils.get_model_type", lambda path: "qwen4_exp")
    assert module._get_req_queue_class(arguments(**change), None, 1) is ChunkedPrefillQueue


def test_request_limit_is_unchanged():
    candidate = queue([Request(i) for i in range(8)])
    assert not candidate._can_add_new_req(Request(8), True)


def test_unknown_namespace_gets_no_credit(monkeypatch):
    monkeypatch.delenv("LIGHTLLM_UNIQUE_SERVICE_NAME_ID")
    candidate = queue([Request(i) for i in range(7)])
    assert not candidate._can_add_new_req(Request(7), True)


def test_missing_prompt_handle_attaches_via_existing_request_api():
    requests = [Request(i) for i in range(8)]
    linked = []
    for request in requests:
        del request.shm_prompt_ids

        def attach(request=request):
            linked.append(request.request_id)
            request.shm_prompt_ids = SimpleNamespace(arr=np.arange(request.input_len, dtype=np.int64))

        request.link_prompt_ids_shm_array = attach
    candidate = queue(requests[:7])
    assert candidate._can_add_new_req(requests[7], True)
    assert linked == list(range(8))
