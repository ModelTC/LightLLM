import json
from dataclasses import asdict
import pytest

from lightllm.server.core.objs.req import Req
from lightllm.server.core.objs.sampling_params import SamplingParams
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.kv_cache_utils import compute_token_list_hash


@pytest.mark.parametrize("role", ["normal", "prefill", "decode"])
def test_hybrid_decode_cpu_hashes_use_complete_kv_pages_independently_of_radix(monkeypatch, local_req_memory, role):
    args = StartArgs(
        run_mode=role,
        enable_cpu_cache=True,
        linear_att_hash_page_size=64,
        linear_att_page_block_num=10000000 if role == "decode" else 2,
        cpu_cache_token_page_size=128,
    )
    monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(asdict(args)))
    get_env_start_args.cache_clear()
    monkeypatch.setattr("lightllm.server.core.objs.req.is_hybrid_att_model", lambda _: True)
    try:
        tokens = list(range(321))
        req = Req()
        req.init(1, tokens, {"max_new_tokens": 1}, None)
        assert req.hybrid_token_hash_list.get_all() == compute_token_list_hash(tokens, 64)
        expected_ends = [128, 256] if role == "decode" else [128, 256, 320]
        assert req.token_hash_page_len_list.get_all() == expected_ends
        expected_hashes = (
            compute_token_list_hash(tokens, 128)
            if role == "decode"
            else [req.hybrid_token_hash_list.get_all()[index] for index in (1, 3, 4)]
        )
        assert req.token_hash_list.get_all() == expected_hashes
    finally:
        get_env_start_args.cache_clear()


@pytest.mark.parametrize("role", ["normal", "prefill", "decode"])
@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("prefill_first", [False, True])
@pytest.mark.parametrize(
    "input_len,complete_ends,hybrid_ends",
    [
        (1, [], []),
        (128, [], [64]),
        (129, [128], [128]),
        (130, [128], [128]),
        (256, [128], [128, 192]),
        (257, [128, 256], [128, 256]),
        (258, [128, 256], [128, 256]),
        (385, [128, 256, 384], [128, 256, 384]),
        (386, [128, 256, 384], [128, 256, 384]),
    ],
)
def test_initialized_cpu_hashes_back_off_one_page_without_changing_radix(
    monkeypatch, local_req_memory, role, hybrid, prefill_first, input_len, complete_ends, hybrid_ends
):
    args = StartArgs(
        run_mode=role,
        enable_cpu_cache=True,
        cpu_cache_token_page_size=128,
        linear_att_hash_page_size=64,
        linear_att_page_block_num=2,
    )
    monkeypatch.setattr("lightllm.server.core.objs.req.get_env_start_args", lambda: args)
    monkeypatch.setattr("lightllm.server.core.objs.req.is_hybrid_att_model", lambda _: hybrid)
    params = SamplingParams()
    params.init(tokenizer=None, max_new_tokens=1)
    params.pd_prefill_first = prefill_first
    tokens = list(range(input_len))
    req = local_req_memory()
    req.init(1, tokens, params, None)
    radix_hashes = compute_token_list_hash(tokens, 64)
    expected_ends = hybrid_ends if hybrid and role != "decode" else complete_ends
    if role == "decode" and hybrid and input_len in (129, 257, 385):
        expected_ends = expected_ends[:-1]
    assert req.token_hash_page_len_list.get_all() == expected_ends
    assert req.token_hash_list.get_all() == [radix_hashes[end // 64 - 1] for end in expected_ends]
    assert req.token_hash_list.size == req.token_hash_page_len_list.size
    assert req.hybrid_token_hash_list.get_all() == (radix_hashes if hybrid else [])
    assert req.cpu_cache_match_page_indexes.is_empty()
