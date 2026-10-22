from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

# Follow the production import order to avoid the existing package initialization cycle.
from lightllm.server.router.model_infer.mode_backend import generic_pre_process  # noqa: F401
from lightllm.server.core.objs.sampling_params import SamplingParams
from lightllm.server.router.model_infer import infer_batch
from lightllm.server.router.model_infer.infer_batch import InferReq, InferSamplingParams
from lightllm.server.router.model_infer.mode_backend import base_backend
from unit_tests.common.cpu_index_helpers import make_cpu_index_manager, bind_cpu_indexes


def _make_req_with_cached_prefix(monkeypatch, prompt_logprobs, disable_prompt_cache=False, hybrid=False):
    sample_params = SamplingParams()
    sample_params.top_k = -1
    sample_params.prompt_logprobs = prompt_logprobs
    sample_params.disable_prompt_cache = disable_prompt_cache
    req = InferReq.__new__(InferReq)
    req.req_idx = 0
    req.args = SimpleNamespace(
        page_size=2, linear_att_hash_page_size=2, linear_att_page_block_num=2, max_req_total_len=16
    )
    req.cur_kv_len = req.hold_kv_len = 0
    req.shared_kv_node = req.tail_small_page_buffer_id = None
    req.shm_req = SimpleNamespace(
        sample_params=sample_params,
        input_len=6,
        prompt_cache_len=0,
        shm_prompt_ids=SimpleNamespace(arr=torch.arange(6).numpy()),
        hybrid_token_hash_list=SimpleNamespace(size=2, get_all=lambda: [10, 20]),
    )
    req.sampling_param = InferSamplingParams(req.shm_req, vocab_size=8)
    req.get_cur_total_len = lambda: 6
    node = SimpleNamespace(node_prefix_total_len=4, big_page_buffer_idx=0, is_big_page_node=lambda: True)
    match_prefix = Mock(return_value=(node, 4, torch.arange(4, dtype=torch.int32)))
    manager = make_cpu_index_manager(
        token_table=torch.zeros((1, 8), dtype=torch.int32),
        restore_big_page_state=Mock(),
        init_hybrid_attention_state=Mock(),
        page_size=2,
    )
    bind_cpu_indexes(req, manager)
    monkeypatch.setattr(
        infer_batch,
        "g_infer_context",
        SimpleNamespace(
            is_hybrid_att_model=hybrid, req_manager=manager, radix_cache=SimpleNamespace(match_prefix=match_prefix)
        ),
    )
    return req, match_prefix


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize(
    "prompt_logprobs, disable_prompt_cache, expected_prefix_len",
    [(-1, False, 4), (0, False, 0), (3, False, 0), (-1, True, 0)],
)
def test_prompt_logprobs_requests_recompute_cached_prefix(
    monkeypatch, hybrid, prompt_logprobs, disable_prompt_cache, expected_prefix_len
):
    req, match_prefix = _make_req_with_cached_prefix(monkeypatch, prompt_logprobs, disable_prompt_cache, hybrid)

    if hybrid:
        req._hybrid_match_radix_cache()
    else:
        req._match_radix_cache()

    assert req.cur_kv_len == req.hold_kv_len == expected_prefix_len
    assert req.shm_req.prompt_cache_len == expected_prefix_len
    assert req.shm_req.shm_cur_kv_len == expected_prefix_len
    assert match_prefix.call_count == (1 if expected_prefix_len else 0)


@pytest.mark.parametrize("prompt_logprobs", [0, 3])
def test_cached_prompt_still_captures_logprobs_for_every_prompt_position(monkeypatch, prompt_logprobs):
    req, _ = _make_req_with_cached_prefix(monkeypatch, prompt_logprobs)
    req._match_radix_cache()
    req.prompt_selected_logprobs = SimpleNamespace(add_chunk=Mock())
    backend = base_backend.ModeBackend.__new__(base_backend.ModeBackend)
    backend.disable_chunked_prefill = True
    backend.is_master_in_dp = True
    q_len = req.get_cur_total_len() - req.cur_kv_len
    indexes = torch.arange(q_len, dtype=torch.int32)
    backend.model = SimpleNamespace(return_all_prompt_logics=True, _select_mem_indexes=lambda _: indexes)
    capture_manager = SimpleNamespace(is_buffer_initialized=lambda: True, capture=Mock())
    monkeypatch.setattr(base_backend.PromptLogprobsCaptureManager, "get_instance", lambda: capture_manager)
    logits = torch.arange(q_len * 8, dtype=torch.float32).reshape(q_len, 8)

    backend._capture_prompt_logprobs_if_needed(model_input=None, run_reqs=[req], prompt_logits=logits)

    if prompt_logprobs == 0:
        req.prompt_selected_logprobs.add_chunk.assert_called_once()
        start, end, logprobs, ranks = req.prompt_selected_logprobs.add_chunk.call_args.args
        assert (start, end) == (1, req.shm_req.input_len)
        assert logprobs.shape == ranks.shape == (5,)
        expected = logits[:5].log_softmax(dim=-1)[torch.arange(5), torch.arange(1, 6)]
        torch.testing.assert_close(logprobs, expected)
    else:
        capture_manager.capture.assert_called_once()
        captured = capture_manager.capture.call_args.kwargs
        assert captured["mem_indexes"].tolist() == list(range(5))
        assert captured["top_token_ids"].shape == captured["top_logprobs"].shape == (5, prompt_logprobs)


@pytest.mark.parametrize("kind", ["prompt_logprobs", "routed_experts"])
def test_final_metadata_extracts_owned_token_indexes_through_req_helper(monkeypatch, kind):
    from lightllm.server.router.model_infer import infer_req_ext

    req, _ = _make_req_with_cached_prefix(monkeypatch, prompt_logprobs=3)
    req.cur_kv_len = 6
    req.cur_output_len = 0
    req.token_index.append_token_indexes(torch.tensor([8, 9, 2, 3, 14, 15], dtype=torch.int32))
    manager = SimpleNamespace(extract=Mock(return_value="metadata"))
    metadata = infer_req_ext.FinalTokenMetadataExt(req)
    if kind == "prompt_logprobs":
        monkeypatch.setattr(infer_req_ext.PromptLogprobsCaptureManager, "get_instance", lambda: manager)
        result = metadata.collect_prompt_logprobs()
    else:
        monkeypatch.setattr(infer_req_ext.MoeRouteInfoManager, "get_instance", lambda: manager)
        result = metadata.collect_routed_experts()
    assert result == "metadata"
    indexes = manager.extract.call_args.args[0]
    assert indexes.tolist() == [8, 9, 2, 3, 14]
    req.token_index.reset()
    req.token_index.append_token_indexes(torch.arange(6, dtype=torch.int32))
    assert indexes.tolist() == [8, 9, 2, 3, 14]
