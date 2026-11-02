"""Tool round trips must retire waiting leases even when the assistant header changes."""
import asyncio
import uuid
from types import SimpleNamespace

import pytest
import torch
from sortedcontainers import SortedDict

from lightllm.server import build_prompt
from lightllm.server.core.objs import FinishStatus
from lightllm.server.core.objs.sampling_params import SamplingParams, ToolCallCacheMarkers
from lightllm.server.router.model_infer.infer_batch import InferenceContext
from lightllm.server.router.dynamic_prompt.radix_cache import RadixCache
from lightllm.server.router.dynamic_prompt.hybrid_att_radix_cache import HybridAttPagedRadixCache
from lightllm.utils import shm_utils


@pytest.fixture
def tree(monkeypatch):
    monkeypatch.setattr(shm_utils, "get_unique_server_name", lambda: f"boundary_{uuid.uuid4().hex}")
    result = RadixCache(40, 0)
    result.enable_tool_call_retention(10, 0.5)
    result.tool_call_retention.advance_clock(100)
    return result


def insert(tree, tokens):
    return tree.insert(torch.tensor(tokens, dtype=torch.int64))[1]


@pytest.mark.parametrize("tail", [0, 1, 8, 2048])
def test_prefix_hint_survives_shared_sampling_roundtrip_and_resets(tail):
    params = SamplingParams()
    params.init(None, tool_call_cache_markers=[[8, 9]], tool_call_cache_prompt_tail_tokens=tail)
    other = SamplingParams()
    other.init(None, **params.to_origin_dict())
    assert other.tool_call_cache_markers.prompt_tail_tokens == tail
    assert other.tool_call_cache_markers.to_list() == [[8, 9]]
    other.init(None)
    assert other.tool_call_cache_markers.prompt_tail_tokens == 0
    assert other.tool_call_cache_markers.to_list() == []


@pytest.mark.parametrize("tail", [-1, 2049, 0.5, True, "8"])
def test_invalid_internal_prefix_hint_is_rejected(tail):
    with pytest.raises(ValueError):
        ToolCallCacheMarkers().initialize([[8, 9]], prompt_tail_tokens=tail)


def test_roundtrip_header_rewrite_consumes_the_stable_input_lease(tree):
    # 1..4 is stable history; 5..6 is a generation-only header; 8..9
    # is the accepted tool call. The next chat template rewrites that header.
    tail = insert(tree, [1, 2, 3, 4, 5, 6, 8, 9])
    metadata = ToolCallCacheMarkers()
    metadata.initialize([[8, 9]], prompt_tail_tokens=2)
    req = SimpleNamespace(
        finish_status=FinishStatus(FinishStatus.FINISHED_STOP),
        finish_output_len=2,
        sampling_param=SimpleNamespace(
            shm_param=SimpleNamespace(tool_call_cache_markers=metadata), disable_prompt_cache=False
        ),
        shm_req=SimpleNamespace(input_len=6),
        get_input_token_ids=lambda: [1, 2, 3, 4, 5, 6, 8, 9],
    )
    InferenceContext(radix_cache=tree)._retain_tool_call_prefix(req, tail)
    retention = tree.tool_call_retention
    assert retention.protected_tokens == 4
    assert not tail.cache_protected
    matched, hit, _ = tree.match_prefix(torch.tensor([1, 2, 3, 4, 99, 100]), update_refs=True)
    assert hit == 4 and matched.ref_counter == 1
    assert retention.protected_tokens == 0
    assert retention.get_stats()["consumed_tokens"] == 4
    tree.dec_node_ref_counter(matched)
    tree.evict_tree_set._check()


def test_hint_does_not_protect_an_empty_stable_prefix(tree):
    tail = insert(tree, [1, 2, 3, 4])
    tree.tool_call_retention.protect(tail, reusable_prefix_len=0)
    assert tree.tool_call_retention.protected_tokens == 0


@pytest.mark.parametrize("suffix", ["", "assistant", "<assistant><think></think>"])
def test_prompt_hint_tokenizes_only_the_generation_suffix(monkeypatch, suffix):
    encoded = []

    def encode(text, **kwargs):
        encoded.append(text)
        return list(map(ord, text))

    tokenizer = SimpleNamespace(
        apply_chat_template=lambda **kwargs: "history",
        encode=encode,
    )
    monkeypatch.setattr(build_prompt, "tokenizer", tokenizer)
    assert build_prompt._cache_generation_tail_tokens({}, [], "history" + suffix) == len(suffix) + 1
    assert encoded == [suffix]


@pytest.mark.parametrize("variant", ["rewrite", "long_suffix", "exception"])
def test_unusual_templates_fall_back_without_breaking_requests(monkeypatch, variant):
    def render(**kwargs):
        if variant == "exception":
            raise ValueError("no history-only render")
        return "other history" if variant == "rewrite" else "history"

    tokenizer = SimpleNamespace(
        apply_chat_template=render, encode=lambda *args, **kwargs: pytest.fail("unexpected encode")
    )
    monkeypatch.setattr(build_prompt, "tokenizer", tokenizer)
    prompt = "history" + ("x" * 4097 if variant == "long_suffix" else "assistant")
    assert build_prompt._cache_generation_tail_tokens({}, [], prompt) == 0


@pytest.mark.parametrize("enabled", [False, True])
def test_build_prompt_keeps_disabled_render_path_unchanged(monkeypatch, enabled):
    from lightllm.server import api_openai

    renders = []

    def render(**kwargs):
        renders.append(kwargs["add_generation_prompt"])
        return "history<assistant>" if kwargs["add_generation_prompt"] else "history"

    monkeypatch.setattr(
        build_prompt, "tokenizer", SimpleNamespace(apply_chat_template=render, encode=lambda s, **k: list(s))
    )
    monkeypatch.setattr(build_prompt, "get_model_type_v1", lambda: "qwen3")
    monkeypatch.setattr(api_openai, "_is_force_thinking_mode", lambda _: False)
    request = SimpleNamespace(
        messages=[], character_settings=None, role_settings=None, reasoning_effort=None, chat_template_kwargs=None
    )
    result = asyncio.run(build_prompt.build_prompt(request, [], return_cache_hint=enabled))
    if enabled:
        assert result == ("history<assistant>", len("<assistant>") + 1)
        assert renders == [True, False]
    else:
        assert result == "history<assistant>"
        assert renders == [True]


def test_hybrid_stable_boundary_keeps_only_a_resumable_history_checkpoint(monkeypatch):
    monkeypatch.setattr(shm_utils, "get_unique_server_name", lambda: f"hybrid_boundary_{uuid.uuid4().hex}")
    buffers = SimpleNamespace(free_state_cache=lambda *args, **kwargs: None)
    tree = HybridAttPagedRadixCache(40, 0, 2, 2, SimpleNamespace(big_page_buffers=buffers), buffers)
    tree.enable_tool_call_retention(10, 0.5)
    _, tail = tree.insert(
        torch.arange(12),
        block_hashs=[11, 22, 33, 44, 55, 66],
        block_state_idxs=[None] * 6,
        len_to_big_page_id=SortedDict({4: 100, 8: 101, 12: 102}),
    )
    # Stable history ends at6; the checkpoint at8 contains a rewritten header.
    tree.tool_call_retention.protect(tail, reusable_prefix_len=6)
    assert not tail.cache_protected and not tail.parent.cache_protected
    assert tail.parent.parent.cache_protected
    assert tree.tool_call_retention.protected_tokens == 4
    node, length, _ = tree.match_prefix(torch.arange(4), block_hashs=[11, 22], update_refs=True)
    assert length == 4 and node.big_page_buffer_idx == 100
    assert tree.tool_call_retention.protected_tokens == 0
    tree.dec_node_ref_counter(node)
    tree._evict_tree_set._check()
