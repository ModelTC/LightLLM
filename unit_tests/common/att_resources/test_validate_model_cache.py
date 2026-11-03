from lightllm.common.attention_state.storage import allocate_state_storage
from types import SimpleNamespace

import pytest

from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.attention_state.full_att import create_compressed_att_state
from lightllm.models.deepseek_v4.config import DSV4AttentionConfig
from test.att_resources import validate_model_cache


@pytest.mark.parametrize("draft_count,bad_role", [(0, None), (2, None), (2, "main"), (2, "draft1")])
def test_validator_checks_all_compressed_states(monkeypatch, draft_count, bad_role):
    resources = AttentionStateManager(request_capacity=0)
    resources.full_att_allocator = SimpleNamespace(size=512)
    req_manager = object()
    models = []
    for index, role in enumerate(["main"] + [f"draft{i}" for i in range(draft_count)]):
        config = DSV4AttentionConfig(
            logical_layer_ids=(index * 2, index * 2 + 1),
            compress_ratios=(4, 128),
            head_dim=512,
            rope_head_dim=64,
            indexer_head_dim=128,
            sliding_window=128,
        )
        for ratio in (4, 128):
            state = create_compressed_att_state(config.create_compressed_config(ratio), device="meta")
            allocate_state_storage(state, 256 if role == bad_role and ratio == 4 else 512)
            resources.register_state(role, state)
        models.append(SimpleNamespace(att_state_manager=resources, att_state_role=role, req_manager=req_manager))
    resources.finish_state_registration()
    monkeypatch.setattr(validate_model_cache, "StaticBenchmarkExecutor", lambda **kwargs: None)
    monkeypatch.setattr(validate_model_cache, "TokenSource", lambda args: None)
    # Stop at input validation after checking resources, before any GPU execution.
    args = SimpleNamespace(input_len=255, cpu_cache_token_page_size=256)
    if bad_role is not None:
        with pytest.raises(AssertionError):
            validate_model_cache.validate(models[0], args, models[1:])
    else:
        with pytest.raises(ValueError, match="input_len must contain complete CPU pages"):
            validate_model_cache.validate(models[0], args, models[1:])
