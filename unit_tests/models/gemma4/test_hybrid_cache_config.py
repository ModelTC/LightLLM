import json
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.state_cache_manager import SlidingWindowCacheConfig, get_hybrid_cache_config
from lightllm.utils import config_utils, envs_utils


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("tp_world_size", [1, 2])
def test_gemma_hybrid_cache_factory_preserves_shared_owners_and_cpu_page_bytes(
    monkeypatch, tmp_path, wrapped, tp_world_size
):
    text_config = {
        "model_type": "gemma4_text",
        "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        "num_kv_shared_layers": 2,
        "num_key_value_heads": 4,
        "num_global_key_value_heads": 2,
        "sliding_window": 8,
        "head_dim": 4,
        "global_head_dim": 8,
    }
    config = {"model_type": "gemma4", "text_config": text_config} if wrapped else text_config
    (tmp_path / "config.json").write_text(json.dumps(config))
    args = SimpleNamespace(
        model_dir=str(tmp_path),
        tp=tp_world_size * 2,
        dp=2,
        linear_att_hash_page_size=8,
        linear_att_page_block_num=2,
        cpu_cache_token_page_size=16,
    )
    monkeypatch.setattr(envs_utils, "get_env_start_args", lambda: args)
    monkeypatch.setattr(envs_utils, "get_llm_data_type", lambda: torch.bfloat16)

    assert config_utils.is_hybrid_att_model(args.model_dir)
    layout = get_hybrid_cache_config()
    assert isinstance(layout, SlidingWindowCacheConfig)
    assert layout.sliding_layer_to_cache_index == {0: 0, 2: 0}
    assert layout.full_layer_to_cache_index == {1: 0, 3: 0}
    assert layout.sliding_head_num == 4 // tp_world_size
    assert layout.full_head_num == 2 // tp_world_size
    # Across all TP ranks: 1024 bytes of full KV plus 512 bytes of window checkpoint.
    assert layout.get_cpu_cache_big_page_bytes() == 1536
    assert layout.get_cpu_cache_big_page_bytes(16, tp_world_size) == 1536
