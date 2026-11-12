import json

import pytest

from .runtime import runtime

from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.utils import envs_utils


@pytest.fixture(autouse=True)
def native_runtime(monkeypatch):
    args = StartArgs()
    args.hardware_platform = "cuda"
    args.mtp_step = 0
    args.enable_ep_moe = False
    args.auto_update_redundancy_expert = False
    args.enable_tpsp_mix_mode = False
    monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(vars(args)))
    for name, value in {
        "LIGHTLLM_DP_WORLD_SIZE": 1,
        "LIGHTLLM_CURRENT_RANK_IN_DP": 0,
        "LIGHTLLM_CURRENT_DEVICE_ID": 0,
        "LIGHTLLM_GLOBAL_RANK": 0,
        "LIGHTLLM_GLOBAL_WORLD_SIZE": 1,
    }.items():
        monkeypatch.setenv(name, str(value))
    envs_utils.get_env_start_args.cache_clear()
    envs_utils.get_added_mtp_kv_layer_num.cache_clear()
    envs_utils.get_redundancy_expert_num.cache_clear()
    envs_utils.get_redundancy_expert_ids.cache_clear()
    yield
    envs_utils.get_env_start_args.cache_clear()
    envs_utils.get_added_mtp_kv_layer_num.cache_clear()
    envs_utils.get_redundancy_expert_num.cache_clear()
    envs_utils.get_redundancy_expert_ids.cache_clear()


@pytest.fixture
def config():
    return dict(
        hidden_size=32,
        n_embed=32,
        num_hidden_layers=4,
        n_layer=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        full_attention_interval=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        hc_count=2,
        hc_lowrank=8,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        output_gate_type="sigmoid",
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        num_experts=2,
        num_experts_per_tok=1,
        norm_topk_prob=True,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=8,
        indexer_compress_ratio=4,
        ple_layer_ids=[2],
        ple_embed_dim=32,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=5,
        make_ngram_vocab_size_divisible_by=4,
        split_ngram_parts=4,
        eos_token_id=63,
        vocab_size=64,
        partial_rotary_factor=0.5,
        rope_scaling={"mrope_section": [1, 1, 0]},
        tie_word_embeddings=False,
    )
