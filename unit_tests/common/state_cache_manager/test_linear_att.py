import pytest
import torch

from lightllm.common.state_cache_manager import LinearAttCacheConfig


@pytest.mark.parametrize("draft_layers", [0, 1, 3])
@pytest.mark.parametrize(
    "layer_types,expected_indices,linear_layers",
    [
        (
            ("linear_attention", "linear_attention", "linear_attention", "full_attention") * 2,
            (0, 1, 2, 0, 3, 4, 5, 1),
            6,
        ),
        (
            (
                "deepseek_sparse_attention",
                "linear_attention",
                "linear_attention",
                "deepseek_sparse_attention",
                "linear_attention",
            ),
            (0, 0, 1, 1, 2),
            3,
        ),
        (
            ("linear_attention", "full_attention", "linear_attention") * 2 + ("full_attention",),
            (0, 0, 1, 2, 1, 3, 2),
            4,
        ),
    ],
)
def test_hybrid_layer_mapping_and_mtp_tail(layer_types, expected_indices, linear_layers, draft_layers):
    config = LinearAttCacheConfig(
        tp_world_size=1,
        full_att_all_num_kv_heads=1,
        full_att_dtype=torch.bfloat16,
        full_att_num_kv_heads=1,
        full_att_head_dim=128,
        global_linear_k_heads=1,
        global_linear_v_heads=1,
        num_linear_k_heads=1,
        num_linear_v_heads=1,
        head_linear_k_dim=128,
        head_linear_v_dim=128,
        conv_kernel_size=4,
        conv_state_dtype=torch.bfloat16,
        ssm_state_dtype=torch.float32,
        layer_types=layer_types,
        draft_full_att_kv_layer_num=draft_layers,
    )
    assert config.all_layer_num == len(layer_types)
    assert config.linear_layer_num == linear_layers
    full_layers = len(layer_types) - linear_layers
    assert config.get_main_model_full_att_layer_num() == full_layers
    assert config.get_full_att_kv_layer_num_with_draft_model() == full_layers + draft_layers
    for layer, (kind, expected) in enumerate(zip(layer_types, expected_indices)):
        get_index = (
            config.get_linear_state_layer_index if kind == "linear_attention" else config.get_full_att_kv_layer_index
        )
        assert get_index(layer) == expected
    for draft in range(draft_layers):
        assert config.get_full_att_kv_layer_index(len(layer_types) + draft) == full_layers + draft
