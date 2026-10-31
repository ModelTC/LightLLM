import pytest

from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.attention_state.full_att import FullAttentionState
from lightllm.common.attention_state.bounded_att import BoundedAttentionState
from lightllm.models.qwen3next.config import Qwen3NextResourceConfig
from lightllm.models.qwen3next.model import Qwen3NextTpPartModel
from lightllm.models.qwen3_5_mtp.model import Qwen3_5MTPModel
from lightllm.server.core.objs.start_args_type import StartArgs


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
def test_hybrid_layer_mapping_and_mtp_states(
    layer_types: tuple[str, ...],
    expected_indices: tuple[int, ...],
    linear_layers: int,
    draft_layers: int,
) -> None:
    model_config = dict(
        num_hidden_layers=len(layer_types),
        hidden_size=128,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=128,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        full_attention_interval=4,
        layer_types=layer_types,
    )
    args = StartArgs(data_type="bfloat16", linear_att_ssm_data_type="float32")
    config = Qwen3NextResourceConfig.from_model_config(model_config, args)
    logical_layer_ids = tuple(range(len(layer_types)))
    (full_state,) = Qwen3NextTpPartModel.create_full_att_states(
        model_config=model_config, args=args, model_logical_layer_ids=logical_layer_ids,
    )
    (bounded_state,) = Qwen3NextTpPartModel.create_bounded_att_states(
        model_config=model_config, args=args, model_logical_layer_ids=logical_layer_ids, device="meta",
    )
    manager = AttentionStateManager(request_capacity=1)
    manager.register_state(model_role="main", state=full_state)
    manager.register_state(model_role="main", state=bounded_state)
    # draft 各自创建独立 State，不再把其层数和物理行追加到 main 的配置中。
    for draft_index in range(draft_layers):
        (state,) = Qwen3_5MTPModel.create_full_att_states(
            model_config=model_config,
            args=args,
            model_logical_layer_ids=(len(layer_types) + draft_index,),
        )
        manager.register_state(model_role=f"draft{draft_index}", state=state)
    manager.finish_state_registration()

    assert config.all_layer_num == len(layer_types)
    full_layers = len(layer_types) - linear_layers
    assert config.get_full_att_layer_indices() == tuple(
        layer for layer, kind in enumerate(layer_types) if kind != "linear_attention"
    )
    assert len(full_state.config.logical_layer_groups) == full_layers
    assert bounded_state.conv_state.shape[0] == bounded_state.ssm_state.shape[0] == linear_layers
    assert sum(len(state.config.logical_layer_groups) for state in manager.get_full_attention_states()) == (
        full_layers + draft_layers
    )
    for layer, (kind, expected) in enumerate(zip(layer_types, expected_indices)):
        is_linear = kind == "linear_attention"
        state = manager.get_state(
            model_role="main",
            state_type_filter="",
            logical_layer_id=layer,
            state_class=BoundedAttentionState if is_linear else FullAttentionState,
        )
        if is_linear:
            conv, ssm = state.get_layer_view(logical_layer_id=layer)
            assert conv.storage_offset() == expected * state.conv_state.stride(0)
            assert ssm.storage_offset() == expected * state.ssm_state.stride(0)
        else:
            assert state.get_storage_layer_index(logical_layer_id=layer) == expected
    for draft in range(draft_layers):
        state = manager.get_state(
            model_role=f"draft{draft}",
            state_type_filter="gqa",
            logical_layer_id=len(layer_types) + draft,
        )
        assert state is not full_state
        assert state.get_storage_layer_index(logical_layer_id=len(layer_types) + draft) == 0
