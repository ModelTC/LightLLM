from lightllm.common.attention_state.storage import allocate_state_storage
from dataclasses import replace

import pytest
import torch

from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.attention_state.full_att import GQAAttentionConfig, create_gqa_state


@pytest.mark.parametrize("kv_type", ["None", "int4kv", "int8kv", "fp8kv_spt", "fp8kv_sph"])
def test_gqa_layouts_coexist_and_match_across_tp_and_registration_order(kv_type, pd_args):
    config = GQAAttentionConfig(logical_layer_groups=((0,),), global_head_num=8, head_dim=128, dtype=torch.bfloat16)
    configs = [
        config,
        replace(config, logical_layer_groups=((1,),), global_head_num=16),
        replace(config, logical_layer_groups=((2,),), head_dim=256),
    ]
    source, target = (AttentionStateManager(request_capacity=0) for _ in range(2))
    for manager, items, tp in ((source, configs, 1), (target, reversed(configs), 2)):
        for item in items:
            state = create_gqa_state(replace(item, tp_rank=tp - 1, tp_world_size=tp), kv_type=kv_type, device="meta")
            manager.register_state("main", state)
        manager.finish_state_registration()
    states = source.get_full_attention_states("main")
    assert len({state.state_name for state in states}) == 3
    for layer in range(3):
        state = source.get_state("main", "gqa", layer)
        peer = target.get_state("main", "gqa", layer)
        assert state.state_name == peer.state_name
        assert state.config.global_head_num == configs[layer].global_head_num
        assert state.config.head_dim == configs[layer].head_dim
    for getter in (
        lambda m: m.get_cpu_cache_operator().get_page_layouts(),
        lambda m: m.get_pd_operator().get_pd_full_att_page_layouts(),
    ):
        layouts, peer_layouts = getter(source), getter(target)
        assert set(layouts) == set(peer_layouts) == {state.state_name for state in states}
        ordered = sorted(layouts.values(), key=lambda layout: layout.byte_start)
        assert all(a.byte_end <= b.byte_start for a, b in zip(ordered, ordered[1:]))


def test_fp8_bindings_keep_calibration_with_each_head_layout():
    from lightllm.common.basemodel.attention.base_att import BaseAttBackend
    from unit_tests.common.basemodel.attention.test_binding import bind_test_model

    class Fp8Backend(BaseAttBackend):
        supported_kv_encodings = ("fp8_e4m3fn_per_head",)

    states = []
    for layer, heads in enumerate((2, 4)):
        config = GQAAttentionConfig(
            logical_layer_groups=((layer,),), global_head_num=heads, head_dim=128, dtype=torch.bfloat16
        )
        state = create_gqa_state(config, kv_type="fp8kv_sph", device="cpu")
        state.bind_logical_calibration(
            {
                "quant_type": "per_head",
                "layers": {
                    "main": {str(layer): {"k": list(range(1, heads + 1)), "v": list(range(heads + 1, 2 * heads + 1))}}
                },
            },
            "main",
        )
        allocate_state_storage(state, 8)
        states.append(state)
    model = bind_test_model(states, backend=Fp8Backend(object()))
    for layer, state in zip(model.layers_infer, states):
        torch.testing.assert_close(layer.attention.control.kv_scales, state.scales[0])
        assert layer.attention.format.local_heads == state.config.head_num
