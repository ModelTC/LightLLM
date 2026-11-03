from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.attention_state.full_att import GQAAttentionConfig, create_gqa_state


@pytest.mark.parametrize("kv_type", ["None", "int4kv", "int8kv", "fp8kv_spt", "fp8kv_sph"])
def test_gqa_layouts_coexist_and_match_across_tp_and_registration_order(kv_type, pd_args, monkeypatch):
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
    copied = []
    for state in states:
        monkeypatch.setattr(
            state,
            "copy_tokens_from_dp",
            lambda *, remote_dp_full_att_state, **_: copied.append(remote_dp_full_att_state),
        )
    source.get_dp_shared_operator().copy_from_dp(
        remote_att_manager=target,
        src_token_indices=torch.tensor([0]),
        dst_token_indices=torch.tensor([1]),
        src_big_page_ids=[],
        dst_big_page_ids=[],
        src_req_idx=0,
        dst_req_idx=1,
    )
    assert copied == [getattr(target, state.state_name) for state in states]


@pytest.mark.parametrize("prefill", [False, True])
def test_fa3_prepares_descales_for_each_head_layout(prefill, monkeypatch):
    from lightllm.common.basemodel.attention.fa3.fp8 import (
        Fp8Fa3PrefillAttState,
        Fp8Fa3DecodeAttState,
        Fa3PrefillAttState,
        Fa3DecodeAttState,
    )

    manager = AttentionStateManager(request_capacity=2)
    for layer, heads in enumerate((2, 4)):
        config = GQAAttentionConfig(
            logical_layer_groups=((layer,),), global_head_num=heads, head_dim=128, dtype=torch.bfloat16
        )
        state = create_gqa_state(config, kv_type="fp8kv_sph", device="cpu")
        state.scales = torch.arange(1, 2 * heads + 1, dtype=torch.float32).view(1, -1)
        manager.register_state("main", state)
    manager.finish_state_registration()
    infer = SimpleNamespace(att_state_manager=manager, att_state_role="main", batch_size=2)
    cls = Fp8Fa3PrefillAttState if prefill else Fp8Fa3DecodeAttState
    parent = Fa3PrefillAttState if prefill else Fa3DecodeAttState
    monkeypatch.setattr(parent, "init_state", lambda self: None)
    att = cls(backend=None, _infer_state_weak_ref=lambda: infer)
    att.b_att_seq_len = torch.ones(2, dtype=torch.int32)
    att.init_state()
    for state in manager.get_full_attention_states("main"):
        heads = state.config.head_num
        torch.testing.assert_close(state.get_k_descale(0, 2), state.scales[0, :heads].expand(2, heads))
        torch.testing.assert_close(state.get_v_descale(0, 2), state.scales[0, heads:].expand(2, heads))
