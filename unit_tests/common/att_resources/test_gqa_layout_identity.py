from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.attention_state.full_att import GQAAttentionConfig, create_gqa_state
from lightllm.common.basemodel.attention.base_att import AttControl


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
def test_fa3_reads_descale_for_each_head_layout(prefill, monkeypatch):
    from lightllm.common.basemodel.attention.fa3 import fp8 as fa3_fp8

    manager = AttentionStateManager(request_capacity=2)
    for layer, heads in enumerate((2, 4)):
        config = GQAAttentionConfig(
            logical_layer_groups=((layer,),), global_head_num=heads, head_dim=128, dtype=torch.bfloat16
        )
        state = create_gqa_state(config, kv_type="fp8kv_sph", device="cpu")
        state.scales = torch.arange(1, 2 * heads + 1, dtype=torch.float32).view(1, -1)
        state.q_scales = torch.arange(1, heads + 1, dtype=torch.float32).view(1, heads)
        manager.register_state("main", state)
    manager.finish_state_registration()
    # prefill 必须按 b_seq_len 取 batch，不能读这个未设置的请求 batch。
    infer = SimpleNamespace(
        att_state_manager=manager,
        att_state_role="main",
        batch_size=None,
        b_seq_len=torch.ones(2, dtype=torch.int32),
        max_q_seq_len=1,
    )
    cls = fa3_fp8.Fp8Fa3PrefillAttState if prefill else fa3_fp8.Fp8Fa3DecodeAttState
    parent = fa3_fp8.Fa3PrefillAttState if prefill else fa3_fp8.Fa3DecodeAttState
    monkeypatch.setattr(parent, "init_state", lambda self: None)
    monkeypatch.setattr(fa3_fp8, "q_per_head_static_fp8_quant", lambda q, scales: q)
    observed = {}

    def attend(**kwargs):
        observed.update(kwargs)
        return kwargs["q"]

    monkeypatch.setattr(fa3_fp8, "flash_attn_with_kvcache", attend)
    att = cls(backend=None, _infer_state_weak_ref=lambda: infer)
    att.b_att_seq_len = torch.ones(3, dtype=torch.int32)
    att.page_table = att.cu_seqlens_q = att.cu_seqlens_k = None
    att.decode_max_q_seq_len = 1
    att.causal = True
    att.init_state()
    batch_size = infer.b_seq_len.shape[0] if prefill else att.b_att_seq_len.shape[0]
    for state in manager.get_full_attention_states("main"):
        heads = state.config.head_num
        layer = state.config.logical_layer_groups[0][0]
        q = k = v = torch.zeros((2, heads, state.config.head_dim))
        observed.clear()
        run = att.prefill_att if prefill else att.decode_att
        run(q=q, k=k, v=v, att_control=AttControl(logical_layer_id=layer))
        row = state.scales[0]
        k_descale = observed["k_descale"]
        v_descale = observed["v_descale"]
        assert k_descale.data_ptr() == row[:heads].data_ptr()
        assert v_descale.data_ptr() == row[heads:].data_ptr()
        torch.testing.assert_close(k_descale, row[:heads].expand(batch_size, heads))
        torch.testing.assert_close(v_descale, row[heads:].expand(batch_size, heads))
