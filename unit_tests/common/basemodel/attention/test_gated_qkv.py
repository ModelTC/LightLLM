from types import SimpleNamespace

import torch

from lightllm.models.qwen3next.layer_infer import transformer_layer_infer as module


def test_qwen_next_qkv_split_matches_projection_with_separate_gate(monkeypatch):
    layer = module.Qwen3NextTransformerLayerInfer.__new__(module.Qwen3NextTransformerLayerInfer)
    layer.tp_q_head_num_, layer.tp_k_head_num_, layer.tp_v_head_num_ = 4, 2, 2
    layer.head_dim_, layer.embed_dim_, layer.eps_, layer.partial_rotary_factor = 8, 32, 1e-6, 0.25
    layer._tpsp_allgather = lambda input, infer_state: input
    projected = torch.arange(3 * 96, dtype=torch.float32).view(3, 96)
    weight = SimpleNamespace(
        qkvo_gate_proj=SimpleNamespace(mm=lambda x: projected),
        qk_norm_weight_=lambda *args, **kwargs: None,
    )
    infer = SimpleNamespace(position_cos=None, position_sin=None, need_dp_prefill_balance=False)
    monkeypatch.setattr(module, "rotary_emb_fwd", lambda *args, **kwargs: None)
    query, kv = layer._get_qkv(torch.zeros(3, 32), infer, weight)
    torch.testing.assert_close(query, projected[:, :32])
    torch.testing.assert_close(kv, projected[:, 32:64].view(3, 4, 8))
    torch.testing.assert_close(infer.gate_logics_value, projected[:, 64:])
