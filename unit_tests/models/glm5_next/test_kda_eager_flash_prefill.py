"""GLM-5-Next KDA layer prefill outside CUDA graphs with FlashKDA.

The layer splits the fused qkv/beta/gate projection into column views, so the beta logits reach FlashKDA as a
strided tensor, which its kernel rejects. The CUDA graph wrapper makes them contiguous; the eager path must too.
"""

import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from lightllm.server.core.objs.start_args_type import StartArgs

H, D, HIDDEN, LAYER = 2, 128, 256, 0


def _flash_kda_usable():
    try:
        import flash_kda  # noqa: F401
    except Exception:
        return False
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


pytestmark = pytest.mark.skipif(not _flash_kda_usable(), reason="FlashKDA on SM90 required")


def _model(requested):
    from lightllm.common.attention import selector

    model = type("Model", (), {})()
    model.args = StartArgs(llm_prefill_att_backend=["auto", requested], linear_att_ssm_data_type="bfloat16")
    model.data_type = torch.bfloat16

    def select_impls(prefill_config, decode_config):
        model.prefill = selector.select(prefill_config, "prefill", requested)
        return model.prefill.impl, selector.select(decode_config, "decode", "triton").impl

    model.select_impls = select_impls
    return model


def _layer_weight():
    def w(*shape, scale):
        return (torch.randn(*shape, device="cuda") * scale).bfloat16()

    projection = w(HIDDEN, 3 * H * D + H + D + D, scale=HIDDEN**-0.5)
    decay, output_gate = w(D, H * D, scale=D**-0.5), w(D, H * D, scale=D**-0.5)
    conv, out = w(3 * H * D, 4, scale=0.3), w(H * D, HIDDEN, scale=(H * D) ** -0.5)
    return SimpleNamespace(
        linear_qkvbfg_a_proj=SimpleNamespace(mm=lambda x: x @ projection),
        project_kda_fg_b=lambda d, o: (d @ decay, o @ output_gate),
        get_merged_kda_conv_weight=lambda: conv,
        linear_A_log=SimpleNamespace(weight=torch.randn(H, device="cuda") * 0.5),
        linear_dt_bias=SimpleNamespace(weight=torch.randn(H * D, device="cuda") * 0.5),
        linear_o_norm=lambda input, gate_value, eps, alloc_func: input.view(gate_value.shape) * gate_value.sigmoid(),
        linear_o_proj=SimpleNamespace(mm=lambda x: x @ out),
    )


def _layer(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(asdict(StartArgs())))
    from lightllm.models.glm5_next.layer_infer.transformer_layer_infer import Glm5NextTransformerLayerInfer

    config = dict(
        rms_norm_eps=1e-6,
        hidden_size=HIDDEN,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        swiglu_limit=None,
        n_routed_experts=None,
        first_k_dense_replace=0,
        n_shared_experts=0,
        num_experts_per_tok=0,
        norm_topk_prob=False,
        n_group=1,
        topk_group=1,
        linear_attn_config={"num_heads": H, "head_dim": D, "short_conv_kernel_size": 4},
    )
    layer = Glm5NextTransformerLayerInfer(LAYER, config)
    assert layer.is_linear_attention_layer
    return layer


def _prefill(layer, requested, hidden, layer_weight, conv0, ssm0):
    from lightllm.common.attention_state import AttentionStateManager
    from lightllm.common.attention_state.bounded_att import ConvSSMStateConfig
    from lightllm.common.attention_state.bounded_att.kda.create_kda_state import create_kda_state
    from lightllm.common.basemodel.attention_runtime import build_attention_metadata
    from lightllm.common.basemodel.infer_struct import InferStateInfo
    from unit_tests.models.glm5_next.kda_test_utils import prepare_prefill_infer

    state = create_kda_state(ConvSSMStateConfig((LAYER,), H, H, D, D, 3, torch.bfloat16, torch.bfloat16), 4)
    conv, ssm = state.get_layer_view(logical_layer_id=LAYER)
    conv.copy_(conv0)
    ssm.copy_(ssm0)
    manager = AttentionStateManager(request_capacity=4)
    manager.register_state("main", state)
    manager.finish_state_registration()

    lengths, ready = [5, 11], [0, 3]
    infer = InferStateInfo()
    infer.att_state_manager, infer.att_state_role = manager, "main"
    infer.b_req_idx = torch.tensor([2, 0], device="cuda", dtype=torch.int32)
    lens = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    infer.b1_cu_q_seq_len = torch.nn.functional.pad(lens.cumsum(0, dtype=torch.int32), (1, 0))
    infer.b_ready_cache_len = torch.tensor(ready, device="cuda", dtype=torch.int32)
    infer.max_q_seq_len = max(lengths)
    infer.req_manager = SimpleNamespace(req_to_token_indexs=None)
    prepare_prefill_infer(infer)
    model = _model(requested)
    layer.init_attention(model, model.select_impls)
    # The attention state the layer's group builds for this step.
    builder = model.prefill.backend.create_step_builder(layer.prefill_impl.layer_config, None)
    att_state = builder.build(build_attention_metadata(infer, model))
    infer.att_states = {layer.prefill_impl: att_state}
    assert (att_state.flash_kda_fwd is not None) == (requested == "flashkda")
    output = layer.context_attention_forward(hidden, infer, layer_weight)
    return output, ssm


def test_eager_flashkda_prefill_matches_triton(monkeypatch):
    torch.manual_seed(5)
    layer = _layer(monkeypatch)
    layer_weight = _layer_weight()
    hidden = torch.randn(16, HIDDEN, device="cuda").bfloat16()
    conv0 = (torch.randn(4, 3 * H * D, 3, device="cuda") * 0.5).bfloat16()
    ssm0 = (torch.randn(4, H, D, D, device="cuda") * 0.05).bfloat16()
    ssm0[2].zero_()  # request 2 starts fresh

    expected, expected_ssm = _prefill(layer, "triton", hidden, layer_weight, conv0, ssm0)
    actual, actual_ssm = _prefill(layer, "flashkda", hidden, layer_weight, conv0, ssm0)

    scale = expected.float().abs().max()
    assert (actual.float() - expected.float()).abs().max() < 3e-2 * scale
    torch.testing.assert_close(actual_ssm.float(), expected_ssm.float(), atol=2e-2, rtol=5e-2)
