from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention.api import AttentionFamily, AttentionPhase
from lightllm.common.attention.api.inputs import DenseInputs
from lightllm.common.attention.backends.dense.fa3 import fp8 as fa3_fp8
from lightllm.common.attention.backends.dense.flashinfer import fp8 as flashinfer_fp8
from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.basemodel.attention import create_layer_attention_params
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.server.core.objs.start_args_type import StartArgs
from unit_tests.common.att_resources.model_helpers import rebuild_full_state, replace_logical_layer_groups
from unit_tests.common.att_resources.test_capacity import definition
from unit_tests.common.attention.helpers import make_backend, make_batch, make_context, make_layer_attention


@pytest.mark.parametrize("backend_name", ["fa3", "flashinfer"])
@pytest.mark.parametrize("operation", ["prefill", "decode"])
@pytest.mark.parametrize("role", ["main", "draft0"])
@pytest.mark.parametrize("logical_layer_id,storage_layer_index", [(3, 0), (7, 0), (8, 1), (11, 1)])
def test_fp8_backend_selects_calibration_from_logical_layer(
    monkeypatch: pytest.MonkeyPatch,
    backend_name: str,
    operation: str,
    role: str,
    logical_layer_id: int,
    storage_layer_index: int,
) -> None:
    template = definition("static_head" if backend_name == "fa3" else "static_tensor")
    state = rebuild_full_state(
        template=template,
        config=replace_logical_layer_groups(template.config, ((3, 7), (8, 11))),
        token_capacity=8,
        device="cpu",
    )
    state.scales[0].fill_(2)
    state.scales[1].fill_(5)
    expected_scale = 2 if storage_layer_index == 0 else 5
    manager = AttentionStateManager(request_capacity=1)
    manager.register_state(role, state)
    manager.finish_state_registration()
    phase = AttentionPhase(operation)
    cls = fa3_fp8.Fp8Fa3AttBackend if backend_name == "fa3" else flashinfer_fp8.Fp8FlashInferAttBackend
    backend = make_backend(cls)
    context = make_context(backend, make_batch((1,), phase=phase))
    context.page_table = context.cu_seqlens_q = context.cu_seqlens_k = None
    context.b_att_seq_len = context.batch.b_seq_len
    context.decode_max_q_seq_len = 1
    context.causal = True
    calls = []
    if backend_name == "fa3":
        state.q_scales[0].fill_(3)
        state.q_scales[1].fill_(7)
        expected_q_scale = 3 if storage_layer_index == 0 else 7

        def quantize(q, scales):
            assert torch.all(scales == expected_q_scale)
            return q

        def attend(**kwargs):
            assert torch.all(kwargs["k_descale"] == expected_scale)
            assert torch.all(kwargs["v_descale"] == expected_scale)
            assert torch.all(kwargs["q_descale"] == expected_q_scale)
            calls.append(operation)
            return kwargs["q"]

        monkeypatch.setattr(fa3_fp8, "q_per_head_static_fp8_quant", quantize)
        monkeypatch.setattr(fa3_fp8, "flash_attn_with_kvcache", attend)
    else:
        state.cpu_scales = state.scales.cpu().tolist()

        def run(q, paged_kv_cache, k_scale, v_scale, out):
            assert k_scale == v_scale == expected_scale
            calls.append(operation)
            out.copy_(q)

        context.prefill_wrapper = context.decode_wrapper = SimpleNamespace(run=run)
    layer = LlamaTransformerLayerInfer.__new__(LlamaTransformerLayerInfer)
    layer.layer_num_ = logical_layer_id
    layer.tp_q_head_num_ = layer.tp_k_head_num_ = state.config.head_num
    layer.head_dim_ = state.config.head_dim
    model = SimpleNamespace(
        att_state_manager=manager,
        att_state_role=role,
        args=StartArgs(llm_kv_type="fp8kv_sph" if backend_name == "fa3" else "fp8kv_spt"),
    )
    params = create_layer_attention_params(model, layer, None, AttentionFamily.DENSE)
    attention = make_layer_attention(backend, context, params)
    attention = replace(
        attention,
        logical_layer_id=logical_layer_id,
        alloc_tensor=lambda shape, dtype, device: torch.empty(shape, dtype=dtype, device="cpu"),
    )
    setattr(layer, operation + "_attention", attention)
    infer = SimpleNamespace(attention_contexts={0: context})
    monkeypatch.setattr(manager, "get_state", lambda *args: pytest.fail("forward must not look up storage owners"))
    q = torch.ones((1, state.config.head_num, state.config.head_dim))
    if operation == "prefill":
        output = layer._context_attention_kernel(q=q, kv=None, infer_state=infer, layer_weight=None)
    else:
        output = layer._token_attention_kernel(q=q, infer_state=infer, layer_weight=None)
    assert torch.equal(output, q)
    assert calls == [operation]
