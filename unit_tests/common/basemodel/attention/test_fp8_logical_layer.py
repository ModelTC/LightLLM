"""FP8 后端按推理层传入的逻辑层号选择标定，覆盖稀疏层号、共享层和 draft。"""

from dataclasses import replace
from types import SimpleNamespace
import weakref

import pytest
import torch

from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.basemodel.attention.fa3 import fp8 as fa3_fp8
from lightllm.common.basemodel.attention.flashinfer import fp8 as flashinfer_fp8
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from unit_tests.common.att_resources.model_helpers import rebuild_full_state
from unit_tests.common.att_resources.test_capacity import definition


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
        config=replace(template.config, logical_layer_groups=((3, 7), (8, 11))),
        token_capacity=8,
        device="cpu",
    )
    state.scales[0].fill_(2)
    state.scales[1].fill_(5)
    expected_scale = 2 if storage_layer_index == 0 else 5
    manager = AttentionStateManager(request_capacity=1)
    manager.register_state(role, state)
    manager.finish_state_registration()
    infer = LlamaInferStateInfo()
    infer.att_state_manager = manager
    infer.att_state_role = role
    infer.b_seq_len = torch.tensor([1], dtype=torch.int32)
    infer.max_q_seq_len = 1
    calls = []

    # 仅替换外部计算内核，保留推理层、公开 attention 接口及 State 标定查询链路。
    if backend_name == "fa3":
        state.q_scales[0].fill_(3)
        state.q_scales[1].fill_(7)
        expected_q_scale = 3 if storage_layer_index == 0 else 7

        def quantize(q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
            assert torch.all(scales == expected_q_scale)
            return q

        def attend(**kwargs: object) -> torch.Tensor:
            assert torch.all(kwargs["k_descale"] == expected_scale)
            assert torch.all(kwargs["v_descale"] == expected_scale)
            assert torch.all(kwargs["q_descale"] == expected_q_scale)
            calls.append(operation)
            return kwargs["q"]

        monkeypatch.setattr(fa3_fp8, "q_per_head_static_fp8_quant", quantize)
        monkeypatch.setattr(fa3_fp8, "flash_attn_with_kvcache", attend)
        state_class = fa3_fp8.Fp8Fa3PrefillAttState if operation == "prefill" else fa3_fp8.Fp8Fa3DecodeAttState
    else:
        state_class = (
            flashinfer_fp8.Fp8FlashInferPrefillAttState if operation == "prefill"
            else flashinfer_fp8.Fp8FlashInferDecodeAttState
        )

    attention = state_class(backend=SimpleNamespace(), _infer_state_weak_ref=weakref.ref(infer))
    if backend_name == "fa3":
        attention.k_descale = state.scales[:, :2].unsqueeze(1)
        attention.v_descale = state.scales[:, 2:].unsqueeze(1)
        attention.page_table = attention.cu_seqlens_q = attention.cu_seqlens_k = None
        attention.b_att_seq_len = infer.b_seq_len
        attention.decode_max_q_seq_len = 1
        attention.causal = True
    else:
        def run(
            q: torch.Tensor,
            paged_kv_cache: tuple[torch.Tensor, torch.Tensor],
            k_scale: float,
            v_scale: float,
            out: torch.Tensor,
        ) -> None:
            assert k_scale == v_scale == expected_scale
            calls.append(operation)
            out.copy_(q)

        attention.prefill_wrapper = attention.decode_wrapper = SimpleNamespace(run=run)

    infer.prefill_att_state = infer.decode_att_state = attention
    layer = LlamaTransformerLayerInfer.__new__(LlamaTransformerLayerInfer)
    layer.layer_num_ = logical_layer_id
    layer.tp_q_head_num_ = layer.tp_k_head_num_ = state.config.head_num
    layer.head_dim_ = state.config.head_dim
    layer.alloc_tensor = lambda shape, dtype, device: torch.empty(shape, dtype=dtype, device="cpu")
    q = torch.ones((1, state.config.head_num, state.config.head_dim))
    if operation == "prefill":
        output = layer._context_attention_kernel(
            q=q,
            kv=None,
            infer_state=infer,
            layer_weight=None,
        )
    else:
        output = layer._token_attention_kernel(q=q, infer_state=infer, layer_weight=None)
    assert torch.equal(output, q)
    assert calls == [operation]
