"""FP8 后端按推理层传入的逻辑层号选择标定，覆盖稀疏层号、共享层和 draft。"""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.attention.backends.gqa.fa3 import FA3DecodeMetadata, FA3GQAPrefillMetadata
from lightllm.common.attention.backends.gqa.flashinfer import FlashInferGQAMetadata
from lightllm.common.basemodel.attention_backend_manager import AttentionBackendManager
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from unit_tests.common.att_resources.model_helpers import rebuild_full_state
from unit_tests.common.att_resources.model_helpers import replace_logical_layer_groups
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
    infer = LlamaInferStateInfo()
    infer.att_state_manager = manager
    infer.att_state_role = role
    infer.b_seq_len = torch.tensor([1], dtype=torch.int32)
    infer.max_q_seq_len = 1
    calls = []

    # 仅替换外部计算内核，保留推理层、公开 attention 接口、backend 及 State 标定查询链路。
    import lightllm.utils.sgl_utils as sgl_utils
    from lightllm.common.basemodel.triton_kernel.quantization import q_per_head_fp8_quant

    if backend_name == "fa3":
        state.q_scales[0].fill_(3)
        state.q_scales[1].fill_(7)
        expected_q_scale = 3 if storage_layer_index == 0 else 7

        def quantize(q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
            assert torch.all(scales == expected_q_scale)
            return q

        def attend(**kwargs: object) -> None:
            assert torch.all(kwargs["k_descale"] == expected_scale)
            assert torch.all(kwargs["v_descale"] == expected_scale)
            assert torch.all(kwargs["q_descale"] == expected_q_scale)
            calls.append(operation)
            kwargs["out"].copy_(kwargs["q"])

        monkeypatch.setattr(q_per_head_fp8_quant, "q_per_head_static_fp8_quant", quantize)
        monkeypatch.setattr(sgl_utils, "flash_attn_with_kvcache", attend)
        metadata_cls = FA3GQAPrefillMetadata if operation == "prefill" else FA3DecodeMetadata
        extra = {} if operation == "prefill" else {"max_seqlen_k": 1}
        att_metadata = metadata_cls(
            page_table=None,
            cache_seqlens=infer.b_seq_len,
            cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
            cu_seqlens_k=None,
            max_seqlen_q=1,
            page_size=1,
            **extra,
        )
        kv_type = "fp8kv_sph"
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

        att_metadata = FlashInferGQAMetadata(wrapper=SimpleNamespace(run=run), page_size=1)
        kv_type = "fp8kv_spt"

    # The layer creates its attention as the model loads it: kv_quant and the static q scale come
    # from the State, and the layer's group then reads the step's attention state.
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", "0")
    args = SimpleNamespace(
        llm_prefill_att_backend=[backend_name],
        llm_decode_att_backend=[backend_name],
        llm_kv_type=kv_type,
        mtp_step=0,
        enable_ep_moe=False,
        page_size=1,
        running_max_req_size=1,
        enable_tpsp_mix_mode=False,
        enable_decode_microbatch_overlap=False,
        flashinfer_workspace_size=512,
        diverse_mode=False,
    )
    model = SimpleNamespace(
        args=args,
        att_state_manager=manager,
        att_state_role=role,
        data_type=torch.bfloat16,
        is_mtp_draft_model=False,
        max_seq_length=8,
        graph_max_batch_size=1,
        graph_max_len_in_batch=8,
        mtp_manager=SimpleNamespace(get_decode_draft_step=lambda is_draft: 0),
    )
    layer = LlamaTransformerLayerInfer.__new__(LlamaTransformerLayerInfer)
    layer.layer_num_ = logical_layer_id
    layer.tp_q_head_num_ = layer.tp_k_head_num_ = state.config.head_num
    layer.head_dim_ = state.config.head_dim
    layer.alloc_tensor = lambda shape, dtype, device: torch.empty(shape, dtype=dtype, device="cpu")
    model.layers_infer = [layer]
    infer.attention_backend_manager = AttentionBackendManager()
    infer.attention_backend_manager.register_backends(model)
    backend = infer.attention_backend_manager.get_backend(role, operation, logical_layer_id)
    infer.att_metadata = {backend.metadata_group_key: att_metadata}
    q = torch.ones((1, state.config.head_num, state.config.head_dim), dtype=torch.bfloat16)
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
