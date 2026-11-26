from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.attention_state.bounded_att import WindowCompressorState, WindowCompressorStateConfig
from lightllm.common.attention_state.full_att.packed_mla.fp8_e8m0_packed_mla_state import (
    FP8E8M0PackedMLAConfig,
    FP8E8M0PackedMLAState,
)
from lightllm.common.attention_state.full_att.packed_mla.packed_mla_state import (
    PackedMLAAttentionConfig,
    PackedMLAFullAttentionState,
    PackedMLALogicalConfig,
)
from lightllm.common.basemodel.batch_objs import ModelInput, PostLayerOutput
from lightllm.common.basemodel.hidden_collector import FinalHiddenCollector
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight.rowmm_weight import ROWMMWeight
from lightllm.common.basemodel.triton_kernel.norm.rmsnorm import rmsnorm_forward
from lightllm.models.glm5_next.indexer import Glm5NextNsaInfer
from lightllm.models.glm5_next_mtp.layer_infer.post_layer_infer import Glm5NextMTPPostLayerInfer
from lightllm.models.glm5_next_mtp.layer_infer.transformer_layer_infer import Glm5NextMTPTransformerLayerInfer
from lightllm.models.glm5_next_mtp.model import Glm5NextMTPModel
from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer
from lightllm.server.core.objs.start_args_type import StartArgs


def make_model():
    model = Glm5NextMTPModel.__new__(Glm5NextMTPModel)
    model.args = StartArgs()
    model.args.mtp_mode = "eagle_with_att"
    model.args.mtp_step = 1
    model.args.llm_kv_type = "fp8kv_dsa"
    model.tp_world_size_ = model.layers_num = 1
    model.prefill_graph = None
    model.layers_infer = [SimpleNamespace(is_linear_attention_layer=False, use_mhc=False)]
    model.trans_layers_weight = [object()]
    return model


@pytest.mark.parametrize(
    "field,value",
    [
        ("mtp_mode", "eagle_no_att"),
    ],
)
def test_unsupported_execution_uses_full_forward(field, value):
    model = make_model()
    assert model._supports_prefill_kv_only()
    setattr(model.args, field, value)
    assert not model._supports_prefill_kv_only()


@pytest.mark.parametrize("field,value", [("mtp_mode", "eagle_no_att")])
def test_unsupported_layout_uses_full_forward(field, value):
    model = make_model()
    setattr(model.args, field, value)
    assert not model._supports_prefill_kv_only()


def test_eagle_multistep_and_vanilla_multistep_capabilities():
    model = make_model()
    model.args.mtp_step = 3
    assert model._supports_prefill_kv_only()
    model.args.mtp_mode = "vanilla_with_att"
    assert not model._supports_prefill_kv_only()


@pytest.mark.parametrize("kv_type", ["fp8kv_dsa", "bf16"])
@pytest.mark.parametrize("tp_world_size", [1, 4])
def test_cache_only_capability_does_not_depend_on_kv_format_or_tp(kv_type, tp_world_size):
    model = make_model()
    model.args.llm_kv_type = kv_type
    model.tp_world_size_ = tp_world_size
    assert model._supports_prefill_kv_only()


@pytest.mark.parametrize("tokens", [0, 1, 8192])
@pytest.mark.parametrize("mode", ["eagle_with_att", "vanilla_with_att"])
def test_forward_uses_common_prefill(monkeypatch, tokens, mode):
    from lightllm.common.basemodel import basemodel

    model = make_model()
    model.args.mtp_mode = mode
    model.mtp_previous_draft_models = []
    model.req_manager = SimpleNamespace(HOLD_REQUEST_ID=3)
    model.att_state_manager = object()
    model.hidden_collector_prototype = FinalHiddenCollector()
    model.return_all_prompt_logics = False
    model.prefill_att_backend = model.prefill_att_backend1 = None
    model._select_mem_indexes = lambda model_input: torch.zeros_like(model_input.input_ids)
    batch = int(tokens > 0)
    model_input = ModelInput(
        is_prefill=True,
        batch_size=batch,
        total_token_num=tokens,
        max_q_seq_len=tokens,
        max_kv_seq_len=tokens,
        max_cache_len=0,
        input_ids=torch.zeros(tokens, dtype=torch.int64),
        b_req_idx=torch.zeros(batch, dtype=torch.int32),
        b_mtp_index=torch.zeros(batch, dtype=torch.int32),
        b_seq_len=torch.full((batch,), tokens, dtype=torch.int32),
        b_ready_cache_len=torch.zeros(batch, dtype=torch.int32),
        b_prefill_start_loc=torch.zeros(batch, dtype=torch.int32),
        b_is_decode_req=torch.zeros(batch, dtype=torch.bool),
        b_prefill_has_output_cpu=[True] * batch,
        multimodal_params=[{"images": [], "audios": []} for _ in range(batch)],
        mtp_draft_input_hiddens=torch.ones(tokens, 2),
    )
    model.pre_infer = SimpleNamespace(
        context_forward=Mock(side_effect=lambda ids, *_: torch.zeros(ids.shape[0], 2)),
        _tpsp_sp_split=lambda input, **_: input,
    )
    model.post_infer = Glm5NextMTPPostLayerInfer.__new__(Glm5NextMTPPostLayerInfer)
    model.post_infer.tp_world_size_ = 1
    model.post_infer._tpsp_allgather = Mock(side_effect=lambda input, **_: input)
    post_forward = Mock(side_effect=lambda hidden, state, _: PostLayerOutput(torch.ones(state.batch_size, 3)))
    monkeypatch.setattr(LlamaPostLayerInfer, "token_forward", post_forward)
    model.pre_post_weight = object()
    layer = Glm5NextMTPTransformerLayerInfer.__new__(Glm5NextMTPTransformerLayerInfer)
    layer.is_linear_attention_layer = layer.use_mhc = False
    layer.layer_num_, layer.num_hidden_layers = 45, 1
    layer.tp_world_size_ = 1
    layer.embed_dim_ = 2
    layer._att_norm = layer._ffn_norm = lambda hidden, *_: hidden
    layer.context_attention_forward = Mock(side_effect=lambda hidden, *_: torch.zeros_like(hidden))
    layer._ffn = Mock(side_effect=lambda hidden, *_: torch.zeros_like(hidden))
    kv_pack = object()
    projection = SimpleNamespace(
        mm=Mock(),
        mm_param_list=[object(), kv_pack],
        quant_method=SimpleNamespace(
            method_name="fp8w8a8-b128-deepgemm",
            use_packed_ue8m0=False,
            apply=Mock(side_effect=lambda hidden, _: hidden),
        ),
    )
    model.trans_layers_weight = [SimpleNamespace(qkv_a_proj_with_mqa_=projection)]
    layer._normalize_sparse_kv = lambda kv, _: kv
    layer.indexer = SimpleNamespace(write_k_pools=Mock())
    writer = Mock()
    layer._post_cache_kv = writer
    model.layers_infer = [layer]
    allocator = SimpleNamespace(cache_env_in=Mock(), cache_env_out=Mock())
    monkeypatch.setattr(basemodel, "g_cache_manager", allocator)
    monkeypatch.setattr(basemodel.dist_group_manager, "get_group", Mock())
    monkeypatch.setattr(basemodel.dist_group_manager, "clear_deepep_buffer", Mock())
    monkeypatch.setattr(ModelInput, "to_cuda", lambda self: self.check_input())
    monkeypatch.setattr(InferStateInfo, "init_some_extra_state", Mock())
    event = Mock()
    monkeypatch.setattr(torch.cuda, "Event", Mock(return_value=event))
    output = model.forward(model_input)
    assert output.prefill_mem_indexes_ready_event is event
    assert output.logits.numel() == 0
    assert output.mtp_collector.spec_hidden is None
    ids, infer, _ = model.pre_infer.context_forward.call_args.args
    assert ids.shape == (max(1, tokens),)
    assert infer.b_req_idx.tolist() == ([0] if tokens else [model.req_manager.HOLD_REQUEST_ID])
    projection.mm.assert_not_called()
    projection.quant_method.apply.assert_called_once()
    projected_input, projected_pack = projection.quant_method.apply.call_args.args
    assert projected_input.shape == (max(1, tokens), layer.embed_dim_)
    assert projected_pack is kv_pack
    if mode == "eagle_with_att":
        assert infer.mtp_draft_input_hiddens is None
        writer.assert_called_once()
        layer.context_attention_forward.assert_not_called()
        layer._ffn.assert_not_called()
        post_forward.assert_not_called()
        placeholder = model.post_infer._tpsp_allgather.call_args.kwargs["input"]
        assert placeholder.shape == (0, layer.embed_dim_)
        assert placeholder.untyped_storage().nbytes() == 0
        layer.indexer.write_k_pools.assert_called_once()
    else:
        writer.assert_called_once()
        layer.context_attention_forward.assert_not_called()
        layer._ffn.assert_not_called()
        post_forward.assert_not_called()
    allocator.cache_env_in.assert_called_once()
    allocator.cache_env_out.assert_called_once()


@pytest.mark.parametrize("method_name,packed", [("fp8w8a8-b128-deepgemm", True), ("fp8w8a8-b128", False)])
def test_kv_only_uses_fused_projection_for_unsupported_weight_layouts(method_name, packed):
    layer = Glm5NextMTPTransformerLayerInfer.__new__(Glm5NextMTPTransformerLayerInfer)
    layer.is_linear_attention_layer = False
    layer.embed_dim_ = layer.tp_world_size_ = layer.q_lora_rank = layer.kv_lora_rank = 1
    output = torch.tensor([[3.0, 4.0]])
    method = SimpleNamespace(method_name=method_name, use_packed_ue8m0=packed, apply=Mock())
    projection = SimpleNamespace(mm=Mock(return_value=output), quant_method=method)
    weights = SimpleNamespace(qkv_a_proj_with_mqa_=projection)
    infer = SimpleNamespace(need_dp_prefill_balance=False)

    _, _, kv = layer._project_sparse_qkv_a(torch.ones(1, 1), infer, weights, kv_only=True)

    projection.mm.assert_called_once()
    method.apply.assert_not_called()
    torch.testing.assert_close(kv, output[:, 1:])


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="Hopper required"
)
@pytest.mark.parametrize("tokens", [1, 8192])
def test_kv_only_matches_fused_deepgemm_projection(monkeypatch, tokens):
    pytest.importorskip("deep_gemm")
    from lightllm.common.quantization.deepgemm import DeepGEMMFP8w8a8B128QuantizationMethod

    torch.manual_seed(53)
    monkeypatch.setenv("LIGHTLLM_USE_UE8M0_SCALES", "1")
    layer = Glm5NextMTPTransformerLayerInfer.__new__(Glm5NextMTPTransformerLayerInfer)
    layer.is_linear_attention_layer = False
    layer.tp_world_size_ = 1
    layer.embed_dim_, layer.q_lora_rank, layer.kv_lora_rank = 4096, 1536, 512
    quant_method = DeepGEMMFP8w8a8B128QuantizationMethod()
    projection = ROWMMWeight(
        in_dim=layer.embed_dim_,
        out_dims=[layer.q_lora_rank, layer.kv_lora_rank],
        weight_names=["q.weight", "kv.weight"],
        data_type=torch.bfloat16,
        quant_method=quant_method,
        tp_rank=0,
        tp_world_size=1,
    )
    for pack in projection.mm_param_list:
        quant_method.quantize(torch.randn_like(pack.weight, dtype=torch.bfloat16) * 0.1, pack)
    weights = SimpleNamespace(qkv_a_proj_with_mqa_=projection)
    infer = SimpleNamespace(need_dp_prefill_balance=False)
    hidden = torch.randn(tokens, layer.embed_dim_, device="cuda", dtype=torch.bfloat16)

    _, _, expected = layer._project_sparse_qkv_a(hidden, infer, weights)
    _, query, actual = layer._project_sparse_qkv_a(hidden, infer, weights, kv_only=True)

    assert query is None
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kv_type", ["fp8kv_dsa", "bf16"])
@pytest.mark.parametrize(
    "requests",
    [((3, 0, 1),), ((0, 0, 8192),), ((0, 131071, 129), (2, 1048573, 3)), ((1, 3, 1), (0, 5, 7))],
)
def test_cache_only_matches_full_prefill_cache_writes(monkeypatch, kv_type, requests):
    from lightllm.models.glm5_next import indexer as indexer_module

    torch.manual_seed(53)
    layer_id, tokens = 45, sum(q for _, _, q in requests)
    layer = Glm5NextMTPTransformerLayerInfer.__new__(Glm5NextMTPTransformerLayerInfer)
    layer.layer_num_, layer.num_hidden_layers = layer_id, 1
    layer.use_mhc = layer.is_linear_attention_layer = False
    layer.tp_world_size_ = layer.tp_q_head_num_ = 1
    layer.embed_dim_ = layer.q_lora_rank = 64
    layer.kv_lora_rank = layer.qk_nope_head_dim = 512
    layer.eps_ = 1e-5
    layer.indexer = Glm5NextNsaInfer(
        layer_id, dict(index_topk=2048, index_n_heads=1, index_head_dim=128, rms_norm_eps=1e-5, index_kpool=4), 1
    )

    def projection(rows, cols):
        weight = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16) * 0.1
        return SimpleNamespace(mm=Mock(side_effect=lambda x: x @ weight))

    def norm(input, eps, **_):
        return rmsnorm_forward(input, None, eps)

    qkv_projection = ROWMMWeight(
        in_dim=64,
        out_dims=[64, 512],
        weight_names=["q.weight", "kv.weight"],
        data_type=torch.bfloat16,
        tp_rank=0,
        tp_world_size=1,
    )
    qkv_projection.load_hf_weights(
        {
            "q.weight": torch.randn(64, 64, device="cuda", dtype=torch.bfloat16) * 0.1,
            "kv.weight": torch.randn(512, 64, device="cuda", dtype=torch.bfloat16) * 0.1,
        }
    )
    assert qkv_projection.verify_load()
    qkv_projection.mm = Mock(wraps=qkv_projection.mm)
    weights = SimpleNamespace(
        att_norm_weight_=norm,
        qkv_a_proj_with_mqa_=qkv_projection,
        q_a_layernorm_=SimpleNamespace(weight=torch.ones(64, device="cuda", dtype=torch.bfloat16)),
        kv_a_layernorm_=SimpleNamespace(weight=torch.ones(512, device="cuda", dtype=torch.bfloat16)),
        q_b_proj_=projection(64, 512),
        wk_proj_=projection(64, 128),
        k_norm_=norm,
        index_kpool_compress_gate=projection(64, 128),
        index_kpool_compress_ape=SimpleNamespace(weight=torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)),
    )
    resources = AttentionStateManager(request_capacity=4)
    stores = []
    for role in ("main", "draft0"):
        if kv_type == "fp8kv_dsa":
            full = FP8E8M0PackedMLAState(FP8E8M0PackedMLAConfig(logical_layer_groups=((layer_id,),)))
        else:
            full = PackedMLAFullAttentionState(
                PackedMLAAttentionConfig(
                    PackedMLALogicalConfig(
                        logical_layer_groups=((layer_id,),),
                        latent_dim=512,
                        format="packed_bf16",
                        indexer_dim=128,
                    )
                ),
                device="cuda",
            )
        full.allocate_storage(2 * tokens + 2)
        full.storage.fill_(0x5A)
        tail = WindowCompressorState(
            WindowCompressorStateConfig((layer_id,), 4, 4, 128, "indexer", torch.bfloat16), 4, mtp_step=1
        )
        tail.buffer.normal_()
        resources.register_state(role, full)
        resources.register_state(role, tail)
        stores.extend((full.storage, tail.buffer))
    resources.finish_state_registration()
    initial = [s.clone() for s in stores]
    q_lengths = [q for _, _, q in requests]
    lengths = torch.cat([torch.arange(past + 1, past + q + 1, device="cuda") for _, past, q in requests]).int()
    infer = SimpleNamespace(
        is_prefill=True,
        is_draft_model=True,
        prefill_kv_only=True,
        need_dp_prefill_balance=False,
        att_state_role="draft0",
        att_state_manager=resources,
        mem_index=2 * torch.arange(tokens, device="cuda", dtype=torch.int32) + 1,
        b_req_idx=torch.tensor([r for r, _, _ in requests], device="cuda", dtype=torch.int32),
        b_seq_len=torch.tensor([p + q for _, p, q in requests], device="cuda", dtype=torch.int32),
        b1_cu_q_seq_len=torch.tensor([0, *q_lengths], device="cuda", dtype=torch.int32).cumsum(0).int(),
        max_q_seq_len=max(q_lengths),
        max_kv_seq_len=2048,
        prefill_att_state=SimpleNamespace(q_kv_seq_len=lengths, q_req_idx=None),
        req_manager=SimpleNamespace(req_to_token_indexs=None),
    )
    # Top-k outputs are discarded here; run the normal projections and cache writers.
    monkeypatch.setattr(indexer_module, "expand_topk", Mock(return_value=(None, None)))
    hidden = torch.randn(tokens, 64, device="cuda", dtype=torch.bfloat16)
    _, kv = layer._get_qkv(layer._att_norm(hidden, infer, weights), infer, weights)
    layer._post_cache_kv(kv, infer, weights)
    layer.indexer._get_indices(
        **infer.get_topk_indices_params, infer_state=infer, att_state=infer.prefill_att_state, layer_weight=weights
    )
    expected = [s.clone() for s in stores]
    assert not torch.equal(initial[2], expected[2])
    for store, before in zip(stores, initial):
        store.copy_(before)
    qkv_projection.mm.reset_mock()
    weights.q_b_proj_.mm.reset_mock()
    placeholder = layer.context_forward(hidden, infer, weights)
    assert placeholder.shape == (0, layer.embed_dim_)
    assert placeholder.untyped_storage().nbytes() == 0
    qkv_projection.mm.assert_not_called()
    weights.q_b_proj_.mm.assert_not_called()
    for actual, reference in zip(stores, expected):
        assert torch.equal(actual, reference)
    for actual, before in zip(stores[:2], initial[:2]):
        assert torch.equal(actual, before)
    assert torch.equal(stores[2][:, ::2], initial[2][:, ::2])
    assert torch.equal(stores[3][:, 3], initial[3][:, 3])
