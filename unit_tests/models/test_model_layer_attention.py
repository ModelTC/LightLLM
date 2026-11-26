"""Model layers create their Attention with their own layer properties when the model loads."""
from typing import Any

from types import SimpleNamespace
from dataclasses import replace

import pytest
import torch

from lightllm.common.attention.backends.gqa import GQAAttConfig
from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.models.llama.model import LlamaTpPartModel

H, D = 8, 64


class FakeState:
    def __init__(self, kv_heads, head_dim=D):
        self.k = torch.empty(16, kv_heads, head_dim)

    def get_layer_view(self, logical_layer_id):
        return self.k, self.k

    def get_layer_kv_scales(self, logical_layer_id):
        return None


def make_model(
    state: Any, backend: Any = "fa3", draft: Any = False, *, family: Any = "gqa", model_cls: Any = LlamaTpPartModel
):
    args = SimpleNamespace(
        att_backend={f"{family}.{phase}": backend for phase in ("prefill", "decode")},
        mtp_step=1 if draft else 0,
        mtp_mode="dflash" if draft else None,
        enable_ep_moe=False,
        page_size=1,
        flashinfer_workspace_size=64,
        running_max_req_size=8,
        enable_tpsp_mix_mode=False,
        enable_decode_microbatch_overlap=False,
        diverse_mode=False,
        llm_kv_type="None",
    )
    model = SimpleNamespace(
        args=args,
        create_att_backends=model_cls.create_att_backends,
        max_seq_length=1024,
        graph_max_batch_size=4,
        graph_max_len_in_batch=1024,
        tp_world_size_=1,
        is_mtp_draft_model=draft,
        mtp_manager=SimpleNamespace(get_decode_draft_step=lambda is_draft_model: 1 if draft else 0),
        data_type=torch.bfloat16,
        att_state_role="draft0" if draft else "main",
        att_state_manager=SimpleNamespace(
            get_state=lambda **query: state,
            get_bounded_attention_states=lambda role: (),
            get_full_attention_states=lambda role: (state,),
        ),
    )
    if draft:
        model.main_model = SimpleNamespace()
    return model


def register_layers(model: Any, layers: tuple[Any, ...]):
    model.layers_infer = layers
    if not hasattr(model, "config"):
        model.config = layers[0].network_config_
    TpPartBaseModel._init_attention(model)


def config(**fields) -> GQAAttConfig:
    return GQAAttConfig(
        num_heads=fields.get("num_heads", H),
        num_kv_heads=fields.get("num_kv_heads", 2),
        head_dim=fields.get("head_dim", D),
        dtype=fields.get("dtype", torch.bfloat16),
        sliding_window=fields.get("sliding_window", (-1, -1)),
        softmax_scale=fields.get("softmax_scale", None),
        causal=fields.get("causal", True),
        has_sinks=fields.get("has_sinks", False),
        kv_quant=fields.get("kv_quant", None),
        kv_quant_group_size=fields.get("kv_quant_group_size", None),
        has_q_scale=fields.get("has_q_scale", False),
        has_alibi=fields.get("has_alibi", False),
        bidirectional_image_spans=fields.get("bidirectional_image_spans", False),
    )


def layer_backend(model: Any, layer: Any, phase: Any):
    return model.attention_backend_manager.get_backend(
        role=model.att_state_role, phase=phase, logical_layer_id=layer.layer_num_
    )


def layer_configs(model: Any, layer: Any):
    prefill = layer_backend(model=model, layer=layer, phase="prefill")
    decode = layer_backend(model=model, layer=layer, phase="decode")
    return None if prefill is None else prefill.att_config, decode.att_config


def decode_inputs(monkeypatch, model, layer, weights):
    impl = layer_backend(model=model, layer=layer, phase="decode")
    state = object()
    captured = []

    def forward(att_state: Any, layer_inputs: Any, *, out: torch.Tensor, alloc_func: Any):
        assert att_state is state
        captured.append(layer_inputs)
        return out

    monkeypatch.setattr(impl, "decode", forward)
    monkeypatch.setattr(layer, "alloc_tensor", torch.empty)
    infer_state = SimpleNamespace(
        att_state_manager=model.att_state_manager,
        attention_backend_manager=model.attention_backend_manager,
        att_state_role=model.att_state_role,
        att_states={impl: state},
    )
    query = torch.empty(3, layer.tp_q_head_num_ * layer.head_dim_)
    result = layer._token_attention_kernel(query, infer_state, weights)
    assert result.shape == query.shape
    (inputs,) = captured
    assert inputs.q.shape == (3, layer.tp_q_head_num_, layer.head_dim_)
    return inputs


def test_bloom_layers_run_alibi_on_triton_whatever_the_command_line_asks(runtime_start_args, monkeypatch):
    from lightllm.models.bloom.layer_infer.transformer_layer_infer import BloomTransformerLayerInfer
    from lightllm.models.bloom.model import BloomTpPartModel

    layer = BloomTransformerLayerInfer(0, {"num_attention_heads": H, "n_embed": H * D, "layer_norm_epsilon": 1e-5})
    model = make_model(state=FakeState(H), model_cls=BloomTpPartModel)
    register_layers(model=model, layers=(layer,))
    assert layer_configs(model=model, layer=layer) == (config(num_kv_heads=H, has_alibi=True),) * 2
    assert type(layer_backend(model=model, layer=layer, phase="decode")).__name__ == "TritonGQADecodeBackend"
    slopes = torch.ones(H)
    assert decode_inputs(monkeypatch, model, layer, SimpleNamespace(tp_alibi=slopes)).alibi_slopes is slopes


def test_starcoder_layers_run_multi_query_attention_without_alibi(runtime_start_args, monkeypatch):
    from lightllm.models.starcoder.layer_infer.transformer_layer_infer import StarcoderTransformerLayerInfer
    from lightllm.models.starcoder.model import StarcoderTpPartModel

    layer = StarcoderTransformerLayerInfer(0, {"num_attention_heads": H, "n_embed": H * D, "layer_norm_epsilon": 1e-5})
    model = make_model(state=FakeState(1), model_cls=StarcoderTpPartModel)
    register_layers(model=model, layers=(layer,))
    assert layer_configs(model=model, layer=layer) == (config(num_kv_heads=1),) * 2
    assert type(layer_backend(model=model, layer=layer, phase="decode")).__name__ == "FA3GQADecodeBackend"
    assert decode_inputs(monkeypatch, model, layer, SimpleNamespace()).alibi_slopes is None


def test_gemma4_sliding_and_full_layers_keep_separate_backends(runtime_start_args: Any):
    from lightllm.models.gemma4.layer_infer.transformer_layer_infer import Gemma4TransformerLayerInfer
    from lightllm.models.gemma4.model import Gemma4TpPartModel

    net = dict(
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=512,
        head_dim=256,
        global_head_dim=512,
        num_global_key_value_heads=1,
        rms_norm_eps=1e-6,
        sliding_window=48,
        num_hidden_layers=4,
        layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
    )
    # 缓存存储每个 token 的 2 个槽位，每个槽位 256 个：两个滑动 KV 头，或单个 512 维的完整 KV 头。
    model = make_model(state=FakeState(2, 256), model_cls=Gemma4TpPartModel)
    sliding, full, next_sliding, next_full = (Gemma4TransformerLayerInfer(i, net) for i in range(4))
    register_layers(
        model=model,
        layers=(
            sliding,
            full,
            next_sliding,
            next_full,
        ),
    )
    window = dict(num_heads=4, head_dim=256, sliding_window=(47, 0))
    assert layer_configs(model=model, layer=sliding) == (
        config(**window, bidirectional_image_spans=True),
        config(**window),
    )
    assert layer_configs(model=model, layer=full) == (config(num_heads=4, num_kv_heads=1, head_dim=512),) * 2
    for phase in ("prefill", "decode"):
        assert layer_backend(model=model, layer=sliding, phase=phase) is layer_backend(
            model=model, layer=next_sliding, phase=phase
        )
        assert layer_backend(model=model, layer=full, phase=phase) is layer_backend(
            model=model, layer=next_full, phase=phase
        )
    assert all(
        len(model.attention_backend_manager.get_backends(role=model.att_state_role, phase=phase)) == 2
        for phase in ("prefill", "decode")
    )


def test_gpt_oss_layers_have_sinks_and_sliding_layers_a_window(runtime_start_args, monkeypatch):
    from lightllm.models.gpt_oss.layer_infer.transformer_layer_infer import GptOssTransformerLayerInfer
    from lightllm.models.gpt_oss.model import GptOssTpPartModel

    net = dict(
        num_attention_heads=H,
        num_key_value_heads=2,
        hidden_size=H * D,
        head_dim=D,
        rms_norm_eps=1e-6,
        num_experts_per_tok=2,
        sliding_window=128,
        layer_types=["sliding_attention", "full_attention"],
    )
    model = make_model(state=FakeState(2), model_cls=GptOssTpPartModel)
    sliding, full = (GptOssTransformerLayerInfer(i, net) for i in range(2))
    register_layers(
        model=model,
        layers=(
            sliding,
            full,
        ),
    )
    assert layer_configs(model=model, layer=sliding) == (config(has_sinks=True, sliding_window=(127, 0)),) * 2
    assert layer_configs(model=model, layer=full) == (config(has_sinks=True),) * 2
    sinks = torch.ones(H)
    weights = SimpleNamespace(attn_sinks=SimpleNamespace(weight=sinks))
    assert decode_inputs(monkeypatch, model, full, weights).sinks is sinks


def test_neo_layers_attend_inside_image_spans_in_prefill_only(runtime_start_args):
    from lightllm.models.neo_chat.layer_infer.transformer_layer_infer import NeoChatTransformerLayerInfer
    from lightllm.models.neo_chat.model import NeoTpPartModel

    net = dict(num_attention_heads=H, num_key_value_heads=2, hidden_size=H * D, head_dim=D, rms_norm_eps=1e-6)
    layer = NeoChatTransformerLayerInfer(0, net)
    model = make_model(state=FakeState(2), backend="triton", model_cls=NeoTpPartModel)
    register_layers(model=model, layers=(layer,))
    assert layer_configs(model=model, layer=layer) == (config(bidirectional_image_spans=True), config())


@pytest.mark.parametrize("sliding_window, expected", [(0, (-1, -1)), (40, (39, 39))])
def test_dflash_layers_attend_only_in_decode_and_not_causally(sliding_window: Any, expected: Any):
    from lightllm.models.qwen3_dflash2.layer_infer.transformer_layer_infer import Qwen3DFlash2TransformerLayerInfer
    from lightllm.models.qwen3_dflash2.model import Qwen3DFlash2Model

    net = dict(
        num_attention_heads=H,
        num_key_value_heads=2,
        hidden_size=H * D,
        head_dim=D,
        rms_norm_eps=1e-6,
        block_size=2,
        conv_group_size=1,
        sliding_window=sliding_window,
    )
    layer = Qwen3DFlash2TransformerLayerInfer(0, net)
    model = make_model(state=FakeState(2), draft=True, model_cls=Qwen3DFlash2Model)
    model.args.mtp_mode = "dflash2"
    register_layers(model=model, layers=(layer,))
    assert layer_configs(model=model, layer=layer) == (None, config(causal=False, sliding_window=expected))
    assert type(layer_backend(model=model, layer=layer, phase="decode")).__name__ == "FA3GQADecodeBackend"


def test_dflash_decode_needs_an_implementation_of_non_causal_attention():
    from lightllm.models.qwen3_dflash.layer_infer.transformer_layer_infer import Qwen3DFlashTransformerLayerInfer
    from lightllm.models.qwen3_dflash.model import Qwen3DFlashModel

    net = dict(num_attention_heads=H, num_key_value_heads=2, hidden_size=H * D, head_dim=D, rms_norm_eps=1e-6)
    layer = Qwen3DFlashTransformerLayerInfer(0, net)
    with pytest.raises(ValueError, match="does not support"):
        model = make_model(state=FakeState(2), backend="triton", draft=True, model_cls=Qwen3DFlashModel)
        register_layers(model=model, layers=(layer,))


@pytest.mark.parametrize("rope_scaling", [None, {"type": "yarn", "factor": 40, "mscale_all_dim": 1.0}])
def test_deepseek2_layers_attend_over_expanded_keys_in_prefill_and_the_latent_cache_in_decode(
    runtime_start_args, rope_scaling
):
    from lightllm.common.attention.backends.mla import MLADecodeAttConfig, MLAPrefillAttConfig
    from lightllm.models.deepseek2.layer_infer.transformer_layer_infer import Deepseek2TransformerLayerInfer
    from lightllm.models.deepseek2.model import Deepseek2TpPartModel
    from lightllm.models.llama.yarn_rotary_utils import get_deepseek_mscale

    net = dict(
        num_attention_heads=16,
        num_key_value_heads=16,
        hidden_size=1024,
        rms_norm_eps=1e-6,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        q_lora_rank=None,
        kv_lora_rank=512,
        n_routed_experts=None,
        first_k_dense_replace=0,
        n_shared_experts=None,
        num_experts_per_tok=1,
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
        rope_scaling=rope_scaling,
    )
    scale = 192 ** -0.5 * (get_deepseek_mscale(40, 1.0) ** 2 if rope_scaling else 1.0)
    model = make_model(state=None, backend="flashinfer", family="mla", model_cls=Deepseek2TpPartModel)
    layers = [Deepseek2TransformerLayerInfer(i, net) for i in range(2)]
    register_layers(model=model, layers=layers)
    common = dict(num_heads=16, qk_rope_head_dim=64, softmax_scale=scale, dtype=torch.bfloat16)
    assert layer_configs(model=model, layer=layers[0]) == (
        MLAPrefillAttConfig(
            num_heads=common["num_heads"],
            qk_nope_head_dim=common.get("qk_nope_head_dim", 128),
            qk_rope_head_dim=common["qk_rope_head_dim"],
            v_head_dim=common.get("v_head_dim", 128),
            softmax_scale=common["softmax_scale"],
            dtype=common["dtype"],
            causal=common.get("causal", True),
        ),
        MLADecodeAttConfig(
            num_heads=common["num_heads"],
            kv_lora_rank=common.get("kv_lora_rank", 512),
            qk_rope_head_dim=common["qk_rope_head_dim"],
            softmax_scale=common["softmax_scale"],
            dtype=common["dtype"],
            causal=common.get("causal", True),
        ),
    )
    # FlashInfer 使用层共有的 softmax 缩放因子进行规划。
    assert layer_backend(model=model, layer=layers[0], phase="prefill") is layer_backend(
        model=model, layer=layers[1], phase="prefill"
    )
    assert layer_backend(model=model, layer=layers[0], phase="decode") is layer_backend(
        model=model, layer=layers[1], phase="decode"
    )


@pytest.mark.parametrize("kv_type, kv_cache_format", [("None", "bf16"), ("fp8kv_dsa", "fp8_ds_mla")])
def test_deepseek_v32_layers_attend_to_the_indexer_selection(runtime_start_args, kv_type, kv_cache_format):
    from lightllm.common.attention.backends.sparse_mla import SparseMLAAttConfig
    from lightllm.models.deepseek3_2.layer_infer.transformer_layer_infer import Deepseek3_2TransformerLayerInfer
    from lightllm.models.deepseek3_2.model import Deepseek3_2TpPartModel

    layer = Deepseek3_2TransformerLayerInfer.__new__(Deepseek3_2TransformerLayerInfer)
    layer.layer_num_ = 0
    layer.network_config_ = dict(
        num_attention_heads=64, kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64, index_topk=2048
    )
    # 稀疏 MLA 只有一个实现名称；自动选择会验证它。
    model = make_model(state=None, backend="flashmla_sparse", family="sparse_mla", model_cls=Deepseek3_2TpPartModel)
    model.args.llm_kv_type = kv_type
    register_layers(model=model, layers=(layer,))
    expected = SparseMLAAttConfig(
        num_heads=64,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        softmax_scale=192 ** -0.5,
        dtype=torch.bfloat16,
        kv_cache_format=kv_cache_format,
        topk=2048,
    )
    assert layer_configs(model=model, layer=layer) == (expected, expected)


@pytest.mark.parametrize("kv_type, expected_format", [("None", "bf16"), ("fp8kv_dsa", "fp8_group64_e8m0")])
def test_glm5_sparse_spec_describes_the_cache_without_a_workspace(kv_type, expected_format):
    from lightllm.models.glm5_next.layer_infer.transformer_layer_infer import Glm5NextTransformerLayerInfer

    from lightllm.models.glm5_next.model import Glm5NextTpPartModel

    layer = Glm5NextTransformerLayerInfer.__new__(Glm5NextTransformerLayerInfer)
    layer.layer_num_ = 3
    layer.is_linear_attention_layer = False
    model = make_model(state=None, model_cls=Glm5NextTpPartModel)
    model.args.llm_kv_type = kv_type
    model.config = dict(num_attention_heads=64, kv_lora_rank=512, qk_nope_head_dim=256, index_topk=2048, index_kpool=4)
    register_layers(model=model, layers=(layer,))
    assert all(
        backend.att_config.kv_cache_format == expected_format
        for phase in ("prefill", "decode")
        for backend in model.attention_backend_manager.get_backends(role="main", phase=phase)
    )


@pytest.mark.parametrize("prefill_kernel", ["auto", "triton", "flashkda"])
@pytest.mark.parametrize("ssm_state_mode", ["native", "replay"])
def test_glm5_kda_layers_follow_the_linear_backend_and_state_mode(
    runtime_start_args: Any, monkeypatch: pytest.MonkeyPatch, prefill_kernel: Any, ssm_state_mode: Any
):
    from lightllm.common.attention.backends.kda import KDAAttConfig
    from lightllm.common.attention.backends.kda import triton as kda_triton
    from lightllm.models.glm5_next.layer_infer.transformer_layer_infer import Glm5NextTransformerLayerInfer
    from lightllm.models.glm5_next.model import Glm5NextTpPartModel

    monkeypatch.setattr(kda_triton.TritonKDAPrefillBackend, "_check_prefill_kernel", lambda att_config: None)
    layer = Glm5NextTransformerLayerInfer.__new__(Glm5NextTransformerLayerInfer)
    layer.layer_num_ = 0
    layer.is_linear_attention_layer = True
    layer.tp_world_size_, layer.tp_linear_num_heads, layer.linear_head_dim = 1, 4, 128
    layer.network_config_ = {"linear_attn_config": {"num_heads": 4, "head_dim": 128, "short_conv_kernel_size": 4}}
    sparse = Glm5NextTransformerLayerInfer.__new__(Glm5NextTransformerLayerInfer)
    sparse.layer_num_ = 1
    sparse.is_linear_attention_layer = False
    sparse.tp_q_head_num_, sparse.kv_lora_rank, sparse.softmax_scale = 64, 512, 0.0625
    sparse.network_config_ = {"index_topk": 2048, "index_kpool": 4}
    model = make_model(state=None, backend="triton", model_cls=Glm5NextTpPartModel)
    model.config = dict(
        num_attention_heads=64,
        kv_lora_rank=512,
        qk_nope_head_dim=256,
        index_topk=2048,
        index_kpool=4,
        linear_attn_config=layer.network_config_["linear_attn_config"],
    )
    model.args.att_backend["kda.prefill"] = prefill_kernel
    model.args.ssm_state_mode, model.args.linear_att_ssm_data_type = ssm_state_mode, "bfloat16"
    register_layers(
        model=model,
        layers=(
            layer,
            sparse,
        ),
    )
    expected = KDAAttConfig(
        num_heads=4,
        head_dim=128,
        conv_kernel_size=4,
        lower_bound=-5.0,
        dtype=torch.bfloat16,
        replay=ssm_state_mode == "replay",
        ssm_dtype=torch.bfloat16,
        prefill_kernel=prefill_kernel,
    )
    assert layer_configs(model=model, layer=layer) == (expected, replace(expected, prefill_kernel="triton"))
    assert layer_backend(model=model, layer=sparse, phase="prefill").get_name() == "flashmla_sparse_mla_prefill"
    assert layer_backend(model=model, layer=sparse, phase="decode").get_name() == "fa3_sparse_mla_decode"
