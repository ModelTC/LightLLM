"""Model layers create their Attention with their own layer properties when the model loads."""

from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention.backends.gqa import GQALayerConfig
from lightllm.common.basemodel import attention_backend_manager
from lightllm.common.basemodel.basemodel import TpPartBaseModel

H, D = 8, 64


class FakeState:
    def __init__(self, kv_heads, head_dim=D):
        self.k = torch.empty(16, kv_heads, head_dim)

    def get_layer_view(self, logical_layer_id):
        return self.k, self.k

    def get_layer_kv_scales(self, logical_layer_id):
        return None


def make_model(state, backend="fa3", draft=False):
    args = SimpleNamespace(
        llm_prefill_att_backend=[backend],
        llm_decode_att_backend=[backend],
        mtp_step=1 if draft else 0,
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
        max_seq_length=1024,
        graph_max_batch_size=4,
        graph_max_len_in_batch=1024,
        tp_world_size_=1,
        is_mtp_draft_model=draft,
        mtp_manager=SimpleNamespace(get_decode_draft_step=lambda _: 1 if draft else 0),
        data_type=torch.bfloat16,
        att_state_role="main",
        att_state_manager=SimpleNamespace(
            get_state=lambda **query: state,
            get_bounded_attention_states=lambda role: (),
            get_full_attention_states=lambda role: (state,),
        ),
    )
    return model


def bind_layers(model, *layers):
    model.layers_infer = layers
    TpPartBaseModel._init_attention(model)


def config(**fields) -> GQALayerConfig:
    return GQALayerConfig(**{"num_heads": H, "num_kv_heads": 2, "head_dim": D, "dtype": torch.bfloat16, **fields})


def layer_configs(layer):
    prefill = None if layer.prefill_impl is None else layer.prefill_impl.layer_config
    return prefill, layer.decode_impl.layer_config


def decode_inputs(monkeypatch, model, layer, weights):
    impl = layer.decode_impl
    state = object()
    captured = []

    def forward(att_state, inputs, *, out, alloc_func):
        assert att_state is state
        captured.append(inputs)
        return out

    monkeypatch.setattr(impl, "forward", forward)
    monkeypatch.setattr(layer, "alloc_tensor", torch.empty)
    infer_state = SimpleNamespace(
        att_state_manager=model.att_state_manager,
        att_state_role=model.att_state_role,
        att_states={impl.state_key: state},
    )
    query = torch.empty(3, layer.tp_q_head_num_ * layer.head_dim_)
    result = layer._token_attention_kernel(query, infer_state, weights)
    assert result.shape == query.shape
    (inputs,) = captured
    assert inputs.q.shape == (3, layer.tp_q_head_num_, layer.head_dim_)
    return inputs


def test_bloom_layers_run_alibi_on_triton_whatever_the_command_line_asks(runtime_start_args, monkeypatch):
    from lightllm.models.bloom.layer_infer.transformer_layer_infer import BloomTransformerLayerInfer

    layer = BloomTransformerLayerInfer(0, {"num_attention_heads": H, "n_embed": H * D, "layer_norm_epsilon": 1e-5})
    model = make_model(FakeState(H))
    bind_layers(model, layer)
    assert layer_configs(layer) == (config(num_kv_heads=H, has_alibi=True),) * 2
    assert type(layer.decode_impl).__name__ == "TritonGQADecodeBackend"
    slopes = torch.ones(H)
    assert decode_inputs(monkeypatch, model, layer, SimpleNamespace(tp_alibi=slopes)).alibi_slopes is slopes


def test_starcoder_layers_run_multi_query_attention_without_alibi(runtime_start_args, monkeypatch):
    from lightllm.models.starcoder.layer_infer.transformer_layer_infer import StarcoderTransformerLayerInfer

    layer = StarcoderTransformerLayerInfer(0, {"num_attention_heads": H, "n_embed": H * D, "layer_norm_epsilon": 1e-5})
    model = make_model(FakeState(1))
    bind_layers(model, layer)
    assert layer_configs(layer) == (config(num_kv_heads=1),) * 2
    assert type(layer.decode_impl).__name__ == "FA3GQADecodeBackend"
    assert decode_inputs(monkeypatch, model, layer, SimpleNamespace()).alibi_slopes is None


def test_gemma4_sliding_and_full_layers_share_one_triton_group(runtime_start_args):
    from lightllm.models.gemma4.layer_infer.transformer_layer_infer import Gemma4TransformerLayerInfer

    net = dict(
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=512,
        head_dim=256,
        global_head_dim=512,
        num_global_key_value_heads=1,
        rms_norm_eps=1e-6,
        sliding_window=48,
        num_hidden_layers=2,
        layer_types=["sliding_attention", "full_attention"],
    )
    # The cache stores 2 slots of 256 per token: both sliding KV heads, or the single 512-dim full KV head.
    model = make_model(FakeState(2, 256))
    sliding, full = (Gemma4TransformerLayerInfer(i, net) for i in range(2))
    bind_layers(model, sliding, full)
    window = dict(num_heads=4, head_dim=256, sliding_window=(47, 0))
    assert layer_configs(sliding) == (config(**window, bidirectional_image_spans=True), config(**window))
    assert layer_configs(full) == (config(num_heads=4, num_kv_heads=1, head_dim=512),) * 2
    assert all(len(model.attention_backend_manager.backends(phase)) == 1 for phase in ("prefill", "decode"))


def test_gpt_oss_layers_have_sinks_and_sliding_layers_a_window(runtime_start_args, monkeypatch):
    from lightllm.models.gpt_oss.layer_infer.transformer_layer_infer import GptOssTransformerLayerInfer

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
    model = make_model(FakeState(2))
    sliding, full = (GptOssTransformerLayerInfer(i, net) for i in range(2))
    bind_layers(model, sliding, full)
    assert layer_configs(sliding) == (config(has_sinks=True, sliding_window=(127, 0)),) * 2
    assert layer_configs(full) == (config(has_sinks=True),) * 2
    sinks = torch.ones(H)
    weights = SimpleNamespace(attn_sinks=SimpleNamespace(weight=sinks))
    assert decode_inputs(monkeypatch, model, full, weights).sinks is sinks


def test_neo_layers_attend_inside_image_spans_in_prefill_only(runtime_start_args):
    from lightllm.models.neo_chat.layer_infer.transformer_layer_infer import NeoChatTransformerLayerInfer

    net = dict(num_attention_heads=H, num_key_value_heads=2, hidden_size=H * D, head_dim=D, rms_norm_eps=1e-6)
    layer = NeoChatTransformerLayerInfer(0, net)
    model = make_model(FakeState(2), backend="triton")
    bind_layers(model, layer)
    assert layer_configs(layer) == (config(bidirectional_image_spans=True), config())


@pytest.mark.parametrize("sliding_window, expected", [(0, (-1, -1)), (40, (39, 39))])
def test_dflash_layers_attend_only_in_decode_and_not_causally(monkeypatch, sliding_window, expected):
    from lightllm.models.qwen3_dflash2.layer_infer.transformer_layer_infer import Qwen3DFlash2TransformerLayerInfer

    monkeypatch.setattr(attention_backend_manager, "get_env_start_args", lambda: SimpleNamespace(mtp_mode="dflash2"))
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
    model = make_model(FakeState(2), draft=True)
    bind_layers(model, layer)
    assert layer_configs(layer) == (None, config(causal=False, sliding_window=expected))
    assert type(layer.decode_impl).__name__ == "FA3GQADecodeBackend"


def test_dflash_decode_needs_an_implementation_of_non_causal_attention(monkeypatch):
    from lightllm.models.qwen3_dflash.layer_infer.transformer_layer_infer import Qwen3DFlashTransformerLayerInfer

    monkeypatch.setattr(attention_backend_manager, "get_env_start_args", lambda: SimpleNamespace(mtp_mode="dflash"))
    net = dict(num_attention_heads=H, num_key_value_heads=2, hidden_size=H * D, head_dim=D, rms_norm_eps=1e-6)
    layer = Qwen3DFlashTransformerLayerInfer(0, net)
    with pytest.raises(ValueError, match="does not support"):
        model = make_model(FakeState(2), backend="triton", draft=True)
        bind_layers(model, layer)


@pytest.mark.parametrize("rope_scaling", [None, {"type": "yarn", "factor": 40, "mscale_all_dim": 1.0}])
def test_deepseek2_layers_attend_over_expanded_keys_in_prefill_and_the_latent_cache_in_decode(
    runtime_start_args, rope_scaling
):
    from lightllm.common.attention.backends.mla import MLADecodeLayerConfig, MLAPrefillLayerConfig
    from lightllm.models.deepseek2.layer_infer.transformer_layer_infer import Deepseek2TransformerLayerInfer
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
    model = make_model(None, backend="flashinfer")
    layers = [Deepseek2TransformerLayerInfer(i, net) for i in range(2)]
    bind_layers(model, *layers)
    common = dict(num_heads=16, qk_rope_head_dim=64, softmax_scale=scale, dtype=torch.bfloat16)
    assert layer_configs(layers[0]) == (
        MLAPrefillLayerConfig(qk_nope_head_dim=128, v_head_dim=128, **common),
        MLADecodeLayerConfig(kv_lora_rank=512, **common),
    )
    # FlashInfer plans with the layers' softmax scale, which they all share.
    assert layers[0].prefill_impl is layers[1].prefill_impl
    assert layers[0].decode_impl is layers[1].decode_impl


@pytest.mark.parametrize("kv_type, kv_cache_format", [("None", "bf16"), ("fp8kv_dsa", "fp8_ds_mla")])
def test_deepseek_v32_layers_attend_to_the_indexer_selection(runtime_start_args, kv_type, kv_cache_format):
    from lightllm.common.attention.backends.sparse_mla import SparseMLALayerConfig
    from lightllm.models.deepseek3_2.layer_infer.transformer_layer_infer import Deepseek3_2TransformerLayerInfer

    layer = Deepseek3_2TransformerLayerInfer.__new__(Deepseek3_2TransformerLayerInfer)
    layer.tp_q_head_num_, layer.kv_lora_rank, layer.qk_rope_head_dim = 64, 512, 64
    layer.softmax_scale, layer.index_topk = 0.1, 2048
    # Sparse MLA has one implementation name; auto-selection would validate it.
    model = make_model(None, backend="flashmla_sparse")
    model.args.llm_kv_type = kv_type
    bind_layers(model, layer)
    expected = SparseMLALayerConfig(
        num_heads=64,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        softmax_scale=0.1,
        dtype=torch.bfloat16,
        kv_cache_format=kv_cache_format,
        topk=2048,
    )
    assert layer_configs(layer) == (expected, expected)


@pytest.mark.parametrize("kv_type, expected_format", [("None", "bf16"), ("fp8kv_dsa", "fp8_group64_e8m0")])
def test_glm5_sparse_spec_describes_the_cache_without_a_workspace(kv_type, expected_format):
    from lightllm.models.glm5_next.layer_infer.transformer_layer_infer import Glm5NextTransformerLayerInfer

    layer = Glm5NextTransformerLayerInfer.__new__(Glm5NextTransformerLayerInfer)
    layer.is_linear_attention_layer = False
    layer.tp_q_head_num_, layer.kv_lora_rank, layer.softmax_scale = 64, 512, 0.0625
    layer.network_config_ = {"index_topk": 2048, "index_kpool": 4}
    model = SimpleNamespace(args=SimpleNamespace(llm_kv_type=kv_type), data_type=torch.bfloat16)
    spec = layer.attention_config(model)
    assert spec.prefill == spec.decode
    assert spec.prefill.kv_cache_format == expected_format
    assert spec.prefill_implementation == spec.decode_implementation == "flashmla_sparse"


@pytest.mark.parametrize("prefill_backends, prefill_kernel", [(["auto"], "auto"), (["fa3", "flashkda"], "flashkda")])
@pytest.mark.parametrize("ssm_state_mode", ["native", "replay"])
def test_glm5_kda_layers_follow_the_linear_backend_and_state_mode(
    runtime_start_args, monkeypatch, prefill_backends, prefill_kernel, ssm_state_mode
):
    from lightllm.common.attention.backends.kda import KDALayerConfig
    from lightllm.common.attention.backends.kda import triton as kda_triton
    from lightllm.models.glm5_next.layer_infer.transformer_layer_infer import Glm5NextTransformerLayerInfer

    monkeypatch.setattr(kda_triton, "_check_prefill_kernel", lambda layer_config: None)
    layer = Glm5NextTransformerLayerInfer.__new__(Glm5NextTransformerLayerInfer)
    layer.is_linear_attention_layer = True
    layer.tp_world_size_, layer.tp_linear_num_heads, layer.linear_head_dim = 1, 4, 128
    layer.network_config_ = {"linear_attn_config": {"num_heads": 4, "head_dim": 128, "short_conv_kernel_size": 4}}
    model = make_model(None, backend="triton")
    model.args.llm_prefill_att_backend = prefill_backends
    model.args.ssm_state_mode, model.args.linear_att_ssm_data_type = ssm_state_mode, "bfloat16"
    bind_layers(model, layer)
    expected = KDALayerConfig(
        4,
        128,
        4,
        -5.0,
        torch.bfloat16,
        replay=ssm_state_mode == "replay",
        ssm_dtype=torch.bfloat16,
        prefill_kernel=prefill_kernel,
    )
    assert layer_configs(layer) == (expected, expected)
