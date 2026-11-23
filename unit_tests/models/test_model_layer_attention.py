"""Model layers create their Attention with their own layer properties when the model loads."""

from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention.backends.gqa import GQALayerConfig
from lightllm.common.basemodel import attention_groups
from lightllm.common.basemodel.attention_groups import AttentionGroups

H, D = 8, 64


class FakeState:
    def __init__(self, kv_heads, head_dim=D):
        self.k = torch.empty(16, kv_heads, head_dim)

    def get_layer_view(self, logical_layer_id):
        return self.k, self.k

    def get_layer_kv_scales(self, logical_layer_id):
        return None


def make_model(state, model_cls=None, backend="fa3", draft=False):
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
        att_implementation=getattr(model_cls, "att_implementation", None),
        data_type=torch.bfloat16,
        att_state_role="main",
        att_state_manager=SimpleNamespace(
            get_state=lambda **query: state, get_bounded_attention_states=lambda role: ()
        ),
    )
    model.att_groups = AttentionGroups(model)
    return model


def config(**fields) -> GQALayerConfig:
    return GQALayerConfig(**{"num_heads": H, "num_kv_heads": 2, "head_dim": D, "dtype": torch.bfloat16, **fields})


def layer_configs(layer):
    prefill = None if layer.attn.prefill_impl is None else layer.attn.prefill_impl.layer_config
    return prefill, layer.attn.decode_impl.layer_config


def test_bloom_layers_run_alibi_on_triton_whatever_the_command_line_asks(runtime_start_args):
    from lightllm.models.bloom.layer_infer.transformer_layer_infer import BloomTransformerLayerInfer
    from lightllm.models.bloom.model import BloomTpPartModel

    layer = BloomTransformerLayerInfer(0, {"num_attention_heads": H, "n_embed": H * D, "layer_norm_epsilon": 1e-5})
    layer.init_attention(make_model(FakeState(H), BloomTpPartModel))
    assert layer_configs(layer) == (config(num_kv_heads=H, has_alibi=True),) * 2
    assert layer.attn.group == "triton_gqa_prefill+triton_gqa_decode"
    assert layer._gqa_layer_weights(SimpleNamespace(tp_alibi="slopes")) == {"alibi_slopes": "slopes"}


def test_starcoder_layers_run_multi_query_attention_without_alibi(runtime_start_args):
    from lightllm.models.starcoder.layer_infer.transformer_layer_infer import StarcoderTransformerLayerInfer
    from lightllm.models.starcoder.model import StarcoderTpPartModel

    assert StarcoderTpPartModel.att_implementation is None
    layer = StarcoderTransformerLayerInfer(0, {"num_attention_heads": H, "n_embed": H * D, "layer_norm_epsilon": 1e-5})
    layer.init_attention(make_model(FakeState(1)))
    assert layer_configs(layer) == (config(num_kv_heads=1),) * 2
    assert layer.attn.group == "fa3_gqa_prefill+fa3_gqa_decode"
    assert layer._gqa_layer_weights(SimpleNamespace(tp_alibi="slopes")) == {}


def test_gemma4_sliding_and_full_layers_share_one_triton_group(runtime_start_args):
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
        num_hidden_layers=2,
        layer_types=["sliding_attention", "full_attention"],
    )
    # The cache stores 2 slots of 256 per token: both sliding KV heads, or the single 512-dim full KV head.
    model = make_model(FakeState(2, 256), Gemma4TpPartModel)
    sliding, full = (Gemma4TransformerLayerInfer(i, net) for i in range(2))
    sliding.init_attention(model)
    full.init_attention(model)
    window = dict(num_heads=4, head_dim=256, sliding_window=(47, 0))
    assert layer_configs(sliding) == (config(**window, bidirectional_image_spans=True), config(**window))
    assert layer_configs(full) == (config(num_heads=4, num_kv_heads=1, head_dim=512),) * 2
    assert sliding.attn.group == full.attn.group == "triton_gqa_prefill+triton_gqa_decode"


def test_gpt_oss_layers_have_sinks_and_sliding_layers_a_window(runtime_start_args):
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
    sliding.init_attention(model)
    full.init_attention(model)
    assert layer_configs(sliding) == (config(has_sinks=True, sliding_window=(127, 0)),) * 2
    assert layer_configs(full) == (config(has_sinks=True),) * 2
    weights = SimpleNamespace(attn_sinks=SimpleNamespace(weight="sinks"))
    assert full._gqa_layer_weights(weights) == {"sinks": "sinks"}


def test_neo_layers_attend_inside_image_spans_in_prefill_only(runtime_start_args):
    from lightllm.models.neo_chat.layer_infer.transformer_layer_infer import NeoChatTransformerLayerInfer

    net = dict(num_attention_heads=H, num_key_value_heads=2, hidden_size=H * D, head_dim=D, rms_norm_eps=1e-6)
    layer = NeoChatTransformerLayerInfer(0, net)
    layer.init_attention(make_model(FakeState(2), backend="triton"))
    assert layer_configs(layer) == (config(bidirectional_image_spans=True), config())


@pytest.mark.parametrize("sliding_window, expected", [(0, (-1, -1)), (40, (39, 39))])
def test_dflash_layers_attend_only_in_decode_and_not_causally(monkeypatch, sliding_window, expected):
    from lightllm.models.qwen3_dflash2.layer_infer.transformer_layer_infer import Qwen3DFlash2TransformerLayerInfer

    monkeypatch.setattr(attention_groups, "get_env_start_args", lambda: SimpleNamespace(mtp_mode="dflash2"))
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
    layer.init_attention(make_model(FakeState(2), draft=True))
    assert layer_configs(layer) == (None, config(causal=False, sliding_window=expected))
    assert layer.attn.group == "-+fa3_gqa_decode"


def test_dflash_decode_needs_an_implementation_of_non_causal_attention(monkeypatch):
    from lightllm.models.qwen3_dflash.layer_infer.transformer_layer_infer import Qwen3DFlashTransformerLayerInfer

    monkeypatch.setattr(attention_groups, "get_env_start_args", lambda: SimpleNamespace(mtp_mode="dflash"))
    net = dict(num_attention_heads=H, num_key_value_heads=2, hidden_size=H * D, head_dim=D, rms_norm_eps=1e-6)
    layer = Qwen3DFlashTransformerLayerInfer(0, net)
    with pytest.raises(ValueError, match="does not support"):
        layer.init_attention(make_model(FakeState(2), backend="triton", draft=True))


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
    for layer in layers:
        layer.init_attention(model)
    common = dict(num_heads=16, qk_rope_head_dim=64, softmax_scale=scale, dtype=torch.bfloat16)
    assert layer_configs(layers[0]) == (
        MLAPrefillLayerConfig(qk_nope_head_dim=128, v_head_dim=128, **common),
        MLADecodeLayerConfig(kv_lora_rank=512, **common),
    )
    # FlashInfer plans with the layers' softmax scale, which they all share.
    assert layers[0].attn.group == layers[1].attn.group == "flashinfer_mla_prefill+flashinfer_mla_decode"


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
    layer.init_attention(model)
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
    layer.init_attention(model)
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
    assert layer.linear_replay == (ssm_state_mode == "replay")
