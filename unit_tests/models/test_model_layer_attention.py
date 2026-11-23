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
        att_state_manager=SimpleNamespace(get_state=lambda **query: state),
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

    assert StarcoderTpPartModel.uses_attention_groups and StarcoderTpPartModel.att_implementation is None
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
