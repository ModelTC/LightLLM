from types import SimpleNamespace

import pytest
import torch

from lightllm.models.qwen4_exp.checkpoint import normalize_text_weight_names, ngram_table_shape
from lightllm.models.qwen4_exp.layer_weights.transformer_layer_weight import Qwen4ExpTransformerLayerWeight
from lightllm.models.qwen4_exp.layer_weights.pre_and_post_layer_weight import Qwen4ExpPreAndPostLayerWeight
from lightllm.models.qwen4_exp.layer_infer.pre_layer_infer import Qwen4ExpPreLayerInfer
from lightllm.models.qwen4_exp.layer_infer.post_layer_infer import Qwen4ExpPostLayerInfer
from lightllm.common.quantization import Quantcfg
from .helpers import checkpoint, prepost_checkpoint, reference_hyper

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def test_scalar_gate_dispatch_is_model_owned(config):
    from lightllm.common.quantization.no_quant import NoQuantization
    from lightllm.common.quantization.scalar_projection import StableScalarProjection
    from lightllm.models.qwen3_5_moe.layer_weights.transformer_layer_weight import Qwen35MOETransformerLayerWeight

    native = Qwen4ExpTransformerLayerWeight(0, torch.bfloat16, config, Quantcfg(config))
    legacy = Qwen35MOETransformerLayerWeight(0, torch.bfloat16, config, Quantcfg(config))
    assert isinstance(native.shared_expert_gate.quant_method, StableScalarProjection)
    assert type(legacy.shared_expert_gate.quant_method) is NoQuantization


@pytest.mark.parametrize("layer", [0, 1, 3])
def test_incremental_native_checkpoint_loading(config, layer):
    torch.manual_seed(737)
    weights = checkpoint(config, layer)
    native = Qwen4ExpTransformerLayerWeight(layer, torch.bfloat16, config, Quantcfg(config))
    assert not native.attn_hyper_connection.verify_load()
    with pytest.raises(RuntimeError, match="not fully loaded"):
        _ = native.attn_hyper_connection.component

    for key in reversed(list(weights)):
        native.load_hf_weights({key: weights[key]})
    native.verify_load()
    assert not hasattr(native, "att_norm_weight_")
    assert not hasattr(native, "ffn_norm_weight_")
    if layer == 3:
        assert native.qk_norm_weight_.q_weight.dtype == torch.float32
        expected = 1 + weights[f"model.language_model.layers.{layer}.self_attn.q_norm.weight"].float().cuda()
        torch.testing.assert_close(native.qk_norm_weight_.q_weight, expected, atol=0, rtol=0)
        assert native.indexer.component.head_dim == 16
    else:
        assert native.linear_norm.activation == "sigmoid"
        hidden = torch.randn(3, 32, device="cuda", dtype=torch.bfloat16)
        actual = native.linear_in_proj.mm(hidden, use_custom_tensor_mananger=False)
        p = f"model.language_model.layers.{layer}.linear_attn"
        expected = torch.cat(
            [
                torch.nn.functional.linear(hidden, weights[p + f".in_proj_{name}.weight"].cuda())
                for name in ["qkv", "z", "b", "a"]
            ],
            -1,
        )
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    if layer == 1:
        assert native.ngram_embedding.weight.shape == (36, 8)
        assert native.ngram_hasher.component.heads == 4
        assert native.ple.component.context_len == 9


def test_embedding_widening_and_final_native_logits(config):
    torch.manual_seed(737)
    source = prepost_checkpoint(config)
    native = Qwen4ExpPreAndPostLayerWeight(torch.bfloat16, config)
    native.load_hf_weights(dict(source))
    native.verify_load()
    assert not hasattr(native, "final_norm_weight_")
    pre, post = Qwen4ExpPreLayerInfer(config), Qwen4ExpPostLayerInfer(config)
    state = SimpleNamespace(is_prefill=False, batch_size=2, prompt_logics=None, dist_group=None)
    ids = torch.tensor([2, 17], device="cuda", dtype=torch.int64)
    widened = pre.token_forward(ids, state, native)
    torch.testing.assert_close(
        widened, source["model.language_model.embed_tokens.weight"].cuda()[ids].repeat(1, 2), atol=0, rtol=0
    )
    widened = widened + torch.randn_like(widened) * 0.1
    actual = post.token_forward(widened, state, native)
    mixed = reference_hyper(widened, source, "model.language_model.hyper_connection_mixer", config, False)
    expected = torch.nn.functional.linear(mixed, source["lm_head.weight"].cuda()).float()
    torch.testing.assert_close(actual, expected, atol=0.016, rtol=0.016)


def test_namespace_and_allocation_metadata():
    value = torch.zeros(1)
    weights = {
        "model.language_model.x": value,
        "model.visual.x": value,
        "mtp.x": value,
        "adapter.model.language_model.x": value,
    }
    normalize_text_weight_names(weights)
    assert set(weights) == {"model.x", "model.visual.x", "mtp.x", "adapter.model.language_model.x"}
    with pytest.raises(ValueError, match="Conflicting"):
        normalize_text_weight_names({"model.x": value, "model.language_model.x": value.clone()})
    assert ngram_table_shape(20000000, 16, 0, 2560, 128) == (320001536, 160)
