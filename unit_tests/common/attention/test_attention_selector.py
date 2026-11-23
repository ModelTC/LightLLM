"""The selector chooses an implementation per layer config and phase, skipping implementations that reject it."""

import pytest
import torch

from lightllm.common.attention import selector
from lightllm.common.attention.backends.gqa import GQALayerConfig


def gqa(**fields) -> GQALayerConfig:
    return GQALayerConfig(**{"num_heads": 8, "num_kv_heads": 2, "head_dim": 64, "dtype": torch.bfloat16, **fields})


def recorder(passing=("fa3", "flashinfer", "triton")):
    validated = []

    def validate(name):
        validated.append(name)
        return name in passing

    return validated, validate


@pytest.mark.parametrize("phase", selector.PHASES)
@pytest.mark.parametrize("name", ["triton", "fa3", "flashinfer"])
def test_named_implementation_is_used_without_validation(phase, name):
    validated, validate = recorder(passing=())
    selection = selector.select(gqa(), phase, name, validate=validate)
    assert selection.backend.get_name() == f"{name}_gqa_{phase}"
    assert selection.impl.layer_config == gqa()
    assert validated == []


def test_named_implementation_that_rejects_the_config_raises():
    with pytest.raises(ValueError, match="flashinfer.*does not support"):
        selector.select(gqa(sliding_window=(15, 0)), "prefill", "flashinfer")


def test_unknown_implementation_name_raises():
    with pytest.raises(ValueError, match="not available"):
        selector.select(gqa(), "decode", "flashmla_sparse")


def test_one_name_covers_the_quantized_triton_backend():
    config = gqa(kv_quant="int8_group", kv_quant_group_size=8)
    assert selector.select(config, "decode", "triton").backend.get_name() == "triton_quant_gqa_decode"


def test_auto_follows_priority_and_validates_only_implementations_that_accept_the_config():
    validated, validate = recorder()
    assert selector.select(gqa(), "prefill", "auto", validate=validate).backend.get_name() == "fa3_gqa_prefill"
    assert validated == ["fa3"]

    # FA3 and FlashInfer reject an int8 cache, so neither is validated.
    validated, validate = recorder()
    config = gqa(kv_quant="int8_group", kv_quant_group_size=8)
    assert selector.select(config, "decode", "auto", validate=validate).backend.get_name() == "triton_quant_gqa_decode"
    assert validated == ["triton"]


def test_auto_skips_unsupported_configs_and_failed_validation():
    # FlashInfer has no sliding window, so decode moves on to FA3.
    validated, validate = recorder()
    selection = selector.select(gqa(sliding_window=(15, 0)), "decode", "auto", validate=validate)
    assert selection.backend.get_name() == "fa3_gqa_decode"
    assert validated == ["fa3"]

    validated, validate = recorder(passing=("triton",))
    assert selector.select(gqa(), "decode", "auto", validate=validate).backend.get_name() == "triton_gqa_decode"
    assert validated == ["flashinfer", "fa3", "triton"]


def test_auto_honours_priority_and_exclusions():
    _, validate = recorder()
    selection = selector.select(
        gqa(), "decode", "auto", priority=selector.auto_priority(gqa(), "decode", 1), validate=validate
    )
    assert selection.backend.get_name() == "fa3_gqa_decode"
    selection = selector.select(gqa(), "decode", "auto", excluded=("flashinfer",), validate=validate)
    assert selection.backend.get_name() == "fa3_gqa_decode"


def test_auto_falls_back_to_triton_without_validation():
    validated, validate = recorder(passing=())
    assert selector.select(gqa(), "prefill", "auto", validate=validate).backend.get_name() == "triton_gqa_prefill"
    assert validated == ["fa3", "flashinfer", "triton"]

    # Only FA3 runs an FP8 per-head cache; without it nothing is left.
    with pytest.raises(ValueError, match="no prefill attention implementation supports"):
        selector.select(gqa(kv_quant="fp8_per_head"), "prefill", "auto", validate=lambda name: False)


def test_decode_priority_puts_fa3_first_with_mtp():
    assert selector.auto_priority(gqa(), "decode", 0) == ("flashinfer", "fa3", "triton")
    assert selector.auto_priority(gqa(), "decode", 2) == ("fa3", "flashinfer", "triton")
    assert selector.auto_priority(gqa(), "prefill", 2) == ("fa3", "flashinfer", "triton")


def test_unregistered_layer_config_type_is_rejected():
    with pytest.raises(NotImplementedError, match="no attention implementations"):
        selector.select(object(), "prefill", "auto")


def test_mla_phases_have_their_own_layer_config_types():
    from lightllm.common.attention.backends.mla import MLADecodeLayerConfig, MLAPrefillLayerConfig

    common = dict(num_heads=16, qk_rope_head_dim=64, softmax_scale=0.1, dtype=torch.bfloat16)
    prefill = MLAPrefillLayerConfig(qk_nope_head_dim=128, v_head_dim=128, **common)
    decode = MLADecodeLayerConfig(kv_lora_rank=512, **common)
    assert selector.select(prefill, "prefill", "flashinfer").backend.get_name() == "flashinfer_mla_prefill"
    assert selector.select(decode, "decode", "fa3").backend.get_name() == "fa3_mla_decode"
    assert selector.auto_priority(decode, "decode", 1) == ("fa3", "flashinfer", "triton")
    with pytest.raises(ValueError, match="no decode attention implementations"):
        selector.select(prefill, "decode", "triton")


@pytest.mark.parametrize(
    "kv_cache_format, rope, expected",
    [
        ("bf16", 64, ("flashmla_sparse_mla_prefill", "fa3_sparse_mla_decode")),
        ("fp8_ds_mla", 64, ("flashmla_sparse_mla_prefill", "flashmla_sparse_mla_decode")),
        ("bf16", 0, ("flashmla_sparse_mla_prefill", "fa3_sparse_mla_decode")),
        ("fp8_group64_e8m0", 0, ("fa3_sparse_mla_prefill", "flashmla_sparse_mla_decode")),
    ],
)
def test_sparse_mla_runs_flashmla_where_it_supports_the_cache_and_fa3_elsewhere(kv_cache_format, rope, expected):
    from lightllm.common.attention.backends.sparse_mla import SparseMLALayerConfig

    config = SparseMLALayerConfig(
        num_heads=64,
        kv_lora_rank=512,
        qk_rope_head_dim=rope,
        softmax_scale=0.1,
        dtype=torch.bfloat16,
        kv_cache_format=kv_cache_format,
        topk=2048,
    )
    names = tuple(selector.select(config, phase, "flashmla_sparse").backend.get_name() for phase in selector.PHASES)
    assert names == expected
    assert selector.auto_priority(config, "decode", 2) == ("flashmla_sparse",)
    with pytest.raises(ValueError, match="not available"):
        selector.select(config, "decode", "fa3")
