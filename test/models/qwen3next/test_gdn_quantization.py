# test/models/qwen3next/test_gdn_quantization.py
"""Tests for quantization support in GDN layers.

These tests verify that the GDN layer weight classes accept quantization configs
through their __init__ signatures, enabling INT8/INT4 quantization support.
"""
import inspect
import pytest
from lightllm.models.qwen3next.layer_weights.transformer_layer_weight import (
    Qwen3NextGatedDeltaNetTransformerLayerWeight,
    Qwen3NextFullAttentionTransformerLayerWeight,
)


def test_gdn_accepts_quant_cfg_parameter():
    """Test that GDN layer __init__ accepts quant_cfg parameter."""
    sig = inspect.signature(Qwen3NextGatedDeltaNetTransformerLayerWeight.__init__)
    params = sig.parameters

    # Verify quant_cfg parameter exists
    assert "quant_cfg" in params, "quant_cfg parameter not found in __init__"

    # Verify it has a default value of None
    assert params["quant_cfg"].default is None, "quant_cfg should default to None"


def test_full_attention_accepts_quant_cfg_parameter():
    """Test that Full Attention layer __init__ accepts quant_cfg parameter."""
    sig = inspect.signature(Qwen3NextFullAttentionTransformerLayerWeight.__init__)
    params = sig.parameters

    # Verify quant_cfg parameter exists
    assert "quant_cfg" in params, "quant_cfg parameter not found in __init__"

    # Verify it has a default value of None
    assert params["quant_cfg"].default is None, "quant_cfg should default to None"


def test_gdn_quant_cfg_passed_to_weights():
    """Test that GDN layer passes quant_cfg to weight initialization.

    This test verifies the plumbing by checking that the source code
    passes quant_cfg to the weight constructors (COLMMWeight, ROWMMWeight).
    """
    import lightllm.models.qwen3next.layer_weights.transformer_layer_weight as module
    import inspect

    source = inspect.getsource(module.Qwen3NextGatedDeltaNetTransformerLayerWeight._init_gdn_weight)

    # Verify that quant_cfg is passed to linear weights
    assert "quant_cfg=self.quant_cfg" in source, "GDN weights should pass quant_cfg to linear projections"

    # Count how many times it's passed (should be at least 3 for conv1d, in_proj, out_proj)
    count = source.count("quant_cfg=self.quant_cfg")
    assert count >= 3, f"Expected at least 3 quant_cfg passes, found {count}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
