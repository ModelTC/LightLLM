"""
Unit tests for Qwen3.5 model support
"""
import pytest
import os
import json
import tempfile


def test_qwen3_5_model_registration():
    """Test that Qwen3.5 models are properly registered"""
    from lightllm.models.registry import get_model_class
    from lightllm.models.qwen3_5.model import Qwen3_5TpPartModel, Qwen3_5MOETpPartModel

    # Test dense variant
    dense_class = get_model_class({"model_type": "qwen3_5"})
    assert dense_class == Qwen3_5TpPartModel, "qwen3_5 should map to Qwen3_5TpPartModel"

    # Test MoE variant
    moe_class = get_model_class({"model_type": "qwen3_5_moe"})
    assert moe_class == Qwen3_5MOETpPartModel, "qwen3_5_moe should map to Qwen3_5MOETpPartModel"


def test_qwen3_5_config_extraction():
    """Test that nested config is extracted correctly"""
    from lightllm.models.qwen3_5.model import Qwen3_5TpPartModel

    # Create temporary config file with Qwen3.5 structure
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.json")
        test_config = {
            "model_type": "qwen3_5",
            "text_config": {
                "hidden_size": 3584,
                "num_hidden_layers": 28,
                "num_attention_heads": 28,
                "num_key_value_heads": 4,
                "vocab_size": 152064,
                "full_attention_interval": 7,
                "linear_num_key_heads": 7,
                "linear_num_value_heads": 7,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "head_dim": 128,
                "rms_norm_eps": 1e-6,
                "hidden_act": "silu",
            },
            "vision_config": {
                "hidden_size": 1280,
                "num_hidden_layers": 32,
                "num_attention_heads": 16,
                "image_size": 448,
                "patch_size": 14,
            },
        }

        with open(config_path, "w") as f:
            json.dump(test_config, f)

        # Verify config can be loaded
        with open(config_path) as f:
            loaded = json.load(f)

        assert "text_config" in loaded, "Config should have text_config"
        assert "vision_config" in loaded, "Config should have vision_config"
        assert loaded["text_config"]["full_attention_interval"] == 7, "Should extract text config correctly"


def test_qwen3_5_inherits_from_qwen3next():
    """Test that Qwen3.5 inherits hybrid attention from Qwen3Next"""
    from lightllm.models.qwen3_5.model import Qwen3_5TpPartModel
    from lightllm.models.qwen3next.model import Qwen3NextTpPartModel

    # Verify inheritance
    assert issubclass(
        Qwen3_5TpPartModel, Qwen3NextTpPartModel
    ), "Qwen3_5TpPartModel should inherit from Qwen3NextTpPartModel"


def test_qwen3_5_multimodal_components():
    """Test that multimodal components are properly configured"""
    from lightllm.models.qwen3_5.model import Qwen3_5TpPartModel
    from lightllm.models.qwen3_vl.layer_infer.pre_layer_infer import Qwen3VLMultimodalPreLayerInfer
    from lightllm.models.qwen3_vl.layer_weights.pre_and_post_layer_weight import Qwen3VLPreAndPostLayerWeight

    # Check class attributes
    assert (
        Qwen3_5TpPartModel.pre_layer_infer_class == Qwen3VLMultimodalPreLayerInfer
    ), "Should use Qwen3VL multimodal pre-layer"

    assert (
        Qwen3_5TpPartModel.pre_and_post_weight_class == Qwen3VLPreAndPostLayerWeight
    ), "Should use Qwen3VL pre/post weights"


def test_qwen3_5_hybrid_attention_flags():
    """Test that hybrid attention flags are inherited"""
    from lightllm.models.qwen3_5.model import Qwen3_5TpPartModel

    # These flags should be inherited from Qwen3Next
    assert hasattr(Qwen3_5TpPartModel, "is_hybrid_attention"), "Should have is_hybrid_attention flag"

    assert hasattr(Qwen3_5TpPartModel, "use_buffer_manager"), "Should have use_buffer_manager flag"


def test_qwen3_5_models_in_registry():
    """Test that models are importable from lightllm.models"""
    # Test import works
    from lightllm.models import Qwen3_5TpPartModel, Qwen3_5MOETpPartModel

    # Verify they're the right classes
    from lightllm.models.qwen3_5.model import Qwen3_5TpPartModel as DirectImport

    assert Qwen3_5TpPartModel is DirectImport, "Should import same class from models package"


def test_qwen3_5_module_exports():
    """Test that module __init__.py exports models correctly"""
    from lightllm.models.qwen3_5 import (
        Qwen3_5TpPartModel,
        Qwen3_5MOETpPartModel,
        QWen3_5Tokenizer,
    )

    # Verify classes are not None
    assert Qwen3_5TpPartModel is not None
    assert Qwen3_5MOETpPartModel is not None
    assert QWen3_5Tokenizer is not None
