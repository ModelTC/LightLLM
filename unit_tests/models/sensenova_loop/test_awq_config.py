import pytest

from lightllm.models.sensenova_loop.model import SenseNovaLoopTpPartModel


def test_hf_awq_config_preserves_embedding_and_lm_head_metadata():
    config = {
        "model_type": "sensenova_loop",
        "loop_settings": {"start": 4, "end": 28, "counts": [2]},
        "text_config": {
            "num_hidden_layers": 32,
            "hidden_size": 4096,
            "vocab_size": 248320,
            "full_attention_interval": 4,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
        },
        "quantization_config": {
            "quant_method": "awq",
            "bits": 4,
            "group_size": 128,
            "zero_point": True,
            "lm_head": True,
            "embedding": {"bits": 4, "group_size": 128, "layout": "row_major", "zero_point": True},
        },
    }
    config["text_config"]["layer_types"] *= 8
    normalized = SenseNovaLoopTpPartModel.normalize_model_config(config)
    assert normalized["quantization_config"] is config["quantization_config"]
    assert normalized["num_hidden_layers"] == 56
    assert len(normalized["loop_physical_indices"]) == 56
    assert normalized["loop_physical_indices"][4:8] == (4, 5, 6, 7)
