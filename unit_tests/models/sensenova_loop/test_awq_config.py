import pytest

from lightllm.models.sensenova_loop.model import SenseNovaLoopTpPartModel


def test_runtime_scale_export_keeps_execution_depth_and_rejects_other_layouts():
    config = {
        "model_type": "sensenova_loop",
        "loop_settings": {"start": 4, "end": 28, "eval_count": 2, "use_gate": False},
        "text_config": {
            "num_hidden_layers": 32,
            "full_attention_interval": 4,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
        },
        "loop_awq_config": {
            "bits": 4,
            "group_size": 128,
            "zero_point": True,
            "linear_layout": "awq_gemm",
            "input_scale_mode": "runtime",
            "embedding_layout": "row_major_int4",
        },
    }
    config["text_config"]["layer_types"] *= 8
    normalized = SenseNovaLoopTpPartModel.normalize_model_config(config)
    assert normalized["num_hidden_layers"] == 56
    assert normalized["layer_types"].count("full_attention") == 14
    assert normalized["loop_awq_config"]["input_scale_mode"] == "runtime"
    assert normalized["quantization_config"]["quant_method"] == "awq"
    for changes in ({"input_scale_mode": "folded"}, {"group_size": 64}, {"embedding_layout": "awq_gemm"}):
        invalid = dict(config, loop_awq_config=config["loop_awq_config"] | changes)
        with pytest.raises(ValueError, match="Unsupported Loop AWQ export"):
            SenseNovaLoopTpPartModel.normalize_model_config(invalid)
