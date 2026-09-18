import json

import pytest
import torch
from transformers.modeling_rope_utils import _compute_yarn_parameters
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from lightllm.models.qwen2_vl.triton_kernel.mrope import mrope_triton_fused
from lightllm.models.qwen3_5.model import Qwen3_5TpPartModel


@pytest.fixture
def qwen35_model(tmp_path):
    # Qwen3.5-27B with the documented 4x YaRN extension.
    text_config = {
        "hidden_size": 5120,
        "head_dim": 256,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "num_hidden_layers": 64,
        "max_position_embeddings": 262144,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "rope_type": "yarn",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
            "factor": 4.0,
            "original_max_position_embeddings": 262144,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps({"text_config": text_config, "vision_config": {}}))
    model = Qwen3_5TpPartModel.__new__(Qwen3_5TpPartModel)
    model.weight_dir_ = str(tmp_path)
    model.finetune_config = None
    model.tp_world_size_ = 1
    model._init_config()
    model.head_dim_ = model.config["head_dim"]
    model.max_seq_length = 262148
    model.data_type = torch.float32
    return model


@pytest.mark.parametrize("type_key", ["rope_type", "type"])
@pytest.mark.parametrize("rope_type,expected", [("default", "default"), ("mrope", "default"), ("yarn", "yarn")])
def test_mrope_respects_explicit_scaling_type(qwen35_model, monkeypatch, type_key, rope_type, expected):
    rope_scaling = qwen35_model.config["rope_scaling"]
    rope_scaling.pop("rope_type")
    rope_scaling[type_key] = rope_type
    selected = []
    monkeypatch.setattr(qwen35_model, "_init_to_get_rotary", lambda: selected.append("default"))
    monkeypatch.setattr(qwen35_model, "_init_to_get_yarn_rotary", lambda: selected.append("yarn"))

    qwen35_model._init_custom()

    assert selected == [expected]


def _reference_yarn_cache(model, position_ids):
    config = Qwen3_5TextConfig(
        hidden_size=model.config["hidden_size"],
        head_dim=model.head_dim_,
        num_attention_heads=model.config["num_attention_heads"],
        max_position_embeddings=model.config["max_position_embeddings"],
        rope_parameters=model.config["rope_parameters"].copy(),
        partial_rotary_factor=model.config["partial_rotary_factor"],
    )
    inv_freq, attention_factor = _compute_yarn_parameters(config, position_ids.device)
    freqs = position_ids.float().unsqueeze(-1) * inv_freq
    return (freqs.cos() * attention_factor).to(model.data_type), (freqs.sin() * attention_factor).to(model.data_type)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for rotary caches")
@pytest.mark.parametrize("data_type", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("partial_rotary_factor", [0.25, 1.0])
@pytest.mark.parametrize("factor", [1.0, 4.0])
def test_yarn_cache_matches_transformers(qwen35_model, data_type, partial_rotary_factor, factor):
    model = qwen35_model
    model.data_type = data_type
    model.config["partial_rotary_factor"] = partial_rotary_factor
    model.config["rope_scaling"]["partial_rotary_factor"] = partial_rotary_factor
    model.config["rope_scaling"]["factor"] = factor

    model._init_custom()

    half_rotary_dim = int(model.head_dim_ * partial_rotary_factor) // 2
    assert model._cos_cached.shape == (model.max_seq_length, half_rotary_dim)
    assert model._sin_cached.shape == model._cos_cached.shape
    # Include both sides of the original context boundary.
    positions = torch.tensor([0, 1, 127, 8191, 262143, 262144, 262147], device="cuda")
    expected_cos, expected_sin = _reference_yarn_cache(model, positions)
    torch.testing.assert_close(model._cos_cached[positions], expected_cos, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(model._sin_cached[positions], expected_sin, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the MRoPE kernel")
@pytest.mark.parametrize("data_type", [torch.float32, torch.bfloat16])
def test_qwen35_yarn_interleaved_mrope_matches_reference(qwen35_model, data_type):
    model = qwen35_model
    model.data_type = data_type
    model._init_custom()
    position_ids = torch.tensor(
        [[0, 1, 127, 8191, 262143, 262147], [0, 7, 83, 8189, 262144, 262145], [0, 5, 61, 8190, 262145, 262146]],
        device="cuda",
    )
    cos, sin = _reference_yarn_cache(model, position_ids)
    half_rotary_dim = cos.shape[-1]
    rotary_dim = 2 * half_rotary_dim
    channels = torch.arange(half_rotary_dim, device="cuda")
    # [11, 11, 10] assigns the 32 frequencies to repeating T/H/W axes.
    cos = cos[channels % 3, :, channels].T.unsqueeze(1).float()
    sin = sin[channels % 3, :, channels].T.unsqueeze(1).float()
    cos = torch.cat((cos, cos), dim=-1)
    sin = torch.cat((sin, sin), dim=-1)

    def rotate_reference(x):
        result = x.clone()
        rotary = x[..., :rotary_dim].float()
        rotated_half = torch.cat((-rotary[..., half_rotary_dim:], rotary[..., :half_rotary_dim]), dim=-1)
        result[..., :rotary_dim] = (rotary * cos + rotated_half * sin).to(data_type)
        return result

    torch.manual_seed(0)
    q = torch.randn((position_ids.shape[1], 24, 256), dtype=data_type, device="cuda")
    k = torch.randn((position_ids.shape[1], 4, 256), dtype=data_type, device="cuda")
    expected_q, expected_k = rotate_reference(q), rotate_reference(k)

    mrope_triton_fused(
        q,
        k,
        model._cos_cached[position_ids],
        model._sin_cached[position_ids],
        torch.tensor(model.config["rope_scaling"]["mrope_section"], dtype=torch.int32, device="cuda"),
        is_interleaved=True,
        partial_rotary_factor=model.config["partial_rotary_factor"],
    )

    tolerance = 1e-5 if data_type == torch.float32 else 2e-2
    torch.testing.assert_close(q, expected_q, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(k, expected_k, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(q[..., rotary_dim:], expected_q[..., rotary_dim:], rtol=0, atol=0)
    torch.testing.assert_close(k[..., rotary_dim:], expected_k[..., rotary_dim:], rtol=0, atol=0)
