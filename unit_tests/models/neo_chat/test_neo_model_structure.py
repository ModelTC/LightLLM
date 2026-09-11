import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lightllm.models.neo_chat.infer_struct import NeoChatInferStateInfo
from lightllm.models.neo_chat.layer_infer.transformer_layer_infer import (
    NeoChatTransformerLayerInfer,
)
from lightllm.models.neo_chat.layer_weights.pre_and_post_layer_weight import (
    NeoChatPreAndPostLayerWeight,
)
from lightllm.models.neo_chat.layer_weights.transformer_layer_weight import (
    NeoChatTransformerLayerWeight,
)
from lightllm.models.neo_chat.model import NeoTpPartModel, build_neo_rope_tables
from lightllm.models.neo_chat.tokenizer import NeoChatTokenizer
from lightllm.models.neo_chat import vision_process as neo_vision_process
from lightllm.models.qwen2_vl import vision_process as qwen_vision_process
from lightllm.models.qwen3.layer_infer.transformer_layer_infer import (
    Qwen3TransformerLayerInfer,
)
from lightllm.models.qwen3.model import Qwen3TpPartModel
from lightllm.models.qwen_vl.layer_infer.pre_layer_infer import (
    LlamaMultimodalPreLayerInfer,
)
from lightllm.models.registry import get_model_class


class _FakeTokenizer:
    image_start_id = 101
    image_end_id = 102

    def convert_tokens_to_ids(self, token):
        return {
            "<img>": self.image_start_id,
            "</img>": self.image_end_id,
        }[token]

    def encode(self, prompt, add_special_tokens=True):
        assert add_special_tokens
        assert prompt == "describe <img></img> please"
        return [11, self.image_start_id, self.image_end_id, 12]


class _FakeImageItem:
    def __init__(self, *, width, height, token_id):
        self.image_w = width
        self.image_h = height
        self.token_id = token_id
        self.token_num = None
        self.start_idx = None
        self.grid_thwd = None
        self.extra_params = {}


def test_dense_neo_registry_and_class_wiring():
    config = {
        "model_type": "neo_chat",
        "llm_config": {"model_type": "qwen3"},
    }

    assert get_model_class(config) is NeoTpPartModel
    assert issubclass(NeoTpPartModel, Qwen3TpPartModel)
    assert NeoTpPartModel.pre_layer_infer_class is LlamaMultimodalPreLayerInfer
    assert NeoTpPartModel.transformer_layer_infer_class is NeoChatTransformerLayerInfer
    assert NeoTpPartModel.pre_and_post_weight_class is NeoChatPreAndPostLayerWeight
    assert NeoTpPartModel.transformer_weight_class is NeoChatTransformerLayerWeight
    assert NeoTpPartModel.infer_state_class is NeoChatInferStateInfo
    assert issubclass(NeoChatTransformerLayerInfer, Qwen3TransformerLayerInfer)


def test_neo_reuses_qwen_smart_resize():
    assert neo_vision_process.smart_resize is qwen_vision_process.smart_resize


def test_neo_generation_model_does_not_own_server_runtime():
    spec = importlib.util.find_spec("lightllm.models.neo_chat.generation")
    assert spec is not None
    source = Path(spec.origin).read_text(encoding="utf-8")
    assert "lightllm.server.generationserver" not in source


def test_build_neo_rope_tables_uses_temporal_and_spatial_widths():
    temporal_cos, temporal_sin, spatial_cos, spatial_sin = build_neo_rope_tables(
        head_dim=128,
        max_seq_len=17,
        rope_theta=10_000.0,
        rope_theta_hw=1_000.0,
        dtype=torch.float32,
        device="cpu",
    )

    assert temporal_cos.shape == temporal_sin.shape == (17, 32)
    assert spatial_cos.shape == spatial_sin.shape == (17, 16)
    torch.testing.assert_close(temporal_cos[0], torch.ones(32))
    torch.testing.assert_close(temporal_sin[0], torch.zeros(32))
    torch.testing.assert_close(spatial_cos[0], torch.ones(16))
    torch.testing.assert_close(spatial_sin[0], torch.zeros(16))


def test_nested_neo_config_is_flattened_for_the_qwen3_tower(tmp_path):
    outer_config = {
        "model_type": "neo_chat",
        "vision_config": {"hidden_size": 1024},
        "llm_config": {
            "model_type": "qwen3",
            "n_head": 8,
            "n_embd": 1024,
            "n_layer": 24,
            "vocab_size": 151_936,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(outer_config))

    model = object.__new__(NeoTpPartModel)
    model.weight_dir_ = str(tmp_path)
    model.finetune_config = None
    model._init_config()

    assert model.outer_config == outer_config
    assert model.config["model_type"] == "qwen3"
    assert model.config["num_attention_heads"] == 8
    assert model.config["hidden_size"] == 1024
    assert model.config["num_hidden_layers"] == 24
    assert model.config["num_key_value_heads"] == 8
    assert model.config["head_dim"] == 128


def test_tokenizer_image_length_and_image_tag_validation():
    adapter = NeoChatTokenizer(
        _FakeTokenizer(),
        {
            "image_token_id": 200,
            "vision_config": {
                "patch_size": 16,
                "downsample_ratio": 0.5,
                "min_pixels": 1,
                "max_pixels": 1_000_000,
            },
        },
    )
    image = _FakeImageItem(width=96, height=64, token_id=500)
    multimodal_params = SimpleNamespace(images=[image])

    # SamplingParams deliberately has no model-specific min/max pixel fields.
    adapter.init_imageitem_extral_params(
        image,
        multimodal_params,
        sampling_params=SimpleNamespace(),
    )
    image.token_num = adapter.get_image_token_length(image)

    assert image.extra_params == {"min_pixels": 1, "max_pixels": 1_000_000}
    assert image.token_num == 6
    assert image.grid_thwd == (1, 2, 3, -5)
    assert adapter.encode("describe <image> please", multimodal_params) == [
        11,
        adapter.image_start_id,
        500,
        501,
        502,
        503,
        504,
        505,
        adapter.image_end_id,
        12,
    ]
    assert image.start_idx == 2

    with pytest.raises(ValueError, match="must be followed by an image end token"):
        adapter.encode(
            [11, adapter.image_start_id, 99, adapter.image_end_id],
            multimodal_params,
            already_tokenized=True,
        )

    with pytest.raises(ValueError, match="0 image tags for 1 images"):
        adapter.encode([11, 12], multimodal_params, already_tokenized=True)
