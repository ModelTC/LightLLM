import importlib.util
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from lightllm.models.deepseek_v4 import deepseek_v4_visual as visual
from lightllm.models.deepseek_v4 import image_processor
from lightllm.models.deepseek_v41.deepseek_v41_visual import DeepseekV41VisionModel
from lightllm.models.deepseek_v41.tokenizer import DeepSeekV41Tokenizer
from lightllm.utils.config_utils import get_config_json, has_vision_module


@pytest.fixture
def config():
    return dict(
        model_type="deepseek_v41",
        image_token_id=63,
        text_config=dict(hidden_size=8, vocab_size=64),
        vision_config=dict(
            num_hidden_layers=1,
            hidden_size=8,
            num_attention_heads=2,
            intermediate_size=12,
            patch_size=2,
            rope_theta=10000,
            downsample_ratio=1,
            max_image_tokens=64,
            min_pixels=0,
            max_wh_ratio=None,
        ),
    )


@pytest.fixture(scope="module")
def reference():
    path = "/mtc/models/DeepSeek-V4.1-Flash/inference/image_processor.py"
    from pathlib import Path

    if not Path(path).is_file():
        pytest.skip("V4.1 reference image processor is not available")
    spec = importlib.util.spec_from_file_location("dsv41_reference_image_processor", path)
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "width,height", [(14, 14), (41, 512), (800, 600), (100, 2000), (2000, 100), (30000, 1), (1, 30000)]
)
def test_image_preprocessing_and_layout_match_checkpoint(reference, width, height):
    args = SimpleNamespace(
        vision_row_major=True,
        vision_patch_size=14,
        vision_downsample_ratio=3,
        vision_max_n_token=1024,
        vision_min_pixels=295936,
        vision_max_wh_ratio=None,
    )
    rng = np.random.default_rng(width * height)
    image = Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))
    raw = io.BytesIO()
    image.save(raw, format="PNG")
    expected = reference.load_image({"data": raw.getvalue()}, args)
    actual = image_processor.load_image(image, args)
    assert actual[1:] == expected[1:]
    torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
    n_h, n_w = actual[-2:]
    types, perm = image_processor.build_canonical_image_block(n_h, n_w, row_major=True)
    type_map = torch.tensor(
        [image_processor.IMAGE_START, image_processor.IMAGE, image_processor.IMAGE_NEW_LINE, image_processor.IMAGE_END]
    )
    assert torch.equal(types, type_map[reference.image_token_types(n_h, n_w)])
    assert torch.equal(perm, torch.arange(n_h * n_w))
    grid = image_processor.get_image_grid(height, width, 14, 3, 1024, 295936, None, row_major=True)
    assert grid[-1] == types.numel() <= 1024


@pytest.mark.parametrize("prefix_len", [0, 1, 2, 3, 7])
def test_vision_config_registry_and_tokenizer_preserve_row_major_positions(config, tmp_path, prefix_len):
    from lightllm.models.registry import ModelRegistry
    from lightllm.models.deepseek_v41.model import DeepseekV41VisionTpPartModel

    (tmp_path / "config.json").write_text(json.dumps(config))
    assert has_vision_module(str(tmp_path))
    assert ModelRegistry.get_model_class(config) is DeepseekV41VisionTpPartModel
    assert get_config_json(str(tmp_path)) == config
    tokenizer = DeepSeekV41Tokenizer(SimpleNamespace(convert_tokens_to_ids=lambda _: 63), str(tmp_path), config)
    image = SimpleNamespace(image_h=4, image_w=6, token_id=1000, token_num=10)
    assert tokenizer.get_image_token_length(image) == 10
    ids = tokenizer.encode([7] * prefix_len + [63, 8], SimpleNamespace(images=[image]))
    assert ids == [7] * prefix_len + list(range(1000, 1010)) + [8]
    assert image.start_idx == image.block_start_idx == prefix_len
    assert image.block_end_idx == prefix_len + 10


def test_visual_weight_loading_and_encode_without_image_padding(config, tmp_path, monkeypatch):
    from lightllm.common.basemodel.attention_vit.sdpa.fp import SdpaVitAttBackend
    from lightllm.server import visualserver

    monkeypatch.setattr(visualserver, "VIT_ATTN_BACKEND", SdpaVitAttBackend)
    (tmp_path / "config.json").write_text(json.dumps(config))
    config = get_config_json(str(tmp_path))
    model = DeepseekV41VisionModel({}, **config).eval()
    assert "image_pad" not in model.state_dict()
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.02)
    weights = model.state_dict()
    save_file(weights, str(tmp_path / "visual.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {name: "visual.safetensors" for name in weights},
            }
        )
    )
    loaded = DeepseekV41VisionModel({}, **config).eval().load_model(str(tmp_path))
    raw = io.BytesIO()
    image = Image.new("RGB", (6, 4), color=(1, 2, 3))
    image.save(raw, format="PNG")
    monkeypatch.setattr(visual, "read_shm", lambda _: raw.getvalue())
    embeds, uuids, spans = loaded.encode([SimpleNamespace(uuid=1)])
    patches, n_h, n_w, _, _ = image_processor.load_image(image, loaded.config)
    with torch.inference_mode():
        rows = loaded.aligner(loaded.vision(patches, n_h, n_w), n_h, n_w).view(n_h, n_w, -1)
    expected = torch.cat(
        [
            loaded.image_start[None],
            torch.cat([torch.cat([row, loaded.image_newline[None]]) for row in rows]),
            loaded.image_end[None],
        ]
    )
    torch.testing.assert_close(embeds, expected, atol=0, rtol=0)
    assert uuids == [1]
    assert spans == [[0, 10]]


def test_checkpoint_image_prompt_expansion_matches_reference(reference):
    from pathlib import Path
    from lightllm.server.tokenizer import get_tokenizer

    path = "/mtc/models/DeepSeek-V4.1-Flash"
    if not Path(path, "tokenizer.json").is_file():
        pytest.skip("V4.1 checkpoint tokenizer is not available")
    tokenizer = get_tokenizer(path)
    config = get_config_json(path)
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "unused.png"}},
                    {"type": "text", "text": "Describe this image."},
                ],
            },
        ],
        thinking=False,
    )
    raw = io.BytesIO()
    Image.new("RGB", (640, 480)).save(raw, format="PNG")
    vision = config["vision_config"]
    args = SimpleNamespace(
        vision_enabled=True,
        image_token_id=config["image_token_id"],
        vision_patch_size=vision["patch_size"],
        vision_downsample_ratio=vision["downsample_ratio"],
        vision_max_n_token=vision["max_image_tokens"],
        vision_min_pixels=vision["min_pixels"],
        vision_max_wh_ratio=vision.get("max_wh_ratio"),
    )
    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"encoding": tokenizer._get_encoding_module()}):
        expected_ids, _, images = reference.prepare_vl_inputs(prompt, [{"data": raw.getvalue()}], tokenizer, args)
    vocab_size = config["text_config"]["vocab_size"]
    image = SimpleNamespace(image_h=480, image_w=640, token_id=vocab_size + 1000)
    image.token_num = tokenizer.get_image_token_length(image)
    actual = tokenizer.encode(prompt, SimpleNamespace(images=[image]))
    assert image.start_idx == images[0].start
    assert image.token_num == images[0].types.numel()
    assert [config["image_token_id"] if token >= vocab_size else token for token in actual] == expected_ids
