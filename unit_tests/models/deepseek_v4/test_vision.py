import importlib.util
import io
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from lightllm.models.deepseek_v4 import deepseek_v4_visual as visual
from lightllm.models.deepseek_v4 import image_processor as ours


REFERENCE_DIR = "/mtc/models/DeepSeek-V4-Flash-Vision-Exp/inference"
REFERENCE_PROCESSOR_PATH = os.path.join(REFERENCE_DIR, "image_processor.py")
REFERENCE_VISION_PATH = os.path.join(REFERENCE_DIR, "vision.py")

PATCH_SIZE = 14
DOWNSAMPLE_RATIO = 3
MAX_N_TOKEN = 384
MIN_PIXELS = 147456
MAX_WH_RATIO = 8

REFERENCE_ARGS = SimpleNamespace(
    vision_patch_size=PATCH_SIZE,
    vision_downsample_ratio=DOWNSAMPLE_RATIO,
    vision_max_n_token=MAX_N_TOKEN,
    vision_min_pixels=MIN_PIXELS,
    vision_max_wh_ratio=MAX_WH_RATIO,
)


def _load_reference_processor():
    spec = importlib.util.spec_from_file_location("deepseek_v4_reference_image_processor", REFERENCE_PROCESSOR_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_reference_vision():
    spec = importlib.util.spec_from_file_location("deepseek_v4_reference_vision", REFERENCE_VISION_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def reference_processor():
    if not os.path.exists(REFERENCE_PROCESSOR_PATH):
        pytest.skip("DeepSeek-V4 vision reference processor is not available")
    return _load_reference_processor()


@pytest.fixture(scope="module")
def reference_vision():
    if not os.path.exists(REFERENCE_VISION_PATH):
        pytest.skip("DeepSeek-V4 reference vision tower is not available")
    return _load_reference_vision()


@pytest.mark.parametrize(
    "height,width",
    [(14, 14), (15, 30), (41, 512), (100, 100), (410, 756), (756, 42)],
)
def test_grid_tokens_matches_reference(reference_processor, height, width):
    assert ours.grid_tokens(height, width, PATCH_SIZE, DOWNSAMPLE_RATIO) == reference_processor.grid_tokens(
        height, width, PATCH_SIZE, DOWNSAMPLE_RATIO
    )


@pytest.mark.parametrize("start_pos", range(8))
@pytest.mark.parametrize("n_llm_h,n_llm_w", [(1, 1), (2, 3), (3, 2), (4, 5), (7, 8)])
def test_image_block_and_permutation_match_reference(reference_processor, n_llm_h, n_llm_w, start_pos):
    expected_types, expected_perm = reference_processor.build_image_block(n_llm_h, n_llm_w, start_pos)
    types, perm = ours.build_image_block(n_llm_h, n_llm_w, start_pos)
    assert torch.equal(types, expected_types)
    assert torch.equal(perm, expected_perm)


@pytest.mark.parametrize(
    "width,height",
    [(800, 600), (100, 2000), (2000, 100), (50, 50), (13, 7), (384, 384)],
)
def test_image_geometry_and_pixels_match_reference(reference_processor, width, height):
    rng = np.random.default_rng(width * 10000 + height)
    image = Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")

    expected = reference_processor.load_image({"data": image_bytes.getvalue()}, REFERENCE_ARGS)
    actual = ours.load_image(image, REFERENCE_ARGS)
    assert actual[1:] == expected[1:]
    assert torch.equal(actual[0], expected[0])

    n_vit_h, n_vit_w, n_llm_h, n_llm_w, token_num = ours.get_image_grid(
        height,
        width,
        PATCH_SIZE,
        DOWNSAMPLE_RATIO,
        MAX_N_TOKEN,
        MIN_PIXELS,
        MAX_WH_RATIO,
    )
    assert (n_vit_h, n_vit_w, n_llm_h, n_llm_w) == actual[1:]
    reference_types, _ = reference_processor.build_image_block(n_llm_h, n_llm_w, start_pos=0)
    assert token_num == len(reference_types)
    assert token_num <= MAX_N_TOKEN


def _tiny_model():
    return visual.DeepseekV4VisionModel(
        {"weight_dir": "/unused"},
        hidden_size=8,
        vision_n_layers=1,
        vision_dim=8,
        vision_n_heads=2,
        vision_inter_dim=12,
        vision_patch_size=2,
        vision_rope_theta=10000.0,
        vision_downsample_ratio=1,
        vision_max_n_token=64,
        vision_min_pixels=0,
        vision_max_wh_ratio=8,
    )


def _tiny_vision_args():
    return SimpleNamespace(
        hidden_size=12,
        dim=12,
        vision_n_layers=2,
        vision_dim=16,
        vision_n_heads=4,
        vision_inter_dim=24,
        vision_patch_size=2,
        vision_rope_theta=10000.0,
        vision_downsample_ratio=2,
    )


@pytest.mark.parametrize("n_vit_h,n_vit_w", [(4, 4), (5, 3)])
def test_vit_matches_reference(reference_vision, n_vit_h, n_vit_w):
    torch.manual_seed(0)
    args = _tiny_vision_args()
    model = visual.ViT(args).eval()
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        reference = reference_vision.ViT(args).eval()
    finally:
        torch.set_default_dtype(previous_dtype)
    reference.load_state_dict(model.state_dict())
    patches = torch.randn(
        n_vit_h * n_vit_w,
        3,
        args.vision_patch_size,
        args.vision_patch_size,
        dtype=torch.bfloat16,
    )
    with torch.inference_mode():
        actual = model(patches, n_vit_h, n_vit_w)
        expected = reference(patches, n_vit_h, n_vit_w)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("n_vit_h,n_vit_w", [(4, 4), (5, 3)])
def test_aligner_matches_reference(reference_vision, n_vit_h, n_vit_w):
    torch.manual_seed(0)
    args = _tiny_vision_args()
    model = visual.Aligner(args).eval()
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        reference = reference_vision.Aligner(args).eval()
    finally:
        torch.set_default_dtype(previous_dtype)
    reference.load_state_dict(model.state_dict())
    hidden = torch.randn(n_vit_h * n_vit_w, args.vision_dim, dtype=torch.bfloat16)
    with torch.inference_mode():
        actual = model(hidden, n_vit_h, n_vit_w)
        expected = reference(hidden, n_vit_h, n_vit_w)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_visual_parameter_dtypes():
    model = _tiny_model()
    for module in model.modules():
        if isinstance(module, nn.Linear):
            assert module.weight.dtype == torch.bfloat16
            if module.bias is not None:
                assert module.bias.dtype == torch.bfloat16
        elif isinstance(module, visual.RMSNorm):
            assert module.weight.dtype == torch.float32
    assert model.image_start.dtype == torch.bfloat16
    assert model.image_pad.dtype == torch.bfloat16


def test_load_model_reads_only_indexed_visual_weights(tmp_path, monkeypatch):
    model = _tiny_model()
    expected = {
        name: torch.full_like(tensor, index + 1) for index, (name, tensor) in enumerate(model.state_dict().items())
    }
    weight_map = {name: "visual.safetensors" for name in expected}
    weight_map["layers.0.self_attn.weight"] = "language.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))

    requested = []
    opened = []

    class FakeSafetensors:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def get_tensor(self, key):
            requested.append(key)
            return expected[key]

    def fake_safe_open(path, framework, device):
        opened.append(os.path.basename(path))
        assert framework == "pt"
        assert device == "cpu"
        return FakeSafetensors()

    monkeypatch.setattr(visual, "safe_open", fake_safe_open)
    model.load_model(str(tmp_path))

    assert opened == ["visual.safetensors"]
    assert set(requested) == set(expected)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, expected[name])


def test_encode_returns_canonical_cache_block(monkeypatch):
    model = _tiny_model()
    with torch.no_grad():
        model.image_start.fill_(10)
        model.image_pad.fill_(20)
        model.image_newline.fill_(30)
        model.image_end.fill_(40)

    class FakeVision(nn.Module):
        def forward(self, patches, n_h, n_w):
            return torch.zeros((n_h * n_w, 1), dtype=torch.bfloat16)

    class FakeAligner(nn.Module):
        def forward(self, x, n_h, n_w):
            values = torch.arange(n_h * n_w, dtype=torch.bfloat16)
            return values[:, None].expand(-1, model.config.hidden_size)

    model.vision = FakeVision()
    model.aligner = FakeAligner()

    image_bytes = io.BytesIO()
    Image.new("RGB", (6, 4), color=(1, 2, 3)).save(image_bytes, format="PNG")
    shm_names = []
    monkeypatch.setattr(visual, "get_shm_name_data", lambda uuid: f"{uuid}-data")

    def fake_read_shm(name):
        shm_names.append(name)
        return image_bytes.getvalue()

    monkeypatch.setattr(visual, "read_shm", fake_read_shm)
    embeds, uuids, valid_ids = model.encode([SimpleNamespace(uuid="image-1")])

    types, perm = ours.build_canonical_image_block(2, 3)
    sentinel_table = torch.stack(
        [
            model.image_start,
            model.image_pad,
            model.image_pad,
            model.image_newline,
            model.image_end,
        ]
    )
    expected = sentinel_table[types]
    row_major = torch.arange(6, dtype=torch.bfloat16)[:, None].expand(-1, model.config.hidden_size)
    expected[types == ours.IMAGE] = row_major[perm]

    assert shm_names == ["image-1-data"]
    assert uuids == ["image-1"]
    assert valid_ids == [[0, len(types)]]
    assert torch.equal(embeds, expected)
    assert torch.equal(types[:3], torch.full((3,), ours.IMAGE_PAD, dtype=torch.int64))
    assert types[3] == ours.IMAGE_START
    assert types[-1] == ours.IMAGE_END
