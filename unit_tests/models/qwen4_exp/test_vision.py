import pytest
import torch

from lightllm.models.qwen4_exp.vision import Qwen4ExpVisionTransformer
from lightllm.server import visualserver
from lightllm.common.basemodel.attention_vit.create_utils import get_vit_att_backend_class
from .vision_reference import interpolate_positions, vision_reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_vision_encoder_matches_independent_packed_frames(dtype, monkeypatch):
    monkeypatch.setattr(visualserver, "VIT_ATTN_BACKEND", get_vit_att_backend_class("sdpa"))
    config = dict(
        depth=2,
        hidden_size=32,
        out_hidden_size=16,
        intermediate_size=48,
        hidden_act="gelu_pytorch_tanh",
        num_heads=4,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        num_position_embeddings=16,
        deepstack_visual_indexes=[],
    )
    torch.manual_seed(1907)
    model = Qwen4ExpVisionTransformer({"data_type": dtype}, **config).eval().to(device="cuda", dtype=dtype)
    grid = torch.tensor([[1, 4, 6], [2, 2, 4]], device="cuda")
    pixels = torch.randn(40, 24, device="cuda", dtype=dtype)
    source = {name: value.detach().clone() for name, value in model.state_dict().items()}
    tolerance = 0.016 if dtype == torch.bfloat16 else 2e-5
    with torch.no_grad():
        actual_positions = model.fast_pos_embed_interpolate(grid)
        expected_positions = interpolate_positions(source["pos_embed.weight"], grid, 2)
        torch.testing.assert_close(actual_positions, expected_positions, atol=tolerance, rtol=tolerance)
        actual, deepstack = model(pixels, grid)
        expected = vision_reference(pixels, grid, source, config)
    torch.testing.assert_close(actual_positions, expected_positions, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    assert not deepstack
    assert actual.shape == (10, 16)


def test_vision_rejects_unsupported_deepstack():
    with pytest.raises(ValueError, match="deepstack"):
        Qwen4ExpVisionTransformer({"data_type": "bf16"}, deepstack_visual_indexes=[1])
