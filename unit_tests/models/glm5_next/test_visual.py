import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from lightllm.models.glm5_next.tokenizer import Glm5NextTokenizer
from lightllm.models.glm5_next.vision_process import Glm5NextImageProcessor, smart_resize
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.server.multimodal_params import ImageItem, MultimodalParams
from lightllm.utils.envs_utils import set_env_start_args


@pytest.mark.parametrize("width,height,canvas", [(7, 11, (168, 112)), (173, 89, (112, 196)), (448, 448, (448, 448))])
def test_image_canvas_and_token_expansion(tmp_path, width, height, canvas):
    assert smart_resize(height, width) == canvas
    (tmp_path / "processor_config.json").write_text(json.dumps({"image_processor": {}}))
    tokenizer = Glm5NextTokenizer(
        None, {"image_start_token_id": 100, "image_end_token_id": 101, "image_token_id": 102}, str(tmp_path)
    )
    image = ImageItem(type="base64", data="")
    image.image_w, image.image_h = width, height
    image.token_id = 1000
    image.token_num = tokenizer.get_image_token_length(image)
    assert image.token_num == canvas[0] * canvas[1] // 28 ** 2
    multi = MultimodalParams()
    multi.images = [image]
    ids = tokenizer.encode([1, 100, 102, 101, 2], multi)
    assert ids == [1, 100, *range(1000, 1000 + image.token_num), 101, 2]
    assert image.start_idx == 2
    assert image.grid_thwd[-1] == 0
    assert tokenizer.encode([1, 2, 3]) == [1, 2, 3]
    assert tokenizer.encode([100, 102, 101]) == [100, 101]


def test_image_patch_order_temporal_repeat_and_black_padding():
    pixels = np.zeros((29, 31, 3), dtype=np.uint8)
    pixels[:, :, 0] = np.arange(31)[None, :]
    pixels[:, :, 1] = np.arange(29)[:, None]
    pixels[:, :, 2] = 127
    processor = Glm5NextImageProcessor(min_image_tokens=1, do_rescale=False, do_normalize=False)
    patches, grid = processor._preprocess_bydevice(Image.fromarray(pixels), device="cpu")
    assert grid.tolist() == [[1, 4, 4]]
    patches = patches.reshape(16, 3, 2, 14, 14)
    assert torch.equal(patches[:, :, 0], patches[:, :, 1])
    assert torch.equal(patches[0, :, 0], torch.from_numpy(pixels[:14, :14]).permute(2, 0, 1))
    assert torch.equal(patches[1, :, 0], torch.from_numpy(pixels[:14, 14:28]).permute(2, 0, 1))
    assert torch.equal(patches[2, :, 0], torch.from_numpy(pixels[14:28, :14]).permute(2, 0, 1))
    assert not patches[-1].any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_vision_batch_preserves_independent_image_attention():
    from lightllm.models.glm5_next.glm5_next_visual import Glm5NextVisionTransformer
    from lightllm.server.visualserver import set_vit_att_backend

    torch.manual_seed(53)
    set_vit_att_backend("sdpa")
    model = (
        Glm5NextVisionTransformer(
            {"data_type": "float32"},
            hidden_size=128,
            out_hidden_size=256,
            depth=2,
            intermediate_size=256,
            projection_intermediate_size=512,
            num_heads=2,
        )
        .cuda()
        .eval()
    )
    grid = torch.tensor([[1, 4, 6], [1, 6, 4]])
    x = torch.randn(48, 3 * 2 * 14 ** 2, device="cuda")
    with torch.inference_mode():
        batch = model(x, grid)
        individual = torch.cat([model(x[:24], grid[:1]), model(x[24:], grid[1:])])
    assert batch.shape == (12, 256)
    torch.testing.assert_close(batch, individual, atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_target_and_mtp_prefill_embed_images_before_hidden_fusion(monkeypatch):
    from lightllm.models.glm5_next.layer_infer.pre_layer_infer import Glm5NextPreLayerInfer
    from lightllm.models.glm5_next_mtp.layer_infer.pre_layer_infer import Glm5NextMTPPreLayerInfer
    from lightllm.server.router.model_infer.infer_batch import g_infer_context

    set_env_start_args(dataclasses.asdict(StartArgs(mtp_mode="eagle_with_att", mtp_step=2)))
    monkeypatch.setenv("LIGHTLLM_CURRENT_RANK_IN_DP", "0")
    monkeypatch.setenv("LIGHTLLM_DP_WORLD_SIZE", "1")
    hidden = 128
    config = {"hidden_size": hidden, "rms_norm_eps": 1e-5, "hc_mult": 4}
    image_cache = torch.randn(3, 1, hidden, pin_memory=True)
    monkeypatch.setattr(g_infer_context, "cpu_embed_cache_client", SimpleNamespace(cpu_embed_cache_tensor=image_cache))
    text = torch.randn(8, hidden, device="cuda")
    ids = torch.tensor([1, 1000, 1001, 2], device="cuda")
    metadata = {"token_id": 1000, "token_num": 2, "start_index_in_embed_cache": 1}
    state = SimpleNamespace(multimodal_params=[{"images": [metadata], "audios": []}])
    weight = SimpleNamespace(wte_weight_=SimpleNamespace(weight=text, tp_vocab_start_id=0, tp_vocab_end_id=8))
    expected = torch.cat((text[1:2], image_cache[1:, 0].cuda(), text[2:3]))
    expanded = Glm5NextPreLayerInfer(config).context_forward(ids, state, weight)
    torch.testing.assert_close(expanded.reshape(4, 4, hidden), expected[:, None].expand(-1, 4, -1))

    def norm(input, eps, out):
        out.copy_(input * torch.rsqrt(input.square().mean(-1, keepdim=True) + eps))
        return out

    state.mtp_draft_input_hiddens = torch.randn(4, hidden, device="cuda")
    main_norm_weight = torch.randn(hidden, device="cuda")
    old_hidden = state.mtp_draft_input_hiddens.clone()
    projection = torch.randn(2 * hidden, hidden, device="cuda") * 0.1
    weight.enorm_weight_ = weight.hnorm_weight_ = norm
    weight.main_norm_weight_ = lambda input, eps, out: out.copy_(
        torch.nn.functional.rms_norm(input, (hidden,), main_norm_weight, eps)
    )
    weight.eh_proj_weight_ = SimpleNamespace(mm=lambda x: x @ projection)
    actual = Glm5NextMTPPreLayerInfer(config).context_forward(ids, state, weight)
    old_hidden = torch.nn.functional.rms_norm(old_hidden, (hidden,), main_norm_weight, 1e-5)
    normalized = [x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5) for x in (expected, old_hidden)]
    torch.testing.assert_close(actual, torch.cat(normalized, -1) @ projection)
