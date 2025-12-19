import torch
import pytest
from lightllm.common.basemodel.triton_kernel.multimodal_emb import mark_multimodal_obj, multimodal_emb
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def test_mark_multimodal_obj():
    obj_start_ids = torch.tensor([1, 4, 100], device="cuda", dtype=torch.int64)
    obj_token_lens = torch.tensor([1, 3, 2], device="cuda", dtype=torch.int64)
    input_ids = torch.tensor([1, 7, 9, 333], device="cuda", dtype=torch.int64)

    mark_obj = mark_multimodal_obj(
        obj_start_token_ids=obj_start_ids, obj_token_lens=obj_token_lens, input_ids=input_ids
    )

    assert torch.equal(mark_obj, torch.tensor([1, 0, 0], device="cuda"))


def test_multimodal_emb():
    S, D = 1024 * 10, 128 * 64
    vob_size = 3200
    image_size = 10
    image_token_size = 512

    text_weight = torch.randn((vob_size, D), device="cuda", dtype=torch.float16)
    img_weight = torch.randn((image_size * image_token_size, 1, D), device="cuda", dtype=torch.float16)
    img_token_lens = torch.full((image_size,), image_token_size, device="cuda", dtype=torch.long)
    img_start_token_ids = (
        (torch.arange(0, image_size * image_token_size, image_token_size) + vob_size * 10).cuda().long()
    )
    img_start_locs = torch.arange(0, image_size * image_token_size, image_token_size).cuda().long()

    prompt_ids = torch.arange(0, S, 1).cuda().long()
    prompt_ids[0 : image_size * image_token_size] = (
        (vob_size * 10 + torch.arange(0, image_size * image_token_size, 1)).cuda().long()
    )

    out = torch.zeros((S, D), dtype=torch.float16, device="cuda")
    multimodal_emb(
        out, prompt_ids, text_weight, img_weight, img_token_lens, img_start_token_ids, img_start_locs, 0, vob_size, 1
    )
    assert torch.allclose(out[0 : image_size * image_token_size, :].view(-1), img_weight.view(-1), atol=0.0001)
    return


if __name__ == "__main__":
    pytest.main()
