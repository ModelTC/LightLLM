from types import SimpleNamespace

from lightllm.models.gemma4.tokenizer import Gemma4Tokenizer
from lightllm.server.multimodal_params import ImageItem


class _Tokenizer:
    bos_token_id = 2

    def __call__(self, prompt, add_special_tokens=False):
        return SimpleNamespace(input_ids=prompt)


def test_tokenizer_sets_image_block_span():
    tokenizer = Gemma4Tokenizer(
        _Tokenizer(),
        {
            "image_token_id": 90,
            "boi_token_id": 91,
            "eoi_token_id": 92,
            "vision_soft_tokens_per_image": 3,
        },
    )
    image = ImageItem(type="image_size", data=(1, 1))
    image.token_id = 100
    image.token_num = 3

    result = tokenizer.encode(
        [7, 90, 90, 92, 8],
        SimpleNamespace(images=[image]),
    )

    assert result == [7, 91, 100, 101, 102, 92, 8]
    assert image.start_idx == 2
    assert image.block_start_idx == 2
    assert image.block_end_idx == 5
    assert image.to_dict()["block_start_idx"] == 2
    assert image.to_dict()["block_end_idx"] == 5
