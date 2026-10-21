from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lightllm.models.gemma3.model import Gemma3Tokenizer
from lightllm.models.internvl.model import InternvlTokenizer
from lightllm.models.llava.model import LlavaTokenizer
from lightllm.models.qwen3_omni_moe_thinker.model import QWen3OmniTokenizer
from lightllm.models.qwen_vl.model import QWenVLTokenizer
from lightllm.models.tarsier2.model import Tarsier2Tokenizer
from lightllm.server.multimodal_params import MultimodalParams
from lightllm.utils.multimodal_chunk_utils import get_mm_input_spans


def make_params(images=(), audios=()):
    params = MultimodalParams(
        images=[{"type": "url", "data": "unused"} for _ in images],
        audios=[{"type": "url", "data": "unused"} for _ in audios],
    )
    for items, specs in ((params.images, images), (params.audios, audios)):
        for item, (token_id, token_num) in zip(items, specs):
            item.token_id, item.token_num = token_id, token_num
    return params


def make_tokenizer(tokenizer_class, origin_ids):
    tokenizer = tokenizer_class.__new__(tokenizer_class)
    tokenizer.tokenizer = Mock()
    tokenizer.tokenizer.encode.return_value = origin_ids
    tokenizer.image_start_id, tokenizer.image_end_id, tokenizer.image_token_id = 10, 11, 12
    tokenizer.audio_start_id, tokenizer.audio_end_id, tokenizer.audio_token_id = 20, 21, 22
    return tokenizer


@pytest.mark.parametrize(
    "tokenizer_class", [LlavaTokenizer, Gemma3Tokenizer, QWenVLTokenizer, Tarsier2Tokenizer, InternvlTokenizer]
)
def test_images_record_positions_even_when_embeddings_are_reused(tokenizer_class):
    tokenizer = make_tokenizer(tokenizer_class, [1, 10, 11, 2, 10, 11, 3])
    tokenizer.tokenizer.side_effect = [SimpleNamespace(input_ids=[token]) for token in (1, 2, 3)]
    tokenizer.tokenizer.tokenizer = tokenizer.tokenizer
    tokenizer.image_token = tokenizer.boi_token = "<image>"
    tokenizer.image_length = tokenizer.mm_tokens_per_image = 2
    tokenizer.boi_token_index, tokenizer.eoi_token_index = 10, 11
    tokenizer.image_start_tag, tokenizer.image_end_tag = "<img>", "</img>"
    tokenizer.skip_start = False
    params = make_params(images=[(100, 2), (100, 2)])

    prompt = "a<img></img>b<img></img>c" if tokenizer_class is QWenVLTokenizer else "a<image>b<image>c"
    kwargs = {"add_special_tokens": False} if tokenizer_class is InternvlTokenizer else {}
    input_ids = tokenizer.encode(prompt, params, **kwargs)

    if tokenizer_class is LlavaTokenizer:
        assert input_ids == [1, 100, 101, 2, 100, 101, 3]
        expected_spans = [(1, 3), (4, 6)]
    else:
        assert input_ids == [1, 10, 100, 101, 11, 2, 10, 100, 101, 11, 3]
        expected_spans = [(2, 4), (7, 9)]
    assert get_mm_input_spans(params.to_dict()) == expected_spans


@pytest.mark.parametrize("tokenizer_class", [InternvlTokenizer, QWen3OmniTokenizer])
@pytest.mark.parametrize(
    "origin_ids,image_starts,audio_starts,expected_ids",
    [
        (
            [1, 20, 21, 2, 10, 11, 3, 20, 21, 4, 10, 11, 5],
            [7, 19],
            [2, 13],
            [1, 20, 200, 201, 21, 2, 10, 100, 101, 102, 11, 3, 20, 300, 301, 302, 21, 4, 10, 100, 101, 102, 11, 5],
        ),
        (
            [1, 10, 11, 2, 20, 21, 3, 10, 11, 4, 20, 21, 5],
            [2, 13],
            [8, 19],
            [1, 10, 100, 101, 102, 11, 2, 20, 200, 201, 21, 3, 10, 100, 101, 102, 11, 4, 20, 300, 301, 302, 21, 5],
        ),
    ],
)
def test_positions_refer_to_final_input_after_both_modalities_expand(
    tokenizer_class, origin_ids, image_starts, audio_starts, expected_ids
):
    tokenizer = make_tokenizer(tokenizer_class, origin_ids)
    params = make_params(images=[(100, 3), (100, 3)], audios=[(200, 2), (300, 3)])

    input_ids = tokenizer.encode("mixed image/audio input", params, add_special_tokens=False)

    assert input_ids == expected_ids
    assert [image.start_idx for image in params.images] == image_starts
    assert [audio.start_idx for audio in params.audios] == audio_starts
    spans = get_mm_input_spans(params.to_dict())
    assert len(spans) == 4
    for item in params.images + params.audios:
        assert (item.start_idx, item.start_idx + item.token_num) in spans
        assert input_ids[item.start_idx : item.start_idx + item.token_num] == list(
            range(item.token_id, item.token_id + item.token_num)
        )
