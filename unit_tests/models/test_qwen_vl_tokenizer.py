from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lightllm.models.qwen2_vl.model import QWen2VLTokenizer
from lightllm.models.qwen3_vl.model import QWen3VLTokenizer
from lightllm.utils.error_utils import InvalidRequestError


@pytest.fixture(params=[QWen2VLTokenizer, QWen3VLTokenizer])
def tokenizer(request):
    instance = request.param.__new__(request.param)
    instance.tokenizer = Mock()
    instance.image_start_id = 10
    instance.image_end_id = 11
    instance.image_token_id = 12
    return instance


def test_image_expansion(tokenizer):
    images = [SimpleNamespace(token_id=100, token_num=2), SimpleNamespace(token_id=200, token_num=3)]
    prompt = [1, 10, 12, 11, 2, 10, 11, 3]
    assert tokenizer.encode(prompt, SimpleNamespace(images=images)) == [
        1,
        10,
        100,
        101,
        11,
        2,
        10,
        200,
        201,
        202,
        11,
        3,
    ]
    assert [image.start_idx for image in images] == [2, 7]
    assert prompt == [1, 10, 12, 11, 2, 10, 11, 3]
    assert tokenizer.encode(prompt, None) == [1, 10, 11, 2, 10, 11, 3]


@pytest.mark.parametrize("prompt,image_count", [([10, 12, 11], 0), ([10, 11, 10, 11], 1), ([1], 1)])
def test_image_count_mismatch(tokenizer, prompt, image_count):
    images = [SimpleNamespace(token_id=100, token_num=2) for _ in range(image_count)]
    with pytest.raises(InvalidRequestError, match="invalid image tag num"):
        tokenizer.encode(prompt, SimpleNamespace(images=images))
    assert all(not hasattr(image, "start_idx") for image in images)


@pytest.mark.parametrize("prompt", [[10], [10, 12], [10, 1, 11], [10, 10, 11]])
@pytest.mark.parametrize("params", [None, SimpleNamespace(images=[])])
def test_invalid_image_sequence(tokenizer, prompt, params):
    with pytest.raises(InvalidRequestError, match="image"):
        tokenizer.encode(prompt, params)


def test_text_and_encode_options(tokenizer):
    tokenizer.tokenizer.encode.return_value = [1, 12, 2]
    assert tokenizer.encode("text", SimpleNamespace(images=[]), add_special_tokens=False) == [1, 12, 2]
    tokenizer.tokenizer.encode.assert_called_once_with("text", add_special_tokens=False)


def test_unpaired_image_end_tokens(tokenizer):
    assert tokenizer.encode([1, 11, 2], SimpleNamespace(images=[])) == [1, 11, 2]
    images = [SimpleNamespace(token_id=100, token_num=2), SimpleNamespace(token_id=200, token_num=1)]
    prompt = [11, 1, 10, 12, 11, 11, 2, 10, 11, 11]
    assert tokenizer.encode(prompt, SimpleNamespace(images=images)) == [11, 1, 10, 100, 101, 11, 11, 2, 10, 200, 11, 11]
    assert tokenizer.encode(prompt, None) == [11, 1, 10, 11, 11, 2, 10, 11, 11]
