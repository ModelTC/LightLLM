import asyncio
import pickle
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from tokenizers import Tokenizer, models, processors

from lightllm.models.glm5_next.tokenizer import Glm5NextTokenizer
from lightllm.models.qwen3_5.model import QWen3_5Tokenizer
from lightllm.server.core.objs import SamplingParams
from lightllm.server.httpserver.manager import HttpServerManager
from lightllm.server.httpserver_for_pd_master.manager import HttpServerManagerForPDMaster
from lightllm.server.pd_io_struct import NodeRole


@pytest.mark.parametrize("multimodal", [False, True])
@pytest.mark.parametrize("add_special_tokens", [False, True])
def test_master_encodes_once_and_prefill_preserves_output_ids(multimodal, add_special_tokens):
    tokenizer = Tokenizer(models.BPE({"P": 0, "a": 1, "Pa": 2, "<bos>": 3}, [("P", "a")]))
    tokenizer.post_processor = processors.TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", 3)])
    master = object.__new__(HttpServerManagerForPDMaster)
    master.tokenizer = SimpleNamespace(
        encode=Mock(side_effect=lambda text, _, **kwargs: tokenizer.encode(text, **kwargs).ids)
    )
    params = SamplingParams()
    params.init(None, add_special_tokens=add_special_tokens)
    resources = SimpleNamespace(images=[], audios=[], verify_resource_limits=lambda: None)
    prompt_ids, token_count = master._encode_prompt(
        "P", resources, params, {"add_special_tokens": params.add_special_tokens}
    )
    assert prompt_ids == ([3] if add_special_tokens else []) + [0]
    assert token_count == len(prompt_ids)
    master.tokenizer.encode.assert_called_once_with("P", None, add_special_tokens=add_special_tokens)

    manager = object.__new__(HttpServerManager)
    manager.enable_multimodal = multimodal
    manager.pd_mode = NodeRole.P
    manager.vocab_size = 4
    manager._alloc_multimodal_resources = AsyncMock()
    # P expands media placeholders in the supplied IDs without text tokenization.
    manager.tokenizer = SimpleNamespace(encode=Mock(side_effect=lambda ids, *_args, **_kwargs: [2 ** 32] + ids))
    params.pd_previous_output_len = 1
    continuation_ids = pickle.loads(pickle.dumps(prompt_ids + [1]))
    result = asyncio.run(manager._encode(continuation_ids, resources, params))

    assert tokenizer.encode("Pa", add_special_tokens=False).ids == [2]
    assert result == ([2 ** 32] if multimodal else []) + prompt_ids + [1]
    if multimodal:
        manager.tokenizer.encode.assert_called_once_with(
            continuation_ids, resources, add_special_tokens=add_special_tokens
        )
        manager._alloc_multimodal_resources.assert_awaited_once_with(resources, params)
    else:
        manager.tokenizer.encode.assert_not_called()
        manager._alloc_multimodal_resources.assert_not_awaited()


@pytest.mark.parametrize("tokenizer_class", [QWen3_5Tokenizer, Glm5NextTokenizer], ids=["qwen3.5", "glm5.3-flash"])
@pytest.mark.parametrize("image_count", [0, 1, 2])
def test_multimodal_continuation_expands_images_without_retokenizing(tokenizer_class, image_count):
    tokenizer = object.__new__(tokenizer_class)
    tokenizer.tokenizer = SimpleNamespace(encode=Mock(return_value=[10, 12, 11] * image_count + [1]))
    tokenizer.image_start_id, tokenizer.image_end_id, tokenizer.image_token_id = 10, 11, 12
    tokenizer.get_image_token_length = lambda image: image.token_num
    images = [SimpleNamespace(token_id=1000, token_num=2) for _ in range(image_count)]
    resources = SimpleNamespace(images=images, audios=[], verify_resource_limits=lambda: None)
    params = SamplingParams()
    params.init(None)
    master = object.__new__(HttpServerManagerForPDMaster)
    master.tokenizer = tokenizer
    master.args = SimpleNamespace(max_image_token_count=16)
    prompt_ids, token_count = master._encode_prompt("prompt", resources, params)
    assert prompt_ids == [10, 11] * image_count + [1]

    manager = object.__new__(HttpServerManager)
    manager.enable_multimodal = True
    manager.pd_mode = NodeRole.P
    manager.tokenizer = tokenizer
    manager.args = master.args
    manager._alloc_multimodal_resources = AsyncMock()
    history = [2, 3]
    params.pd_previous_output_len = len(history)
    encoded = asyncio.run(manager._encode(prompt_ids + history, resources, params))
    assert encoded == [10, 1000, 1001, 11] * image_count + [1] + history
    assert token_count + len(history) == len(encoded)
    assert [image.start_idx for image in images] == [4 * index + 1 for index in range(image_count)]

    # D receives P's expanded IDs and must preserve them, including the history suffix.
    manager.pd_mode = NodeRole.D
    assert asyncio.run(manager._encode(encoded, resources, params)) == encoded
    tokenizer.tokenizer.encode.assert_called_once_with("prompt")
    manager._alloc_multimodal_resources.assert_awaited_once_with(resources, params)
    assert master.tokens("prompt", resources, params) == token_count
    assert manager.tokens("prompt", resources, params) == token_count
