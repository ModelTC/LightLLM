from transformers.configuration_utils import PretrainedConfig

from lightllm.models.glm5_next.model import (
    Glm5NextMultimodalTpPartModel,
    Glm5NextTpPartModel,
)
from lightllm.models.glm5_next.tokenizer import Glm5NextTokenizer
from lightllm.models.registry import get_model_class
from lightllm.server.multimodal_params import MultimodalParams
from lightllm.utils.config_utils import has_vision_module


class _FakeImageProcessor:
    patch_size = 14
    merge_size = 2
    min_image_tokens = 16
    max_image_tokens = 8000

    @staticmethod
    def get_number_of_image_patches(height, width):
        assert (height, width) == (448, 448)
        return 1024


class _FakeTokenizer:
    def __init__(self):
        self.last_conversation = None

    def apply_chat_template(self, conversation, **kwargs):
        self.last_conversation = conversation
        return conversation[0]["content"][0]["text"]

    @staticmethod
    def encode(prompt):
        assert prompt == Glm5NextTokenizer.image_placeholder
        return [7, 10, 12, 11, 8]


def _make_tokenizer():
    model_cfg = {
        "image_start_token_id": 10,
        "image_end_token_id": 11,
        "image_token_id": 12,
    }
    return Glm5NextTokenizer(
        tokenizer=_FakeTokenizer(),
        image_processor=_FakeImageProcessor(),
        model_cfg=model_cfg,
    )


def test_glm5_image_prompt_and_virtual_tokens():
    tokenizer = _make_tokenizer()
    conversation = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}],
        }
    ]

    prompt = tokenizer.apply_chat_template(conversation=conversation)
    assert prompt == tokenizer.image_placeholder
    assert conversation[0]["content"][0]["type"] == "image_url"

    multimodal_params = MultimodalParams(images=[{"type": "base64", "data": ""}])
    image = multimodal_params.images[0]
    image.image_w = image.image_h = 448
    image.token_num = tokenizer.get_image_token_length(image)
    image.token_id = 1000

    input_ids = tokenizer.encode(prompt, multimodal_params=multimodal_params)
    assert image.token_num == 256
    assert image.start_idx == 2
    assert input_ids == [7, 10, *range(1000, 1256), 11, 8]


def test_glm5_vision_config_selects_multimodal_model(monkeypatch):
    assert get_model_class({"model_type": "glm5_next"}) is Glm5NextTpPartModel
    assert (
        get_model_class({"model_type": "glm5_next", "vision_config": {"hidden_size": 1024}})
        is Glm5NextMultimodalTpPartModel
    )

    monkeypatch.setattr(
        PretrainedConfig,
        "get_config_dict",
        staticmethod(
            lambda _: (
                {"model_type": "glm5_next", "vision_config": {"hidden_size": 1024}},
                {},
            )
        ),
    )
    has_vision_module.cache_clear()
    assert has_vision_module("unused") is True
    has_vision_module.cache_clear()
