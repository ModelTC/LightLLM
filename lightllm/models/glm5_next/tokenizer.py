from __future__ import annotations

import copy
from typing import List, Union

from lightllm.common.basemodel.multimodal_tokenizer import BaseMultiModalTokenizer
from lightllm.models.qwen2_vl.model import QWen2VLTokenizer
from lightllm.server.multimodal_params import ImageItem, MultimodalParams


class Glm5NextTokenizer(QWen2VLTokenizer):
    """Multimodal tokenizer adapter for GLM-5 Next checkpoints.

    The released GLM-5.3-Flash chat template deliberately renders OpenAI
    image parts as a text-only reminder.  LightLLM carries image bytes through
    ``MultimodalParams``, so template-facing image parts must instead become
    one GLM image placeholder.  ``encode`` (inherited from QWen2VLTokenizer)
    replaces that placeholder with the virtual token range allocated by the
    embedding cache.
    """

    image_placeholder = "<|begin_of_image|><|image|><|end_of_image|>"

    def __init__(self, tokenizer=None, image_processor=None, **kwargs):
        BaseMultiModalTokenizer.__init__(self, tokenizer)
        self.image_processor = image_processor
        model_cfg = kwargs["model_cfg"]
        self.image_start_id = model_cfg["image_start_token_id"]
        self.image_end_id = model_cfg["image_end_token_id"]
        self.image_token_id = model_cfg["image_token_id"]
        self.patch_size = image_processor.patch_size
        self.merge_size = image_processor.merge_size
        self.min_image_tokens = image_processor.min_image_tokens
        self.max_image_tokens = image_processor.max_image_tokens

    def get_image_token_length(self, img: ImageItem):
        if img.image_w <= 0 or img.image_h <= 0:
            raise ValueError(f"invalid GLM-5 image size: {img.image_w}x{img.image_h}")
        patch_num = self.image_processor.get_number_of_image_patches(img.image_h, img.image_w)
        token_num = patch_num // (self.merge_size ** 2)
        if token_num <= 0:
            raise ValueError(f"GLM-5 image produced no visual tokens: {img.image_w}x{img.image_h}")
        return token_num

    def apply_chat_template(self, conversation=None, messages=None, **kwargs):
        source = conversation if conversation is not None else messages
        if source is None:
            return self.tokenizer.apply_chat_template(conversation=conversation, messages=messages, **kwargs)

        normalized = copy.deepcopy(source)
        for message in normalized:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            rendered_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") in ("image", "image_url"):
                    rendered_parts.append({"type": "text", "text": self.image_placeholder})
                else:
                    rendered_parts.append(part)
            message["content"] = rendered_parts

        if conversation is not None:
            return self.tokenizer.apply_chat_template(conversation=normalized, **kwargs)
        return self.tokenizer.apply_chat_template(messages=normalized, **kwargs)

    def encode(
        self,
        prompt: Union[str, List[int]],
        multimodal_params: MultimodalParams = None,
        **kwargs,
    ):
        return super().encode(prompt, multimodal_params=multimodal_params, **kwargs)
