from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lightllm.common.basemodel.multimodal_tokenizer import BaseMultiModalTokenizer
from lightllm.models.neo_chat.vision_process import smart_resize
from lightllm.server.core.objs import SamplingParams
from lightllm.server.multimodal_params import AudioItem, ImageItem, MultimodalParams


IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_TOKEN = "<image>"


@dataclass(frozen=True)
class GenerationPrefixInput:
    role: str
    prompt: str
    images: tuple[dict[str, Any], ...]


class NeoChatTokenizer(BaseMultiModalTokenizer):
    """Tokenizer adapter for Neo's interleaved understanding prefix."""

    # NeoPP consumes immutable cond/uncond Prefix KV during CFG. Other image
    # models may expose a different role set through the same Omni interface.
    generation_prefix_roles = ("cond", "uncond")

    def __init__(self, tokenizer, model_cfg, **_kwargs):
        super().__init__(tokenizer)
        vision_config = model_cfg["vision_config"]
        self.min_pixels = int(vision_config.get("min_pixels", 65_536))
        self.max_pixels = int(vision_config.get("max_pixels", 4_194_304))
        self.patch_size = int(vision_config.get("patch_size", 16))
        self.downsample_ratio = float(vision_config.get("downsample_ratio", 0.5))

        self.image_token_id = model_cfg.get("image_token_id")
        self.image_start_tag = IMG_START_TOKEN
        self.image_start_id = tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
        self.image_end_tag = IMG_END_TOKEN
        self.image_end_id = tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        self.image_tag = IMG_TOKEN

    def init_imageitem_extral_params(
        self,
        img: ImageItem,
        multi_params: MultimodalParams,
        sampling_params: SamplingParams,
    ):
        # Resize policy is owned by Neo's visual adapter.  SamplingParams does
        # not carry model-specific pixel fields.
        img.extra_params["min_pixels"] = self.min_pixels
        img.extra_params["max_pixels"] = self.max_pixels
        if self.min_pixels > self.max_pixels:
            raise ValueError("Neo min_pixels must not exceed max_pixels")

    def init_audioitem_extral_params(
        self,
        audio: AudioItem,
        multi_params: MultimodalParams,
        sampling_params: SamplingParams,
    ):
        raise NotImplementedError("Neo does not have an audio encoder")

    def get_audio_token_length(self, audio: AudioItem):
        raise NotImplementedError("Neo does not have an audio encoder")

    def get_image_token_length(self, img: ImageItem):
        factor = int(self.patch_size / self.downsample_ratio)
        resized_height, resized_width = smart_resize(
            height=img.image_h,
            width=img.image_w,
            factor=factor,
            min_pixels=img.extra_params.get("min_pixels", self.min_pixels),
            max_pixels=img.extra_params.get("max_pixels", self.max_pixels),
        )
        grid_h = resized_height // self.patch_size
        grid_w = resized_width // self.patch_size
        token_h = int(grid_h * self.downsample_ratio)
        token_w = int(grid_w * self.downsample_ratio)
        token_num = token_h * token_w
        # t/h/w position construction consumes the post-downsample grid.  The
        # final value shifts following text as if the whole image occupied one
        # temporal position.
        img.grid_thwd = (1, token_h, token_w, 1 - token_num)
        return token_num

    def encode(self, prompt, multimodal_params: MultimodalParams = None, **kwargs):
        if multimodal_params is None:
            return self.tokenizer.encode(
                prompt,
                add_special_tokens=kwargs.get("add_special_tokens", True),
            )

        if kwargs.get("already_tokenized", False):
            origin_ids = list(prompt)
        else:
            # Replace every placeholder.  The validation loop below then
            # reports a precise error when the prompt/image counts differ.
            prompt = prompt.replace(IMG_TOKEN, IMG_START_TOKEN + IMG_END_TOKEN)
            origin_ids = self.tokenizer.encode(
                prompt,
                add_special_tokens=kwargs.get("add_special_tokens", True),
            )

        input_ids = []
        image_index = 0
        while self.image_start_id in origin_ids:
            start = origin_ids.index(self.image_start_id)
            if start + 1 >= len(origin_ids):
                # Neo generation prefixes deliberately end at a lone <img>.
                # It marks where the image-token diffusion branch starts and
                # is ordinary text KV, not an input-image placeholder.
                break
            if origin_ids[start + 1] != self.image_end_id:
                raise ValueError(
                    "Neo image start token must be followed by an image end token"
                )
            if image_index >= len(multimodal_params.images):
                raise ValueError(
                    "Neo prompt contains more image tags than supplied images"
                )
            input_ids.extend(origin_ids[: start + 1])
            image = multimodal_params.images[image_index]
            if image.token_id is None or image.token_num is None:
                raise ValueError(
                    "Neo image cache resources must be allocated before tokenization"
                )
            image.start_idx = len(input_ids)
            input_ids.extend(range(image.token_id, image.token_id + image.token_num))
            input_ids.append(self.image_end_id)
            origin_ids = origin_ids[start + 2 :]
            image_index += 1

        if image_index != len(multimodal_params.images):
            raise ValueError(
                f"Neo prompt has {image_index} image tags for "
                f"{len(multimodal_params.images)} images"
            )
        input_ids.extend(origin_ids)
        return input_ids

    def _build_t2i_query(self, message: str, thinking_content: str = "") -> str:
        prompt = self.tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": message}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt + thinking_content + IMG_START_TOKEN

    def fix_prompt(self, prompt: str, image_count: int) -> str:
        prompt_image_count = prompt.count(IMG_TOKEN)
        if prompt_image_count > image_count:
            raise ValueError(
                f"Neo prompt needs {prompt_image_count} images, got {image_count}"
            )
        if prompt_image_count < image_count:
            prompt = f"{IMG_TOKEN}\n" * (image_count - prompt_image_count) + prompt
        return prompt

    def get_query_for_t2i(self, prompt: str, input_image_num: int = 0):
        image_count = prompt.count(IMG_TOKEN)
        condition = (
            prompt if prompt.endswith(IMG_START_TOKEN) else prompt + IMG_START_TOKEN
        )
        uncondition = self._build_t2i_query(
            IMG_TOKEN * input_image_num,
            thinking_content=IMG_TOKEN * (image_count - input_image_num),
        )
        return condition, uncondition

    def get_query_for_it2i(self, prompt: str):
        image_count = prompt.count(IMG_TOKEN)
        condition = (
            prompt if prompt.endswith(IMG_START_TOKEN) else prompt + IMG_START_TOKEN
        )
        text_uncondition = self._build_t2i_query(IMG_TOKEN * image_count)
        image_uncondition = self._build_t2i_query("")
        return condition, text_uncondition, image_uncondition

    def build_generation_prefixes(self, *, prompt, images, generation):
        """Build the model-specific prefixes consumed by the Omni image flow."""

        prompt = self.fix_prompt(prompt, len(images))
        applied_prompt = self.tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        condition, uncondition = self.get_query_for_t2i(
            applied_prompt,
            input_image_num=len(images),
        )
        by_role = {"cond": condition, "uncond": uncondition}
        return [
            GenerationPrefixInput(
                role,
                by_role[role],
                tuple(dict(image) for image in images),
            )
            for role in generation.prefix_roles
        ]

    def get_generation_position_offset(self, input_len: int, multimodal_params) -> int:
        """Map Neo's physical image tokens to its compressed RoPE positions."""

        position_offset = int(input_len)
        for image in multimodal_params.images:
            if image.grid_thwd is None or len(image.grid_thwd) < 4:
                raise ValueError(
                    "generation input image is missing its position metadata"
                )
            position_offset += int(image.grid_thwd[3])
        if position_offset <= 0:
            raise ValueError("generation prefix position offset must be positive")
        return position_offset
