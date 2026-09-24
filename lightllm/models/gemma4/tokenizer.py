import math

from lightllm.common.basemodel.multimodal_tokenizer import BaseMultiModalTokenizer
from lightllm.server.core.objs.sampling_params import SamplingParams
from lightllm.server.multimodal_params import AudioItem, ImageItem, MultimodalParams


class Gemma4Tokenizer(BaseMultiModalTokenizer):
    def __init__(self, tokenizer, model_cfg, image_processor=None):
        super().__init__(tokenizer)
        self.image_token_index = model_cfg.get("image_token_id", 258880)
        self.boi_token_index = model_cfg.get("boi_token_id", 255999)
        self.eoi_token_index = model_cfg.get("eoi_token_id", 258882)
        self.image_processor = image_processor
        self.image_length = model_cfg.get("vision_soft_tokens_per_image", 280)
        self.patch_size = getattr(self.image_processor, "patch_size", 16)
        self.pooling_kernel_size = getattr(self.image_processor, "pooling_kernel_size", 3)
        self.max_soft_tokens = getattr(self.image_processor, "max_soft_tokens", self.image_length)
        # HF Gemma-4 tokenizer does not prepend BOS even with add_special_tokens=True.
        self.bos_token_id = tokenizer.bos_token_id

    def init_imageitem_extral_params(
        self, img: ImageItem, multi_params: MultimodalParams, sampling_params: SamplingParams
    ):
        return

    def init_audioitem_extral_params(
        self, audio: AudioItem, multi_params: MultimodalParams, sampling_params: SamplingParams
    ):
        raise NotImplementedError

    def get_image_token_length(self, img: ImageItem):
        if self.image_processor is None or img.image_w <= 0 or img.image_h <= 0:
            return self.image_length

        patch, kernel = self.patch_size, self.pooling_kernel_size
        unit = patch * kernel
        num_patches_orig = (img.image_h / patch) * (img.image_w / patch)
        scale = math.sqrt(self.max_soft_tokens * kernel ** 2 / num_patches_orig)
        target_h = max(unit, int(math.floor(img.image_h * scale / unit)) * unit)
        target_w = max(unit, int(math.floor(img.image_w * scale / unit)) * unit)
        num_patches = (target_h // patch) * (target_w // patch)
        return min(num_patches // kernel ** 2, self.max_soft_tokens)

    def get_audio_token_length(self, audio: AudioItem):
        raise NotImplementedError

    def encode(self, prompt, multimodal_params: MultimodalParams = None, add_special_tokens=False):
        origin_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        if (
            add_special_tokens
            and self.bos_token_id is not None
            and (len(origin_ids) == 0 or origin_ids[0] != self.bos_token_id)
        ):
            origin_ids = [self.bos_token_id] + origin_ids

        images = [] if multimodal_params is None else getattr(multimodal_params, "images", [])
        if not images:
            return origin_ids

        input_ids = []
        image_id = 0
        start = 0
        while True:
            try:
                image_start = origin_ids.index(self.image_token_index, start)
            except ValueError:
                break

            input_ids.extend(origin_ids[start:image_start])
            image_end = image_start + 1
            while image_end < len(origin_ids) and origin_ids[image_end] == self.image_token_index:
                image_end += 1
            if image_id >= len(images):
                raise ValueError("image token error")

            img = images[image_id]
            if not input_ids or input_ids[-1] != self.boi_token_index:
                input_ids.append(self.boi_token_index)
            img.start_idx = len(input_ids)
            input_ids.extend(range(img.token_id, img.token_id + img.token_num))
            input_ids.append(self.eoi_token_index)

            if image_end < len(origin_ids) and origin_ids[image_end] == self.eoi_token_index:
                image_end += 1
            start = image_end
            image_id += 1

        input_ids.extend(origin_ids[start:])
        image_cnt = len(images)
        if image_cnt != image_id:
            raise ValueError(f"invalid image tag num: {image_cnt} vs {image_id}!")
        return input_ids
