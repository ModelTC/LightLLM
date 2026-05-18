import json
import os
from io import BytesIO
from typing import List

import torch
from PIL import Image
from safetensors import safe_open
from transformers import AutoConfig, AutoProcessor

from lightllm.server.embed_cache.utils import get_shm_name_data, read_shm
from lightllm.server.multimodal_params import ImageItem
from lightllm.utils.log_utils import init_logger
from lightllm.utils.torch_dtype_utils import get_torch_dtype


logger = init_logger(__name__)


class Gemma4VisionModel:
    def __init__(self, data_type="bfloat16"):
        self.vision_tower = None
        self.embed_vision = None
        self.image_processor = None
        self.data_type = data_type if isinstance(data_type, torch.dtype) else get_torch_dtype(data_type)
        self.device = torch.device("cpu")

    def _weight_files(self, weight_dir):
        index_path = os.path.join(weight_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                weight_map = json.load(f)["weight_map"]
            return sorted(set(weight_map.values()))
        return sorted(f for f in os.listdir(weight_dir) if f.endswith(".safetensors"))

    def _load_prefix_state_dict(self, weight_dir, prefix):
        state_dict = {}
        for file_name in self._weight_files(weight_dir):
            file_path = os.path.join(weight_dir, file_name)
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith(prefix):
                        state_dict[key[len(prefix) :]] = f.get_tensor(key)
        return state_dict

    def load_model(self, weight_dir):
        try:
            from transformers.models.gemma4.modeling_gemma4 import (
                Gemma4MultimodalEmbedder,
                Gemma4VisionModel as HFGemma4VisionModel,
            )
        except ImportError as e:
            raise ImportError("Gemma-4 vision requires a transformers build with Gemma4 support.") from e

        config = AutoConfig.from_pretrained(weight_dir, trust_remote_code=True)
        if config.vision_config is None:
            raise ValueError("Gemma-4 checkpoint does not contain vision_config")

        processor = AutoProcessor.from_pretrained(weight_dir)
        self.image_processor = processor.image_processor
        self.vision_tower = HFGemma4VisionModel(config.vision_config).eval()
        self.embed_vision = Gemma4MultimodalEmbedder(config.vision_config, config.text_config).eval()

        vision_state = self._load_prefix_state_dict(weight_dir, "model.vision_tower.")
        embed_state = self._load_prefix_state_dict(weight_dir, "model.embed_vision.")
        missing, unexpected = self.vision_tower.load_state_dict(vision_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Gemma-4 vision_tower weight mismatch: missing={missing}, unexpected={unexpected}")
        missing, unexpected = self.embed_vision.load_state_dict(embed_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Gemma-4 embed_vision weight mismatch: missing={missing}, unexpected={unexpected}")

        return self

    def cuda(self):
        self.device = torch.device("cuda")
        self.vision_tower = self.vision_tower.cuda()
        self.embed_vision = self.embed_vision.cuda()
        return self

    def forward(self, pixel_values, image_position_ids):
        pixel_values = pixel_values.to(self.device, non_blocking=True)
        image_position_ids = image_position_ids.to(self.device, non_blocking=True)
        pooling_k = self.vision_tower.config.pooling_kernel_size
        pooling_k2 = pooling_k * pooling_k

        # Per-image vision-tower call. `output_length` MUST match the per-image
        # num_soft_tokens the image processor declared; otherwise HF's pooler
        # falls back to config.image_seq_length and silently emits a different
        # token count than what `valid_ids` expects.
        per_image_hidden = []
        for i in range(pixel_values.shape[0]):
            pv = pixel_values[i : i + 1]
            pp = image_position_ids[i : i + 1]
            output_length = pv.shape[1] // pooling_k2
            per_image_hidden.append(
                self.vision_tower(
                    pixel_values=pv,
                    pixel_position_ids=pp,
                    output_length=output_length,
                ).last_hidden_state
            )

        # embed_vision is token-independent (RMSNorm + Linear); cat once and
        # project once instead of looping like vllm — same numerics, fewer
        # Python launches, lines up naturally with our flat embed-cache output.
        flat_hidden = torch.cat(per_image_hidden, dim=0)
        target_dtype = self.embed_vision.embedding_projection.weight.dtype
        image_features = self.embed_vision(inputs_embeds=flat_hidden.unsqueeze(0).to(target_dtype)).squeeze(0)
        return image_features.to(self.data_type)

    @torch.inference_mode()
    def encode(self, images: List[ImageItem]):
        pil_images = []
        uuids = []
        for img in images:
            if not isinstance(img, ImageItem):
                raise TypeError(f"Unsupported Gemma-4 image input type: {type(img)}")
            uuids.append(img.uuid)
            image_data = read_shm(get_shm_name_data(img.uuid))
            with Image.open(BytesIO(image_data)) as image:
                pil_images.append(image.convert("RGB"))

        if not pil_images:
            return None

        image_inputs = self.image_processor(pil_images, return_tensors="pt")
        token_nums = image_inputs.pop("num_soft_tokens_per_image")
        pixel_values = image_inputs["pixel_values"]
        image_position_ids = image_inputs["image_position_ids"]

        valid_ids = []
        valid_start = 0
        for img, token_num in zip(images, token_nums):
            token_num = int(token_num)
            if img.token_num != token_num:
                raise ValueError(f"Gemma-4 image token mismatch: allocated={img.token_num}, encoded={token_num}")
            valid_ids.append([valid_start, valid_start + token_num])
            valid_start += token_num

        all_img_embeds = self.forward(pixel_values, image_position_ids)
        if all_img_embeds.shape[0] != valid_start:
            raise ValueError(
                f"Gemma-4 image embed length mismatch: embeds={all_img_embeds.shape[0]}, tokens={valid_start}"
            )
        return all_img_embeds, uuids, valid_ids
