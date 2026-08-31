from __future__ import annotations

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
from lightllm.utils.torch_dtype_utils import get_torch_dtype


class Glm5NextVisionModel:
    """LightLLM visual-server adapter around the official GLM-5 vision tower."""

    def __init__(self, data_type="bfloat16"):
        self.data_type = data_type if isinstance(data_type, torch.dtype) else get_torch_dtype(data_type)
        self.device = torch.device("cpu")
        self.vision_tower = None
        self.image_processor = None

    @staticmethod
    def _weight_files(weight_dir):
        index_path = os.path.join(weight_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as stream:
                weight_map = json.load(stream)["weight_map"]
            return sorted(set(weight_map.values()))
        return sorted(name for name in os.listdir(weight_dir) if name.endswith(".safetensors"))

    @classmethod
    def _load_prefix_state_dict(cls, weight_dir, prefix):
        state_dict = {}
        for file_name in cls._weight_files(weight_dir):
            with safe_open(os.path.join(weight_dir, file_name), framework="pt", device="cpu") as stream:
                for key in stream.keys():
                    if key.startswith(prefix):
                        state_dict[key[len(prefix) :]] = stream.get_tensor(key)
        return state_dict

    def load_model(self, weight_dir):
        try:
            from transformers.models.glm5_next.modeling_glm5_next import (
                Glm5NextVisionModel as HFGlm5NextVisionModel,
            )
        except ImportError as exc:
            raise ImportError("GLM-5 vision requires a Transformers build with glm5_next support") from exc

        config = AutoConfig.from_pretrained(weight_dir, trust_remote_code=True)
        if config.vision_config is None:
            raise ValueError("GLM-5 checkpoint does not contain vision_config")
        # Direct submodel construction bypasses AutoModel's normal attention
        # implementation selection.  SDPA keeps large-image attention
        # memory-efficient and is available in the pinned PyTorch runtime.
        config.vision_config._attn_implementation = "sdpa"
        self.vision_tower = HFGlm5NextVisionModel(config.vision_config).eval()
        self.image_processor = AutoProcessor.from_pretrained(weight_dir).image_processor

        state_dict = self._load_prefix_state_dict(weight_dir, "model.visual.")
        missing, unexpected = self.vision_tower.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"GLM-5 vision weight mismatch: missing={missing}, unexpected={unexpected}")
        return self

    def cuda(self):
        self.device = torch.device("cuda")
        self.vision_tower = self.vision_tower.to(device=self.device, dtype=self.data_type)
        return self

    @torch.inference_mode()
    def forward(self, pixel_values, image_grid_thw):
        output = self.vision_tower(
            hidden_states=pixel_values.to(self.device, dtype=self.data_type, non_blocking=True),
            grid_thw=image_grid_thw.to(self.device, non_blocking=True),
        )
        return output.pooler_output.to(self.data_type)

    @torch.inference_mode()
    def encode(self, images: List[ImageItem]):
        pil_images = []
        uuids = []
        for image_item in images:
            if not isinstance(image_item, ImageItem):
                raise TypeError(f"Unsupported GLM-5 image input type: {type(image_item)}")
            uuids.append(image_item.uuid)
            image_data = read_shm(get_shm_name_data(image_item.uuid))
            with Image.open(BytesIO(image_data)) as image:
                pil_images.append(image.convert("RGB"))

        if not pil_images:
            return None

        image_inputs = self.image_processor(pil_images, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"]
        image_grid_thw = image_inputs["image_grid_thw"]

        merge_area = self.image_processor.merge_size ** 2
        token_nums = [int(grid.prod().item() // merge_area) for grid in image_grid_thw]
        valid_ids = []
        valid_start = 0
        for image_item, token_num in zip(images, token_nums):
            if image_item.token_num is not None and image_item.token_num != token_num:
                raise ValueError(f"GLM-5 image token mismatch: allocated={image_item.token_num}, encoded={token_num}")
            valid_ids.append([valid_start, valid_start + token_num])
            valid_start += token_num

        image_embeds = self.forward(pixel_values, image_grid_thw)
        if image_embeds.shape[0] != valid_start:
            raise ValueError(f"GLM-5 image embed length mismatch: embeds={image_embeds.shape[0]}, tokens={valid_start}")
        return image_embeds, uuids, valid_ids
