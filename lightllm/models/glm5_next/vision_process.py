# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 the HuggingFace Team. All rights reserved.
# Adapted from Hugging Face Transformers' GLM-5-Next image processor.

import json
import math
import os

import numpy as np
import torch
from torchvision.transforms.v2 import functional as F

from lightllm.models.qwen2_vl.vision_process import Qwen2VLImageProcessor


def smart_resize(height, width, factor=28, min_image_tokens=16, max_image_tokens=8000):
    """Choose an aligned canvas; padding, rather than stretching, preserves the image."""
    min_pixels, max_pixels = min_image_tokens * factor ** 2, max_image_tokens * factor ** 2

    def align(value):
        return math.ceil(value / factor) * factor

    target_h, target_w = align(height), align(width)
    if target_h * target_w < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        target_h, target_w = align(max(1, math.ceil(height * scale))), align(max(1, math.ceil(width * scale)))
    if target_h * target_w > max_pixels:
        if max_pixels < factor ** 2:
            raise ValueError("max_image_tokens must allow at least one aligned patch")
        low, high = 1, height
        target_h = target_w = factor
        while low <= high:
            content_h = (low + high) // 2
            content_w = max(1, math.floor(width * content_h / height))
            candidate_h, candidate_w = align(content_h), align(content_w)
            if candidate_h * candidate_w <= max_pixels:
                target_h, target_w = candidate_h, candidate_w
                low = content_h + 1
            else:
                high = content_h - 1
    return target_h, target_w


class Glm5NextImageProcessor(Qwen2VLImageProcessor):
    def __init__(self, min_image_tokens=16, max_image_tokens=8000, patch_expand_factor=1, **kwargs):
        super().__init__(**kwargs)
        self.min_image_tokens = min_image_tokens
        self.max_image_tokens = max_image_tokens
        self.patch_expand_factor = patch_expand_factor

    @classmethod
    def from_pretrained(cls, weight_dir):
        with open(os.path.join(weight_dir, "processor_config.json")) as f:
            return cls(**json.load(f)["image_processor"])

    def get_image_size(self, height, width):
        return smart_resize(
            height,
            width,
            factor=self.patch_size * self.merge_size * self.patch_expand_factor,
            min_image_tokens=self.min_image_tokens,
            max_image_tokens=self.max_image_tokens,
        )

    def _preprocess_bydevice(self, image, device="cuda"):
        pixels = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1).contiguous().to(device)
        height, width = pixels.shape[-2:]
        target_h, target_w = self.get_image_size(height, width)
        factor = self.patch_size * self.merge_size * self.patch_expand_factor
        scale = min(target_h / height, target_w / width)
        if height * width >= factor ** 2 * self.min_image_tokens:
            scale = min(1.0, scale)
        content_h = max(1, min(target_h, math.floor(height * scale)))
        content_w = max(1, min(target_w, math.floor(width * scale)))
        if (content_h, content_w) != (height, width):
            pixels = F.resize(pixels, [content_h, content_w], interpolation=self.interpolation, antialias=True)
        pixels = F.pad(pixels, [0, 0, target_w - content_w, target_h - content_h], fill=0)
        pixels = self.rescale_and_normalize(
            pixels, self.do_rescale, self.rescale_factor, self.do_normalize, self.image_mean, self.image_std
        )
        patch, merge, temporal = self.patch_size, self.merge_size, self.temporal_patch_size
        grid_h, grid_w = target_h // patch, target_w // patch
        pixels = pixels.reshape(3, grid_h // merge, merge, patch, grid_w // merge, merge, patch)
        pixels = pixels.permute(1, 4, 2, 5, 0, 3, 6)
        pixels = pixels.unsqueeze(5).expand(-1, -1, -1, -1, -1, temporal, -1, -1)
        return pixels.reshape(grid_h * grid_w, 3 * temporal * patch ** 2), torch.tensor([[1, grid_h, grid_w]])
