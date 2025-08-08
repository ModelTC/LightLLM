from __future__ import annotations
import math
import torch
import numpy as np
from PIL import Image
from typing import List, Optional, Union

from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_transforms import (
    convert_to_rgb,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    to_numpy_array,
)

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768


def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:

    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def resize_image(image_file: str | Image.Image, size_factor: int = IMAGE_FACTOR) -> tuple[Image.Image, int, int]:

    image = image_file.convert("RGB")
    width, height = image.size

    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=size_factor,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    image = image.resize((resized_width, resized_height))

    return image


class Qwen2VLImageProcessor(BaseImageProcessor):
    def __init__(
        self,
        do_resize: bool = True,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        min_pixels: int = 56 * 56,
        max_pixels: int = 28 * 28 * 1280,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.do_convert_rgb = do_convert_rgb
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.data_format = ChannelDimension.FIRST

    def preprocess(self, image):
        if self.do_convert_rgb:
            image = convert_to_rgb(image)
        image = to_numpy_array(image)
        input_data_format = infer_channel_dimension_format(image)
        height, width = get_image_size(image, channel_dim=input_data_format)

        resized_height, resized_width = height, width
        if self.do_resize:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=self.patch_size * self.merge_size,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
            image = resize(
                image, size=(resized_height, resized_width), resample=self.resample, input_data_format=input_data_format
            )

        if self.do_rescale:
            image = self.rescale(image, scale=self.rescale_factor, input_data_format=input_data_format)

        if self.do_normalize:
            image = self.normalize(
                image=image, mean=self.image_mean, std=self.image_std, input_data_format=input_data_format
            )

        image = to_channel_dimension_format(image, self.data_format, input_channel_dim=input_data_format)

        patches = np.array([image])

        if patches.shape[0] == 1:
            patches = np.tile(patches, (self.temporal_patch_size, 1, 1, 1))
        channel = patches.shape[1]
        grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size
        patches = patches.reshape(
            1,
            self.temporal_patch_size,
            channel,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_h * grid_w, channel * self.temporal_patch_size * self.patch_size * self.patch_size
        )
        image_grid_thw = (1, grid_h, grid_w)
        pixel_values = torch.as_tensor(flatten_patches)
        grid_thw = torch.as_tensor([image_grid_thw])

        return pixel_values, grid_thw
