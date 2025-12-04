from __future__ import annotations
import math
import torch
import numpy as np
from PIL import Image
from collections import defaultdict
from typing import List, Optional, Union, Tuple

from transformers.image_processing_utils_fast import (
    BaseImageProcessorFast,
    group_images_by_shape,
    reorder_images,
)
from torchvision.transforms import InterpolationMode

from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    PILImageResampling,
    SizeDict,
)
from torchvision.transforms.v2 import functional as F

from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


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
            f"absolute aspect ratio must be smaller than MAX_RATIO, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def resize_image(
    image_file: Image.Image, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[Image.Image]:

    image = image_file.convert("RGB")
    width, height = image.size

    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    image = image.resize((resized_width, resized_height))

    return image


class Qwen2VLImageProcessor(BaseImageProcessorFast):
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
        disable_grouping: Optional[bool] = None,
        interpolation: Optional["F.InterpolationMode"] = InterpolationMode.BICUBIC,
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
        self.disable_grouping = disable_grouping
        self.interpolation = interpolation
        self.data_format = ChannelDimension.FIRST
        self._fused_cache = {}  # key: (do_norm, do_rescale, rescale_factor, device)

    def _get_fused_mean_std(
        self,
        do_normalize: bool,
        image_mean: Union[float, list[float]],
        image_std: Union[float, list[float]],
        do_rescale: bool,
        rescale_factor: float,
        device: Optional["torch.device"],
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        key = (bool(do_normalize), bool(do_rescale), float(rescale_factor), str(device))
        if key not in self._fused_cache:
            if do_rescale and do_normalize:
                mean = torch.tensor(image_mean) * (1.0 / rescale_factor)
                std = torch.tensor(image_std) * (1.0 / rescale_factor)
                do_rescale = False
            else:
                mean = torch.tensor(image_mean)
                std = torch.tensor(image_std)
            self._fused_cache[key] = (mean.to(device=device), std.to(device=device), do_rescale)
        return self._fused_cache[key]

    def rescale_and_normalize(
        self,
        images: "torch.Tensor",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Union[float, list[float]],
        image_std: Union[float, list[float]],
    ) -> "torch.Tensor":
        """
        Rescale and normalize images.
        """
        image_mean, image_std, do_rescale = self._get_fused_mean_std(
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            device=images.device,
        )
        # if/elif as we use fused rescale and normalize if both are set to True
        if do_normalize:
            images = self.normalize(images.to(dtype=torch.float32), image_mean, image_std)
        elif do_rescale:
            images = self.rescale(images, rescale_factor)

        return images

    @torch.inference_mode()
    def preprocess(self, image) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            return self._preprocess_bydevice(image, device="cuda")
        except Exception as e:
            logger.warning(f"Exception during image preprocessing on CUDA: {str(e)}")
            torch.cuda.current_stream().synchronize()
            return self._preprocess_bydevice(image, device="cpu")

    def _preprocess_bydevice(self, image, device="cuda") -> Tuple[torch.Tensor, torch.Tensor]:
        image_arr = np.asarray(image, dtype=np.uint8)
        image_data = torch.from_numpy(image_arr).permute(2, 0, 1).contiguous().to(device=device, non_blocking=True)

        grouped_images, grouped_images_index = group_images_by_shape(
            [image_data], disable_grouping=self.disable_grouping
        )
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            height, width = stacked_images.shape[-2:]
            if self.do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=self.patch_size * self.merge_size,
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels,
                )
                stacked_images = self.resize(
                    image=stacked_images,
                    size=SizeDict(height=resized_height, width=resized_width),
                    interpolation=self.interpolation,
                )
            resized_images_grouped[shape] = stacked_images

        grouped_images = None
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)
        resized_images_grouped = None

        grouped_images, grouped_images_index = group_images_by_shape(
            resized_images, disable_grouping=self.disable_grouping
        )
        resized_images = None

        processed_images_grouped = {}
        processed_grids = {}

        for shape, stacked_images in grouped_images.items():
            stacked_images = stacked_images.to("cuda", non_blocking=True)

            resized_height, resized_width = stacked_images.shape[-2:]

            patches = self.rescale_and_normalize(
                stacked_images,
                self.do_rescale,
                self.rescale_factor,
                self.do_normalize,
                self.image_mean,
                self.image_std,
            )
            if patches.ndim == 4:
                patches = patches.unsqueeze(1)

            if patches.shape[1] % self.temporal_patch_size != 0:
                repeats = patches[:, -1:].repeat(1, self.temporal_patch_size - 1, 1, 1, 1)
                patches = torch.cat([patches, repeats], dim=1)

            batch_size, grid_t, channel = patches.shape[:3]
            grid_t = grid_t // self.temporal_patch_size
            grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size

            patches = (
                patches.view(
                    batch_size,
                    grid_t,
                    self.temporal_patch_size,
                    channel,
                    grid_h // self.merge_size,
                    self.merge_size,
                    self.patch_size,
                    grid_w // self.merge_size,
                    self.merge_size,
                    self.patch_size,
                )
                .permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
                .contiguous()
            )

            flatten_patches = patches.view(
                batch_size,
                grid_t * grid_h * grid_w,
                channel * self.temporal_patch_size * self.patch_size * self.patch_size,
            )

            processed_images_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        grouped_images = None

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_grids = reorder_images(processed_grids, grouped_images_index)

        pixel_values = torch.cat(processed_images, dim=0)
        image_grid_thw = torch.as_tensor(processed_grids)

        return pixel_values, image_grid_thw
