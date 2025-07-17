import math
import torch
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


# copy from https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py#L60
def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 256 * 28 * 28, max_pixels: int = 16384 * 28 * 28
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {200}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, floor_by_factor(height / beta, factor))
        w_bar = max(factor, floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    print(f"h_bar is {h_bar}")
    print(f"w_bar is {w_bar}")
    return h_bar, w_bar


def dynamic_preprocess_native_resolution(
    image, size_factor=28, min_pixels=4 * 28 * 28, max_pixels=16384 * 28 * 28, **kwargs
):
    width, height = image.size
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=size_factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    image = image.resize((resized_width, resized_height))

    return image


# 单图处理pixel_values
def preprocess_pixel_values(pixel_values, patch_size=14):
    c, h, w = pixel_values.shape
    grid_h = h // patch_size
    grid_w = w // patch_size

    flatten_pixel_values = (
        pixel_values.view(c, grid_h, patch_size, grid_w, patch_size)
        .permute(1, 3, 0, 2, 4)  # [grid_h, grid_w, c, patch_size, patch_size]
        .reshape(grid_h * grid_w, c * patch_size ** 2)
    )

    grid_hw = torch.tensor([[grid_h, grid_w]])

    return flatten_pixel_values, grid_hw


def get_contrasting_background(image):
    """
    Calculate the color (white or black) that is different from the average foreground color
    to use as the background color
    """
    image_np = np.array(image)
    if (image_np[:, :, 3] == 0).any():
        non_transparent_pixels = image_np[:, :, :3][image_np[:, :, 3] > 0]
        if non_transparent_pixels.size == 0:
            return None
        pixel_mean = non_transparent_pixels.mean()
        contrasting_color = (0, 0, 0) if pixel_mean > 382.5 else (255, 255, 255)
        return contrasting_color
    else:
        return None


def get_image_token(image_w, image_h, patch_size=14, downsample_ratio=0.5):
    image_w, image_h = smart_resize(image_w, image_h)
    num_image_token = int(image_w * image_h // patch_size ** 2 * downsample_ratio ** 2)
    return num_image_token


def load_image_naive(
    image_file, patch_size=14, downsample_ratio=0.5, min_pixels=256 * 28 * 28, max_pixels=3328 * 28 * 28
):
    """
    Load and preprocess an image file, converting it to RGB mode,
    resizing, normalizing, and optionally adding a thumbnail version.
    """

    image = image_file
    if image.mode == "RGBA":
        bg_color = get_contrasting_background(image)
        if bg_color:
            background = Image.new("RGB", image.size, bg_color)
            background.paste(image, mask=image.split()[3])
            image = background.convert("RGB")
        else:
            image = image.convert("RGB")
    else:
        image = image.convert("RGB")

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    image = dynamic_preprocess_native_resolution(
        image, size_factor=int(patch_size // downsample_ratio), min_pixels=min_pixels, max_pixels=max_pixels
    )
    pixel_values, grid_hw = preprocess_pixel_values(transform(image))
    # num_image_token = get_image_token(image, patch_size=patch_size, downsample_ratio=downsample_ratio)

    return pixel_values, grid_hw
