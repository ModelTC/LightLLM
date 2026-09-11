import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from lightllm.models.qwen2_vl.vision_process import smart_resize

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def dynamic_preprocess_native_resolution(
    image, size_factor=32, min_pixels=65536, max_pixels=4194304, **kwargs
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


def preprocess_pixel_values(pixel_values, patch_size=16):
    c, h, w = pixel_values.shape
    grid_h = h // patch_size
    grid_w = w // patch_size

    flatten_pixel_values = (
        pixel_values.view(c, grid_h, patch_size, grid_w, patch_size)
        .permute(1, 3, 0, 2, 4)  # [grid_h, grid_w, c, patch_size, patch_size]
        .reshape(grid_h * grid_w, c * patch_size**2)
    )

    grid_hw = torch.tensor([[grid_h, grid_w]]).to(device=pixel_values.device)

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


def load_image_native(
    image,
    patch_size=16,
    downsample_ratio=0.5,
    min_pixels=65536,
    max_pixels=4194304,
    upscale=False,
):
    """
    Load and preprocess an image file, converting it to RGB mode,
    resizing, normalizing, and optionally adding a thumbnail version.
    """
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

    if upscale:
        image = image.resize((image.width * 2, image.height * 2), Image.BILINEAR)

    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    new_image = dynamic_preprocess_native_resolution(
        image,
        size_factor=int(patch_size // downsample_ratio),
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    pixel_values, grid_hw = preprocess_pixel_values(
        transform(new_image).to(torch.float32), patch_size=patch_size
    )

    return pixel_values, grid_hw
