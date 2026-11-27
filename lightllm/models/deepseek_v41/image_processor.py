import math

from lightllm.models.deepseek_v4.image_processor import get_image_size


def get_image_grid(height, width, patch_size, downsample_ratio, max_n_token, min_pixels, max_wh_ratio):
    height, width = get_image_size(height, width, min_pixels, max_wh_ratio)
    n_vit_h, n_vit_w = math.ceil(height / patch_size), math.ceil(width / patch_size)
    cell = patch_size * downsample_ratio
    budget = max_n_token
    while True:
        n_llm_h, n_llm_w = math.ceil(n_vit_h / downsample_ratio), math.ceil(n_vit_w / downsample_ratio)
        tokens = n_llm_h * (n_llm_w + 1) + 2
        if tokens <= max_n_token:
            return n_vit_h, n_vit_w, n_llm_h, n_llm_w, tokens
        ratio = height / width
        max_w = math.sqrt((budget - 2) / ratio + 0.25) - 0.5
        max_h = max_w * ratio
        if max_w < 1:
            n_vit_h, n_vit_w = (budget - 2) // 2 * downsample_ratio, downsample_ratio
        elif max_h < 1:
            n_vit_h, n_vit_w = downsample_ratio, (budget - 3) * downsample_ratio
        else:
            scale = min(math.floor(max_w) * cell / width, math.floor(max_h) * cell / height)
            n_vit_h, n_vit_w = math.floor(height * scale / patch_size), math.floor(width * scale / patch_size)
        budget -= 1
