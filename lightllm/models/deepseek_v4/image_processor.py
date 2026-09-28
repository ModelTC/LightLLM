# Copyright (c) 2023 DeepSeek

import math
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps


IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(5)
COMPRESS_PAD_TO = 4


def grid_tokens(best_height, best_width, patch_size, downsample_ratio):
    """Number of LLM tokens occupied by an N-layout image block without compressor padding."""
    n_llm_h = math.ceil((best_height // patch_size) / downsample_ratio)
    n_llm_w = math.ceil((best_width // patch_size) / downsample_ratio)
    num_tokens = n_llm_h * (n_llm_w + 1) + 2
    if n_llm_h % 2 == 1:
        num_tokens += n_llm_w + 1
    num_tokens += (n_llm_h + 1) // 2 * (n_llm_w + 1) % 2 * 2
    return n_llm_h, n_llm_w, num_tokens


def solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token):
    ratio = height / width
    max_w_float = math.sqrt((max_n_token - 2) / ratio + 0.25) - 0.5
    max_h_float = max_w_float * ratio
    if max_w_float < 1.0:
        max_w = 1
        max_h = (max_n_token - 2) // (max_w + 1)
        if max_h % 2 == 1:
            max_h -= 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    elif max_h_float < 2.0:
        max_h = 2
        max_w = ((max_n_token - 2) // max_h) - 1
        assert max_w > 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    else:
        max_w = math.floor(max_w_float)
        max_h = math.floor(max_h_float)
        if max_h % 2 == 1:
            max_h -= 1
        beta = min(
            max_w * patch_size * downsample_ratio / width,
            max_h * patch_size * downsample_ratio / height,
        )
        best_width = math.floor(width * beta / patch_size) * patch_size
        best_height = math.floor(height * beta / patch_size) * patch_size
    n_llm_h, n_llm_w, num_tokens = grid_tokens(best_height, best_width, patch_size, downsample_ratio)
    return n_llm_h, n_llm_w, best_height, best_width, num_tokens


def safe_resize(height, width, best_height, best_width, patch_size, downsample_ratio, max_n_token):
    max_n_token -= COMPRESS_PAD_TO - 1
    n_llm_h, n_llm_w, num_tokens = grid_tokens(best_height, best_width, patch_size, downsample_ratio)
    budget = max_n_token
    while num_tokens > max_n_token:
        n_llm_h, n_llm_w, best_height, best_width, num_tokens = solve_resize_ratio(
            height, width, patch_size, downsample_ratio, budget
        )
        budget -= 1
    return n_llm_h, n_llm_w, best_height, best_width


def get_image_grid(
    height: int,
    width: int,
    patch_size: int,
    downsample_ratio: int,
    max_n_token: int,
    min_pixels: int,
    max_wh_ratio: Optional[float],
) -> Tuple[int, int, int, int, int]:
    """Resolve ViT/LLM grids and the position-independent canonical block length."""
    if max_wh_ratio is not None and width > height * max_wh_ratio:
        width = height * max_wh_ratio
    if 0 < width * height < min_pixels:
        ratio = (min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)

    best_width = math.ceil(width / patch_size) * patch_size
    best_height = math.ceil(height / patch_size) * patch_size
    n_llm_h, n_llm_w, best_height, best_width = safe_resize(
        height,
        width,
        best_height,
        best_width,
        patch_size,
        downsample_ratio,
        max_n_token,
    )
    n_vit_h = best_height // patch_size
    n_vit_w = best_width // patch_size
    _, _, core_token_num = grid_tokens(best_height, best_width, patch_size, downsample_ratio)
    canonical_token_num = COMPRESS_PAD_TO - 1 + core_token_num
    return n_vit_h, n_vit_w, n_llm_h, n_llm_w, canonical_token_num


def load_image(image: Image.Image, args):
    """Transform one image into BF16 ViT patches using the checkpoint preprocessing contract."""
    image = image.convert("RGB")
    n_vit_h, n_vit_w, n_llm_h, n_llm_w, _ = get_image_grid(
        image.height,
        image.width,
        args.vision_patch_size,
        args.vision_downsample_ratio,
        args.vision_max_n_token,
        args.vision_min_pixels,
        args.vision_max_wh_ratio,
    )
    best_height = n_vit_h * args.vision_patch_size
    best_width = n_vit_w * args.vision_patch_size
    if args.vision_max_wh_ratio is not None and image.width >= args.vision_max_wh_ratio * image.height:
        image = image.resize((best_width, best_height))
    else:
        image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))

    x = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255
    x = ((x - 0.5) / 0.5).to(torch.bfloat16)
    patch_size = args.vision_patch_size
    patches = (
        x.reshape(3, n_vit_h, patch_size, n_vit_w, patch_size)
        .permute(1, 3, 0, 2, 4)
        .reshape(n_vit_h * n_vit_w, 3, patch_size, patch_size)
    )
    return patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w


def build_image_block(n_llm_h: int, n_llm_w: int, start_pos: int):
    """Build final N-layout token types and the row-major aligner permutation."""
    compress_pad = COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO
    pad_h = n_llm_h % 2
    rows = n_llm_h + pad_h
    row_len = n_llm_w + 1
    pad_last = rows // 2 * row_len % 2 * 2
    types = torch.tensor(
        ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_PAD] * (row_len * pad_h),
        dtype=torch.int64,
    )
    order = torch.arange(rows * row_len).view(rows // 2, 2, row_len).transpose(1, 2).reshape(-1)
    image_idx = torch.full((rows * row_len,), -1, dtype=torch.int64)
    image_idx.view(rows, row_len)[:n_llm_h, :n_llm_w] = torch.arange(n_llm_h * n_llm_w).view(n_llm_h, n_llm_w)
    perm = image_idx[order]
    perm = perm[perm >= 0]
    types = torch.cat(
        [
            torch.full((compress_pad,), IMAGE_PAD, dtype=torch.int64),
            torch.tensor([IMAGE_START]),
            types[order],
            torch.full((pad_last,), IMAGE_PAD, dtype=torch.int64),
            torch.tensor([IMAGE_END]),
        ]
    )
    return types, perm


def build_canonical_image_block(n_llm_h: int, n_llm_w: int):
    """Build the cacheable image block with its maximum three compressor-alignment pads."""
    return build_image_block(n_llm_h, n_llm_w, start_pos=0)
