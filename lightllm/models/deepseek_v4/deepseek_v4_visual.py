# Copyright (c) 2023 DeepSeek

import json
import os
from functools import lru_cache
from io import BytesIO
from types import SimpleNamespace
from typing import List

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open
from torch import nn

from lightllm.models.deepseek_v4.image_processor import (
    IMAGE,
    build_canonical_image_block,
    load_image,
)
from lightllm.server.embed_cache.utils import get_shm_name_data, read_shm
from lightllm.server.multimodal_params import ImageItem


@lru_cache(8)
def get_vision_cos_sin(n_h: int, n_w: int, dim: int, theta: float):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class PatchEmbed(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.proj = nn.Linear(
            3 * args.vision_patch_size ** 2,
            args.vision_dim,
            dtype=torch.bfloat16,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(1))


class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_heads = args.vision_n_heads
        self.head_dim = args.vision_dim // args.vision_n_heads
        self.wqkv = nn.Linear(args.vision_dim, 3 * args.vision_dim, dtype=torch.bfloat16)
        self.wo = nn.Linear(args.vision_dim, args.vision_dim, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        q, k, v = (t.view(n, self.n_heads, self.head_dim) for t in self.wqkv(x).chunk(3, dim=-1))
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        o = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
        return self.wo(o.transpose(0, 1).reshape(n, -1))


class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.w1 = nn.Linear(
            args.vision_dim,
            2 * args.vision_inter_dim,
            bias=False,
            dtype=torch.bfloat16,
        )
        self.w2 = nn.Linear(
            args.vision_inter_dim,
            args.vision_dim,
            bias=False,
            dtype=torch.bfloat16,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.norm1 = RMSNorm(args.vision_dim)
        self.attn = Attention(args)
        self.norm2 = RMSNorm(args.vision_dim)
        self.mlp = MLP(args)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    """DeepSeek ViT: full bidirectional attention over one image with 2D RoPE."""

    def __init__(self, args):
        super().__init__()
        self.rope_dim = args.vision_dim // args.vision_n_heads // 2
        self.rope_theta = args.vision_rope_theta
        self.patch_embed = PatchEmbed(args)
        self.blocks = nn.ModuleList([Block(args) for _ in range(args.vision_n_layers)])
        self.norm = RMSNorm(args.vision_dim)

    def forward(self, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        x = self.patch_embed(patches)
        cos, sin = get_vision_cos_sin(n_h, n_w, self.rope_dim, self.rope_theta)
        cos = cos.to(device=x.device)
        sin = sin.to(device=x.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)


class Aligner(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.downsample_ratio = args.vision_downsample_ratio
        in_dim = args.vision_dim * self.downsample_ratio ** 2
        self.w1 = nn.Linear(in_dim, args.hidden_size, dtype=torch.bfloat16)
        self.w2 = nn.Linear(args.hidden_size, args.hidden_size, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        r = self.downsample_ratio
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % r, 0, -n_h % r))
        x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        return self.w2(F.gelu(self.w1(x)))


class DeepseekV4VisionModel(nn.Module):
    """Replicated visual tower and cacheable sentinel-block encoder."""

    _WEIGHT_PREFIXES = ("vision.", "aligner.", "image_")

    def __init__(self, kvargs, **config):
        super().__init__()
        self.config = SimpleNamespace(**config)
        self.vision = ViT(self.config)
        self.aligner = Aligner(self.config)
        self.image_start = nn.Parameter(torch.empty(self.config.hidden_size, dtype=torch.bfloat16))
        self.image_end = nn.Parameter(torch.empty(self.config.hidden_size, dtype=torch.bfloat16))
        self.image_newline = nn.Parameter(torch.empty(self.config.hidden_size, dtype=torch.bfloat16))
        self.image_pad = nn.Parameter(torch.empty(self.config.hidden_size, dtype=torch.bfloat16))

    def load_model(self, weight_dir):
        index_path = os.path.join(weight_dir, "model.safetensors.index.json")
        with open(index_path, "r") as index_file:
            weight_map = json.load(index_file)["weight_map"]

        keys_by_file = {}
        for key, file_name in weight_map.items():
            if key.startswith(self._WEIGHT_PREFIXES):
                keys_by_file.setdefault(file_name, []).append(key)

        state_dict = {}
        for file_name, keys in sorted(keys_by_file.items()):
            with safe_open(os.path.join(weight_dir, file_name), framework="pt", device="cpu") as weight_file:
                for key in keys:
                    state_dict[key] = weight_file.get_tensor(key)
        self.load_state_dict(state_dict)
        return self

    @torch.inference_mode()
    def encode(self, images: List[ImageItem]):
        blocks = []
        uuids = []
        valid_ids = []
        valid_start = 0
        sentinel_table = torch.stack(
            [
                self.image_start,
                self.image_pad,
                self.image_pad,
                self.image_newline,
                self.image_end,
            ]
        )

        for image_item in images:
            uuids.append(image_item.uuid)
            image_data = read_shm(get_shm_name_data(image_item.uuid))
            with Image.open(BytesIO(image_data)) as image:
                patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = load_image(image, self.config)

            patches = patches.to(device=self.image_start.device, non_blocking=True)
            row_major_embeds = self.aligner(
                self.vision(patches, n_vit_h, n_vit_w),
                n_vit_h,
                n_vit_w,
            )
            types, perm = build_canonical_image_block(n_llm_h, n_llm_w)
            types = types.to(device=row_major_embeds.device)
            block = sentinel_table[types]
            block[types == IMAGE] = row_major_embeds[perm.to(device=row_major_embeds.device)]
            blocks.append(block)

            valid_end = valid_start + block.shape[0]
            valid_ids.append([valid_start, valid_end])
            valid_start = valid_end

        return torch.cat(blocks), uuids, valid_ids
