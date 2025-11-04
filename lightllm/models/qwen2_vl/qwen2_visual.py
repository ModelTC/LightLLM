# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
from typing import List
from torchvision import transforms as T
from lightllm.server.embed_cache.utils import read_shm, get_shm_name_data
from io import BytesIO
import torch.nn as nn
from torch.nn import LayerNorm
from transformers.activations import ACT2FN
from safetensors import safe_open
from lightllm.server.multimodal_params import ImageItem
from lightllm.models.qwen2_vl.vision_process import resize_image, Qwen2VLImageProcessor
from lightllm.models.vit.triton_kernel.flashattention_nopad import flash_attention_fwd
from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager
from lightllm.models.qwen2_vl.triton_kernel.rotary_pos_emb import apply_rotary_pos_emb_triton

# adapted from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py


class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states).view(-1, self.embed_dim)
        return hidden_states


class PatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        self.ln_q = LayerNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x


class VisionMlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, hidden_act: str) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = ACT2FN[hidden_act]
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# copy form vllm
class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self._seq_len_cached = 0
        self._freqs_cos_cached = None
        self._freqs_sin_cached = None

    def update_freqs_cache(self, seqlen: int) -> None:
        if seqlen > self._seq_len_cached:
            seqlen *= 2
            self._seq_len_cached = seqlen
            self.inv_freq = 1.0 / (
                self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float, device=self.inv_freq.device) / self.dim)
            )
            seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(seq, self.inv_freq)
            self._freqs_cos_cached = freqs.cos()
            self._freqs_sin_cached = freqs.sin()

    def forward(self, seqlen: int) -> torch.Tensor:
        self.update_freqs_cache(seqlen)
        return self._freqs_cos_cached[:seqlen], self._freqs_sin_cached[:seqlen]


class VisionFlashAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int = 0,
        rotary_cos: torch.Tensor = None,
        rotary_sin: torch.Tensor = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q = apply_rotary_pos_emb_triton(q, rotary_cos, rotary_sin)
        k = apply_rotary_pos_emb_triton(k, rotary_cos, rotary_sin)

        attn_output = g_cache_manager.alloc_tensor(q.shape, q.dtype, device=q.device)

        flash_attention_fwd(q, k, v, attn_output, cu_seqlens, max_seqlen)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen2VLVisionBlock(nn.Module):
    def __init__(self, embed_dim, mlp_ratio, num_heads, hidden_act) -> None:
        super().__init__()
        self.norm1 = LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = LayerNorm(embed_dim, eps=1e-6)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)

        self.attn = VisionFlashAttention(embed_dim, num_heads=num_heads)
        self.mlp = VisionMlp(dim=embed_dim, hidden_dim=mlp_hidden_dim, hidden_act=hidden_act)

    def forward(self, hidden_states, cu_seqlens, max_seqlen, rotary_cos, rotary_sin) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen2VisionTransformerPretrainedModel(nn.Module):
    def __init__(
        self,
        kvargs,
        depth=32,
        embed_dim=1280,
        hidden_size=3584,
        hidden_act="quick_gelu",
        mlp_ratio=4,
        num_heads=16,
        in_channels=3,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        **kwargs,
    ):
        super().__init__()
        self.data_type = kvargs.get("data_type", "bfloat16")

        self.depth = depth
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size

        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            temporal_patch_size=self.temporal_patch_size,
            in_channels=self.in_channels,
            embed_dim=self.embed_dim,
        )

        head_dim = self.embed_dim // self.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2).cuda()

        self.blocks = nn.ModuleList(
            [
                Qwen2VLVisionBlock(self.embed_dim, self.mlp_ratio, self.num_heads, self.hidden_act)
                for _ in range(self.depth)
            ]
        )
        self.merger = PatchMerger(dim=self.hidden_size, context_dim=self.embed_dim)

        self._init_datatype()

    def _init_datatype(self):
        if isinstance(self.data_type, torch.dtype):
            return
        if self.data_type in ["fp16", "float16"]:
            self.data_type = torch.float16
        elif self.data_type in ["bf16", "bfloat16"]:
            self.data_type = torch.bfloat16
        elif self.data_type in ["fp32", "float32"]:
            self.data_type = torch.float32
        else:
            raise ValueError(f"Unsupport datatype {self.data_type}!")
        return

    def load_model(self, weight_dir):

        processor_config_path = os.path.join(weight_dir, "preprocessor_config.json")
        with open(processor_config_path, "r") as f:
            processor_config_dict = json.load(f)
        self.processor = Qwen2VLImageProcessor(**processor_config_dict)

        bin_weight_files = [file_ for file_ in os.listdir(weight_dir) if file_.endswith(".bin")]
        if bin_weight_files:
            weight_dict = {}
            for file_ in bin_weight_files:
                f = torch.load(os.path.join(weight_dir, file_), "cpu")
                for k, v in f.items():
                    if "visual" in k:
                        weight_dict[k[len("visual.") :]] = v
        else:
            hf_weight_files = [file_ for file_ in os.listdir(weight_dir) if file_.endswith(".safetensors")]
            weight_dict = {}
            for file_ in hf_weight_files:
                f = safe_open(os.path.join(weight_dir, file_), "pt", "cpu")
                for k in f.keys():
                    if "visual" in k:
                        weight_dict[k[len("visual.") :]] = f.get_tensor(k)

        self.load_state_dict(weight_dict)

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        s = self.spatial_merge_size
        for _, h, w in grid_thw:
            pos_shape = (h // s, s, w // s, s)
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            hpos_ids = hpos_ids.reshape(pos_shape).permute(0, 2, 1, 3).flatten()
            wpos_ids = wpos_ids.reshape(pos_shape).permute(0, 2, 1, 3).flatten()

            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        cos_full, sin_full = self.rotary_pos_emb(max_grid_size)
        cos = cos_full[pos_ids].flatten(1)
        sin = sin_full[pos_ids].flatten(1)
        return cos, sin

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        rotary_cos, rotary_sin = self.rot_pos_emb(grid_thw)
        rotary_cos = rotary_cos.to("cuda", non_blocking=True)
        rotary_sin = rotary_sin.to("cuda", non_blocking=True)
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

        cu_seqlens = cu_seqlens.to("cuda", non_blocking=True)
        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                rotary_cos=rotary_cos,
                rotary_sin=rotary_sin,
            )
        return self.merger(hidden_states)

    def encode(self, images: List[ImageItem]):
        img_tensors = []
        valid_ids = []
        valid_id = 0
        img_grids = []
        uuids = []
        for i, img in enumerate(images):
            if isinstance(img, ImageItem):
                uuids.append(img.uuid)
                image_data = read_shm(get_shm_name_data(img.uuid))
                image_data = Image.open(BytesIO(image_data))
                pixel_values, image_grid_thw = self.processor.preprocess(image_data)
                img_tensors.append(pixel_values)
                img_grids.append(image_grid_thw)
            else:
                raise Exception("Unsupport input types: {} for {}".format(type(img), img))

            # must devide merge_length
            cur_num = img_tensors[-1].shape[0] // (self.spatial_merge_size ** 2)

            valid_ids.append([valid_id, valid_id + cur_num])
            valid_id += cur_num

        if len(img_tensors) <= 0:
            return None

        imgs = torch.cat(img_tensors, dim=0)
        grid_thw = torch.cat(img_grids, dim=0)

        pixel_values = imgs.to("cuda", dtype=self.data_type, non_blocking=True)
        image_grid_thw = grid_thw.to("cuda", non_blocking=True)

        all_img_embeds = self.forward(pixel_values, grid_thw=image_grid_thw)

        return all_img_embeds, uuids, valid_ids
