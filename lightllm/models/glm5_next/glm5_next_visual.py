# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 the HuggingFace Team. All rights reserved.
# Adapted from Hugging Face Transformers' GLM-5-Next vision encoder.

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from lightllm.models.qwen2_vl.qwen2_visual import Qwen2VisionTransformerPretrainedModel
from lightllm.models.qwen2_vl.triton_kernel.rotary_pos_emb import apply_rotary_pos_emb_triton
from lightllm.models.vit.triton_kernel.rms_norm_vit import qk_rms_norm, rms_norm
from lightllm.server.visualserver import get_vit_attn_backend
from .vision_process import Glm5NextImageProcessor


class Glm5NextVisionRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        return rms_norm(hidden_states, self.weight, eps=self.eps, round_norm_before_weight=True)


class Glm5NextVisionMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, limit, bias=False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.limit = limit

    def forward(self, x):
        gate = self.gate_proj(x).clamp(max=self.limit)
        up = self.up_proj(x).clamp(min=-self.limit, max=self.limit)
        return self.down_proj(F.silu(gate) * up)


class Glm5NextVisionAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, eps, bias):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.q_norm = Glm5NextVisionRMSNorm(hidden_size // num_heads, eps)
        self.k_norm = Glm5NextVisionRMSNorm(hidden_size // num_heads, eps)

    def forward(self, x, cu_seqlens, max_seqlen, rotary_cos, rotary_sin):
        qkv = self.qkv(x).reshape(x.shape[0], 3, self.num_heads, -1)
        q, k = qk_rms_norm(qkv, self.q_norm.weight, self.k_norm.weight, self.q_norm.eps)
        v = qkv[:, 2]
        q = apply_rotary_pos_emb_triton(q, rotary_cos, rotary_sin)
        k = apply_rotary_pos_emb_triton(k, rotary_cos, rotary_sin)
        out = torch.empty_like(q)
        get_vit_attn_backend()(q, k, v, out, cu_seqlens, max_seqlen)
        return self.proj(out.reshape(x.shape[0], -1))


class Glm5NextVisionBlock(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_heads, rms_norm_eps, swiglu_limit, attention_bias):
        super().__init__()
        self.norm1 = Glm5NextVisionRMSNorm(hidden_size, rms_norm_eps)
        self.norm2 = Glm5NextVisionRMSNorm(hidden_size, rms_norm_eps)
        self.attn = Glm5NextVisionAttention(hidden_size, num_heads, rms_norm_eps, attention_bias)
        self.mlp = Glm5NextVisionMLP(hidden_size, intermediate_size, swiglu_limit, attention_bias)

    def forward(self, x, cu_seqlens, max_seqlen, rotary_cos, rotary_sin):
        x = x + self.attn(self.norm1(x), cu_seqlens, max_seqlen, rotary_cos, rotary_sin)
        return x + self.mlp(self.norm2(x))


class Glm5NextVisionPatchEmbed(nn.Module):
    def __init__(self, in_channels, hidden_size, patch_size, temporal_patch_size):
        super().__init__()
        self.in_channels = in_channels
        self.kernel = (temporal_patch_size, patch_size, patch_size)
        self.proj = nn.Conv3d(in_channels, hidden_size, kernel_size=self.kernel, stride=self.kernel)

    def forward(self, x):
        x = x.reshape(-1, self.in_channels, *self.kernel)
        return self.proj(x).flatten(1)


class Glm5NextVisionPatchMerger(Glm5NextVisionMLP):
    def __init__(self, hidden_size, intermediate_size, limit):
        super().__init__(hidden_size, intermediate_size, limit)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.post_projection_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        return super().forward(F.gelu(self.post_projection_norm(self.proj(x))))


class Glm5NextVisionTransformer(Qwen2VisionTransformerPretrainedModel):
    """GLM encoder using the existing variable-resolution image batching and cache interface."""

    def __init__(
        self,
        kvargs,
        hidden_size=1024,
        out_hidden_size=4096,
        depth=24,
        intermediate_size=4096,
        projection_intermediate_size=10240,
        num_heads=16,
        in_channels=3,
        patch_size=14,
        temporal_patch_size=2,
        spatial_merge_size=2,
        rms_norm_eps=1e-5,
        swiglu_limit=10.0,
        attention_bias=True,
        hidden_act="silu",
        rope_parameters=None,
        **kwargs,
    ):
        nn.Module.__init__(self)
        assert hidden_act == "silu"
        self.data_type = kvargs.get("data_type", "bfloat16")
        self._init_datatype()
        self.hidden_size = out_hidden_size
        self.spatial_merge_size = spatial_merge_size
        self.patch_size = patch_size
        self.patch_embed = Glm5NextVisionPatchEmbed(in_channels, hidden_size, patch_size, temporal_patch_size)
        rope_parameters = rope_parameters or {"rope_type": "axial", "rope_theta": 10000.0}
        assert rope_parameters["rope_type"] == "axial"
        rotary_dim = hidden_size // num_heads // 2
        self.rotary_inv_freq = 1.0 / (
            rope_parameters["rope_theta"] ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )
        self.blocks = nn.ModuleList(
            [
                Glm5NextVisionBlock(
                    hidden_size, intermediate_size, num_heads, rms_norm_eps, swiglu_limit, attention_bias
                )
                for _ in range(depth)
            ]
        )
        self.post_layernorm = Glm5NextVisionRMSNorm(hidden_size, rms_norm_eps)
        self.downsample = nn.Conv2d(
            hidden_size, out_hidden_size, kernel_size=spatial_merge_size, stride=spatial_merge_size
        )
        self.merger = Glm5NextVisionPatchMerger(out_hidden_size, projection_intermediate_size, swiglu_limit)

    def load_model(self, weight_dir):
        self.processor = Glm5NextImageProcessor.from_pretrained(weight_dir)
        weights = {}
        prefix = "model.visual."
        for filename in os.listdir(weight_dir):
            if filename.endswith(".safetensors"):
                with safe_open(os.path.join(weight_dir, filename), framework="pt", device="cpu") as f:
                    for name in f.keys():
                        if name.startswith(prefix):
                            weights[name[len(prefix) :]] = f.get_tensor(name)
        self.load_state_dict(weights, strict=True)

    def rot_pos_emb(self, grid_thw, device):
        positions = []
        size = self.spatial_merge_size
        for t, h, w in grid_thw.tolist():
            rows, cols = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            shape = (h // size, size, w // size, size)
            rows = rows.reshape(shape).transpose(1, 2).flatten()
            cols = cols.reshape(shape).transpose(1, 2).flatten()
            positions.append(torch.stack((rows, cols), -1).repeat(t, 1))
        positions = torch.cat(positions).to(device)
        # Keep frequencies in FP32 and evaluate trig on the same device as attention.
        self.rotary_inv_freq = self.rotary_inv_freq.to(device)
        angles = positions[..., None].float() * self.rotary_inv_freq
        return angles.cos().flatten(1), angles.sin().flatten(1)

    def forward(self, hidden_states, grid_thw):
        hidden_states = self.patch_embed(hidden_states)
        # Grid metadata stays on the CPU; attention and rotary tensors are copied once per batch.
        grid_thw = grid_thw.cpu()
        rotary_cos, rotary_sin = self.rot_pos_emb(grid_thw, hidden_states.device)
        lengths = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
        max_seqlen = lengths.max().item()
        cu_seqlens = F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0)).to(hidden_states.device)
        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens, max_seqlen, rotary_cos, rotary_sin)
        hidden_states = self.post_layernorm(hidden_states)
        size = self.spatial_merge_size
        hidden_states = hidden_states.reshape(-1, size, size, hidden_states.shape[-1]).permute(0, 3, 1, 2)
        hidden_states = self.downsample(hidden_states).reshape(-1, self.hidden_size)
        return self.merger(hidden_states)
