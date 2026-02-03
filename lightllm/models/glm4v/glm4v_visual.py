import os
import json
import torch
import torch.nn as nn
from PIL import Image
from io import BytesIO
from typing import List, Optional
from torch.nn import LayerNorm
import torch.nn.functional as F
from safetensors import safe_open
from transformers.activations import ACT2FN
from lightllm.server.multimodal_params import MultimodalParams, ImageItem
from lightllm.server.embed_cache.utils import read_shm, get_shm_name_data
from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager
from lightllm.models.vit.triton_kernel.flashattention_nopad import flash_attention_fwd
from lightllm.models.qwen2_vl.triton_kernel.rotary_pos_emb import apply_rotary_pos_emb_triton
from lightllm.models.qwen2_vl.vision_process import Qwen2VLImageProcessor


class Glm4vRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Glm4vRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Glm4VisionMlp(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class Glm4vVisionPatchEmbed(nn.Module):
    def __init__(self, patch_size: int, temporal_patch_size: int, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states).view(-1, self.embed_dim)
        return hidden_states


class Glm4vVisionRotaryEmbedding(nn.Module):
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


class Glm4vVisionPatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, hidden_act: str, bias: bool = False) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.post_projection_norm = LayerNorm(dim)
        self.gate_proj = nn.Linear(dim, context_dim, bias=bias)
        self.up_proj = nn.Linear(dim, context_dim, bias=bias)
        self.down_proj = nn.Linear(context_dim, dim, bias=bias)
        self.act1 = nn.GELU()
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.proj(hidden_state)
        hidden_state = self.act1(self.post_projection_norm(hidden_state))
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class Glm4vVisionEmbeddings(nn.Module):
    def __init__(self, hidden_size: int, image_size: int, patch_size: int):
        super().__init__()
        self.embed_dim = hidden_size
        self.image_size = image_size
        self.patch_size = patch_size

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.position_ids = torch.arange(self.num_positions).expand((1, -1))

    def forward(self, embeddings, lengths, image_shapes, h_coords, w_coords) -> torch.Tensor:
        """
        Forward pass with integrated position encoding adaptation using 2D interpolation.

        Args:
            embeddings: Input embeddings tensor
            lengths (torch.Tensor): Sequence lengths for each image in the batch.
            image_shapes (torch.Tensor): Tensor of shape [batch_size, 3] representing the image shapes (t, h, w).
            h_coords (torch.Tensor): Tensor of shape [total_seq] representing the h coordinate for each patch.
            w_coords (torch.Tensor): Tensor of shape [total_seq] representing the w coordinate for each patch.

        Returns:
            torch.Tensor: Embeddings with adapted position encoding added.
        """
        # Get position embedding parameters
        pos_embed_weight = self.position_embedding.weight
        hidden_size = pos_embed_weight.shape[1]
        total_seq = h_coords.shape[0]
        device = pos_embed_weight.device

        # Move coordinates to correct device
        h_coords, w_coords = h_coords.to(device), w_coords.to(device)

        # Handle empty sequence case
        if total_seq == 0:
            adapted_pos_embed = torch.empty(0, hidden_size, device=device, dtype=pos_embed_weight.dtype)
        else:
            # Convert inputs to tensors if needed
            if isinstance(lengths, list):
                lengths = torch.tensor(lengths, device=device, dtype=torch.long)
            if not isinstance(image_shapes, torch.Tensor):
                image_shapes = torch.tensor(image_shapes, device=device, dtype=torch.long)

            # Prepare 2D position embedding
            orig_size_sq = pos_embed_weight.shape[0]
            orig_size = int(orig_size_sq ** 0.5)
            pos_embed_2d = (
                pos_embed_weight.view(orig_size, orig_size, hidden_size)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device=device, dtype=torch.float32)
            )

            # Calculate target dimensions for each patch
            target_h = torch.cat([image_shapes[i, 1].repeat(lengths[i]) for i in range(len(lengths))]).to(
                device=device, dtype=torch.float32
            )
            target_w = torch.cat([image_shapes[i, 2].repeat(lengths[i]) for i in range(len(lengths))]).to(
                device=device, dtype=torch.float32
            )

            # Normalize coordinates to [-1, 1] range for grid_sample
            h_coords = h_coords.to(device=device, dtype=torch.float32)
            w_coords = w_coords.to(device=device, dtype=torch.float32)
            norm_w = ((w_coords + 0.5) / target_w) * 2 - 1
            norm_h = ((h_coords + 0.5) / target_h) * 2 - 1

            # Create sampling grid
            grid = torch.stack((norm_w, norm_h), dim=-1).unsqueeze(0).unsqueeze(2)

            # Perform bicubic interpolation
            interpolated_embed_fp32 = F.grid_sample(
                pos_embed_2d, grid, mode="bicubic", align_corners=False, padding_mode="border"
            )

            # Reshape and convert back to original dtype
            adapted_pos_embed_fp32 = interpolated_embed_fp32.squeeze(0).squeeze(-1).permute(1, 0)
            adapted_pos_embed = adapted_pos_embed_fp32.to(pos_embed_weight.dtype).to(embeddings.device)

        # Add adapted position encoding to embeddings
        embeddings = embeddings + adapted_pos_embed
        return embeddings


class Glm4vVisionAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, attention_bias: bool = False, attention_dropout: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
        self.qkv = nn.Linear(dim, dim * 3, bias=attention_bias)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = attention_dropout
        self.is_causal = False

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


class Glm4vVisionBlock(nn.Module):
    def __init__(self, embed_dim, intermediate_size, num_heads, hidden_act, rms_norm_eps) -> None:
        super().__init__()
        self.norm1 = Glm4vRMSNorm(embed_dim, eps=rms_norm_eps)
        self.norm2 = Glm4vRMSNorm(embed_dim, eps=rms_norm_eps)
        self.attn = Glm4vVisionAttention(embed_dim, num_heads=num_heads)
        self.mlp = Glm4VisionMlp(
            hidden_size=embed_dim, intermediate_size=intermediate_size, hidden_act=hidden_act, bias=False
        )

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


class Glm4vVisionTransformerPretrainedModel(nn.Module):
    def __init__(
        self,
        kvargs,
        depth=24,
        image_size=336,
        hidden_size=1536,
        intermediate_size=13696,
        out_hidden_size=4096,
        hidden_act="silu",
        num_heads=12,
        in_channels=3,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        rms_norm_eps=1e-5,
        **kwargs,
    ):
        super().__init__()
        self.data_type = kvargs.get("data_type", "bfloat16")
        self.depth = depth
        self.intermediate_size = intermediate_size
        self.out_hidden_size = out_hidden_size
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size

        self.embeddings = Glm4vVisionEmbeddings(hidden_size, image_size, patch_size)
        self.patch_embed = Glm4vVisionPatchEmbed(patch_size, temporal_patch_size, in_channels, self.hidden_size)

        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = Glm4vVisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [
                Glm4vVisionBlock(self.hidden_size, self.out_hidden_size, num_heads, hidden_act, rms_norm_eps)
                for _ in range(self.depth)
            ]
        )
        self.merger = Glm4vVisionPatchMerger(
            dim=self.out_hidden_size, context_dim=self.intermediate_size, hidden_act=hidden_act
        )

        self.post_conv_layernorm = Glm4vRMSNorm(hidden_size, eps=rms_norm_eps)
        self.downsample = nn.Conv2d(
            in_channels=hidden_size,
            out_channels=out_hidden_size,
            kernel_size=spatial_merge_size,
            stride=spatial_merge_size,
        )
        self.post_layernorm = Glm4vRMSNorm(hidden_size, eps=rms_norm_eps)

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
                    if "model.visual" in k:
                        weight_dict[k[len("model.visual.") :]] = v
        else:
            hf_weight_files = [file_ for file_ in os.listdir(weight_dir) if file_.endswith(".safetensors")]
            weight_dict = {}
            for file_ in hf_weight_files:
                f = safe_open(os.path.join(weight_dir, file_), "pt", "cpu")
                for k in f.keys():
                    if "model.visual" in k:
                        weight_dict[k[len("model.visual.") :]] = f.get_tensor(k)

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
        return cos, sin, pos_ids

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = self.post_conv_layernorm(hidden_states)
        rotary_cos, rotary_sin, pos_ids = self.rot_pos_emb(grid_thw)
        rotary_cos = rotary_cos.to("cuda", non_blocking=True)
        rotary_sin = rotary_sin.to("cuda", non_blocking=True)
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        cu_seqlens = cu_seqlens.to("cuda", non_blocking=True)
        hidden_states = self.embeddings(hidden_states, seqlens, grid_thw, pos_ids[:, 0], pos_ids[:, 1])

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                rotary_cos=rotary_cos,
                rotary_sin=rotary_sin,
            )
        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = hidden_states.view(
            -1, self.spatial_merge_size, self.spatial_merge_size, hidden_states.shape[-1]
        )
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        hidden_states = self.downsample(hidden_states).view(-1, self.out_hidden_size)
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
