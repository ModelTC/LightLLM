import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Optional
from lightllm.server.embed_cache.utils import read_shm, get_shm_name_data
from io import BytesIO
import torch.nn as nn
from transformers.activations import ACT2FN
from lightllm.models.qwen2_vl.vision_process import resize_image, Qwen2VLImageProcessor
from safetensors import safe_open
from lightllm.server.multimodal_params import ImageItem
from lightllm.models.qwen2_vl.qwen2_visual import PatchEmbed, VisionRotaryEmbedding
from lightllm.models.vit.triton_kernel.rms_norm_vit import rms_norm
from lightllm.models.vit.triton_kernel.flashattention_nopad import flash_attention_fwd
from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager
from lightllm.models.qwen2_vl.triton_kernel.rotary_pos_emb import apply_rotary_pos_emb_triton


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return rms_norm(hidden_states, self.weight, eps=self.variance_epsilon)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen2_5_VLMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        bias: bool = False,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class Qwen2_5_VLVisionFlashAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int = 0,
        rotary_cos: Optional[torch.Tensor] = None,
        rotary_sin: Optional[torch.Tensor] = None,
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


class Qwen2_5_VLVisionBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        num_heads,
        hidden_act,
    ) -> None:
        super().__init__()
        self.norm1 = Qwen2RMSNorm(hidden_size, eps=1e-6)
        self.norm2 = Qwen2RMSNorm(hidden_size, eps=1e-6)

        self.attn = Qwen2_5_VLVisionFlashAttention(hidden_size, num_heads=num_heads)
        self.mlp = Qwen2_5_VLMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            bias=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int = 0,
        rotary_cos: Optional[torch.Tensor] = None,
        rotary_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen2_5_VLPatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        self.ln_q = Qwen2RMSNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x


class Qwen2_5_VisionTransformerPretrainedModel(nn.Module):
    def __init__(
        self,
        kvargs,
        depth=32,
        hidden_size=3584,
        hidden_act="silu",
        intermediate_size=3420,
        num_heads=16,
        in_channels=3,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        tokens_per_second=4,
        window_size=112,
        out_hidden_size=3584,
        fullatt_block_indexes=[7, 15, 23, 31],
        **kwargs,
    ):
        super().__init__()
        self.weight_dir = kvargs["weight_dir"]
        self.data_type = kvargs.get("data_type", "bfloat16")

        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.tokens_per_second = tokens_per_second
        self.window_size = window_size
        self.fullatt_block_indexes = fullatt_block_indexes
        self.out_hidden_size = out_hidden_size

        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            temporal_patch_size=self.temporal_patch_size,
            in_channels=self.in_channels,
            embed_dim=self.hidden_size,
        )

        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [
                Qwen2_5_VLVisionBlock(
                    self.hidden_size,
                    self.intermediate_size,
                    self.num_heads,
                    self.hidden_act,
                )
                for _ in range(self.depth)
            ]
        )

        self.merger = Qwen2_5_VLPatchMerger(
            dim=self.out_hidden_size,
            context_dim=self.hidden_size,
            spatial_merge_size=self.spatial_merge_size,
        )

        self.gradient_checkpointing = False

        processor_config_path = os.path.join(self.weight_dir, "preprocessor_config.json")
        with open(processor_config_path, "r") as f:
            processor_config_dict = json.load(f)
        self.processor = Qwen2VLImageProcessor(**processor_config_dict)

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

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        rotary_cos, rotary_sin = self.rot_pos_emb(grid_thw)
        rotary_cos = rotary_cos.to("cuda", non_blocking=True)
        rotary_sin = rotary_sin.to("cuda", non_blocking=True)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0).to("cuda", non_blocking=True)
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

        window_index, cu_window_seqlens = self.get_window_index(grid_thw)

        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens).to("cuda", non_blocking=True)
        max_window_seqlen = (cu_window_seqlens[1:] - cu_window_seqlens[:-1]).max().item()

        seq_len, _ = hidden_states.size()
        pos_shape = (seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states.reshape(pos_shape)[window_index].view(seq_len, -1)
        rotary_cos = rotary_cos.reshape(pos_shape)[window_index].view(seq_len, -1)
        rotary_sin = rotary_sin.reshape(pos_shape)[window_index].view(seq_len, -1)

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
                max_seqlen_now = max_seqlen
            else:
                cu_seqlens_now = cu_window_seqlens
                max_seqlen_now = max_window_seqlen

            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                max_seqlen=max_seqlen_now,
                rotary_cos=rotary_cos,
                rotary_sin=rotary_sin,
            )

        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]

        return hidden_states

    def load_image(self, img: List[ImageItem]):
        pixel_values = None
        if isinstance(img, ImageItem):
            image_data = read_shm(get_shm_name_data(img.uuid))
            image_data = Image.open(BytesIO(image_data))
            image_data = resize_image(image_data)
            pixel_values, image_grid_thw = self.processor.preprocess(image_data)
        elif isinstance(img, dict):
            image_data = read_shm(get_shm_name_data(img["uuid"]))
            image_data = Image.open(BytesIO(image_data))
            image_data = resize_image(image_data)
            pixel_values, image_grid_thw = self.processor.preprocess(image_data)
        else:
            raise Exception("Unsupport input types: {} for {}".format(type(img), img))
        return pixel_values.to(dtype=self.data_type), image_grid_thw

    def load_model(self, weight_dir):

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
