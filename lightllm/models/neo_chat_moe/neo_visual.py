import os
import torch
import torch.nn.functional as F
from PIL import Image
from typing import List
from io import BytesIO
import torch.nn as nn
from transformers.activations import ACT2FN
from safetensors import safe_open
from lightllm.server.multimodal_params import ImageItem
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from lightllm.models.neo_chat_moe.vision_process import load_image_native
from lightllm.server.embed_cache.utils import read_shm, get_shm_name_data


def apply_rotary_emb_1d(
    x: torch.Tensor,
    cos_cached: torch.Tensor,
    sin_cached: torch.Tensor,
    positions: torch.Tensor,
):
    """对输入张量的一部分应用1D RoPE。"""
    # x: (..., seq_len, dim_part)
    # positions: (..., seq_len)
    # cos_cached: (max_pos, dim_part / 2)
    cos_cached = cos_cached.to(device=positions.device)
    sin_cached = sin_cached.to(device=positions.device)

    cos = cos_cached[positions]  # Shape: (positions.shape, dim_part / 2)
    sin = sin_cached[positions]  # Shape: (positions.shape, dim_part / 2)

    x1 = x[..., 0::2]
    x2 = x[..., 1::2]

    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos

    x_rotated = torch.empty_like(x)
    x_rotated[..., 0::2] = rotated_x1
    x_rotated[..., 1::2] = rotated_x2
    return x_rotated


def apply_2d_rotary_pos_emb(
    x: torch.Tensor,
    cos_cached_x: torch.Tensor,
    sin_cached_x: torch.Tensor,
    cos_cached_y: torch.Tensor,
    sin_cached_y: torch.Tensor,
    abs_positions_x: torch.Tensor,
    abs_positions_y: torch.Tensor,
):
    """应用2D RoPE到输入张量x。"""
    dim = x.shape[-1]
    dim_half = dim // 2

    # 假设我们将embedding的前半部分用于一个方向的RoPE，后半部分用于另一个方向
    # 例如，前一半给X坐标，后一半给Y坐标 (或者反过来，但要保持一致)
    x_part_1 = x[..., :dim_half]
    x_part_2 = x[..., dim_half:]

    # 将与 abs_positions_x 相关的旋转应用于 x_part_1
    rotated_part_1 = apply_rotary_emb_1d(x_part_1, cos_cached_x, sin_cached_x, abs_positions_x)
    # 将与 abs_positions_y 相关的旋转应用于 x_part_2
    rotated_part_2 = apply_rotary_emb_1d(x_part_2, cos_cached_y, sin_cached_y, abs_positions_y)

    # 将它们重新拼接起来。确保顺序与你分割时一致。
    return torch.cat((rotated_part_1, rotated_part_2), dim=-1)


def build_abs_positions_from_grid_hw(grid_hw: torch.Tensor, device=None):
    """
    Compute patch coordinates (x, y)

    Args:
        grid_hw: (B, 2) tensor representing (H, W) per image
    """
    device = grid_hw.device
    B = grid_hw.shape[0]

    # Get the number of patches per image
    H = grid_hw[:, 0]
    W = grid_hw[:, 1]
    N = H * W
    N_total = N.sum()

    # Create the batch index for each patch (B x patch count)
    patch_to_sample = torch.repeat_interleave(torch.arange(B, device=device), N)  # (N_total,)

    # Generate intra-image patch index (row-major order)
    patch_id_within_image = torch.arange(N_total, device=device)
    patch_id_within_image = (
        patch_id_within_image
        - torch.cumsum(torch.cat([torch.tensor([0], device=device), N[:-1]]), dim=0)[patch_to_sample]
    )

    # Get H/W for each patch according to its image
    W_per_patch = W[patch_to_sample]
    abs_x = patch_id_within_image % W_per_patch
    abs_y = patch_id_within_image // W_per_patch

    return abs_x, abs_y


class NeoVisionTransformerPretrainedModel(nn.Module):
    def __init__(
        self,
        kvargs,
        hidden_size: int = 1024,
        llm_hidden_size: int = 2048,
        downsample_ratio: float = 0.5,
        patch_size: int = 16,
        num_channels: int = 3,
        max_position_embeddings_vision: int = 10000,
        rope_theta_vision: float = 10000.0,
        min_pixels: int = 65536,
        max_pixels: int = 2408448,
        **kwargs,
    ):
        super().__init__()
        self.weight_dir = kvargs["weight_dir"]
        self.data_type = kvargs.get("data_type", "bfloat16")
        self.embed_dim = hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.downsample_ratio = downsample_ratio
        self.downsample_factor = int(1 / downsample_ratio)
        self.max_position_embeddings_vision = max_position_embeddings_vision
        self.rope_theta_vision = rope_theta_vision
        self.rope_dim_part = self.embed_dim // 2
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        self.patch_embedding = nn.Conv2d(
            in_channels=num_channels, out_channels=self.embed_dim, kernel_size=patch_size, stride=patch_size
        )

        self.dense_embedding = nn.Conv2d(
            in_channels=self.embed_dim,
            out_channels=self.llm_hidden_size,
            kernel_size=self.downsample_factor,
            stride=self.downsample_factor,
        )
        self.gelu = nn.GELU()

        self.repe_dim_part = self.embed_dim // 2
        self.cos_x, self.sin_x = self.precompute_rope_freqs_sincos()
        self.cos_y, self.sin_y = self.precompute_rope_freqs_sincos()
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
        bin_weight_files = [file_ for file_ in os.listdir(weight_dir) if file_.endswith(".bin")]
        if bin_weight_files:
            weight_dict = {}
            for file_ in bin_weight_files:
                f = torch.load(os.path.join(weight_dir, file_), "cpu")
                for k, v in f.items():
                    if "vision_model" in k:
                        weight_dict[k[len("vision_model.embeddings.") :]] = v
        else:
            hf_weight_files = [file_ for file_ in os.listdir(weight_dir) if file_.endswith(".safetensors")]
            weight_dict = {}
            for file_ in hf_weight_files:
                f = safe_open(os.path.join(weight_dir, file_), "pt", "cpu")
                for k in f.keys():
                    if "vision_model" in k:
                        weight_dict[k[len("vision_model.embeddings.") :]] = f.get_tensor(k)
        self.load_state_dict(weight_dict)

    def precompute_rope_freqs_sincos(self):
        inv_freq = 1.0 / (
            self.rope_theta_vision ** (torch.arange(0, self.rope_dim_part, 2).float() / self.rope_dim_part)
        )
        t = torch.arange(self.max_position_embeddings_vision).type_as(inv_freq)
        freqs = torch.outer(t, inv_freq)
        return torch.cos(freqs), torch.sin(freqs)

    def _apply_2d_rotary_pos_emb(self, patch_embeds, grid_hw):
        """
        Apply 2D Rotary Position Embedding to the patch embeddings.
        """
        abs_pos_x, abs_pos_y = build_abs_positions_from_grid_hw(grid_hw, device=patch_embeds.device)
        embeddings = apply_2d_rotary_pos_emb(
            patch_embeds.to(torch.float32),  # RoPE calculations are often more stable in float32
            self.cos_x,
            self.sin_x,
            self.cos_y,
            self.sin_y,
            abs_pos_x,
            abs_pos_y,
        ).to(self.patch_embedding.weight.dtype)
        return embeddings

    def forward(self, pixel_values: torch.Tensor, grid_hw: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.view(
            -1,
            3,
            self.patch_size,
            self.patch_size,
        )
        patch_embeds = self.gelu(self.patch_embedding(pixel_values)).view(-1, self.embed_dim)
        patch_embeds = self._apply_2d_rotary_pos_emb(patch_embeds, grid_hw)
        assert (grid_hw[:, 0] * grid_hw[:, 1]).sum() == patch_embeds.shape[
            0
        ], "Grid size and patch embeds size mismatch."

        patches_list = []
        cur_position = 0
        for i in range(grid_hw.shape[0]):
            h, w = grid_hw[i]
            patches_per_img = patch_embeds[cur_position : cur_position + h * w].view(h, w, -1).unsqueeze(0)
            patches_per_img = self.dense_embedding(patches_per_img.permute(0, 3, 1, 2))
            patches_per_img = patches_per_img.permute(0, 2, 3, 1)
            patches_list.append(patches_per_img.view(-1, patches_per_img.shape[-1]))
            cur_position += h * w

        embeddings = torch.cat(patches_list, dim=0)  # (N_total // downsample_factor**2, C)
        assert cur_position == patch_embeds.shape[0]
        assert embeddings.shape[0] == int(patch_embeds.shape[0] / self.downsample_factor ** 2)

        return embeddings

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
                pixel_values, image_grid_hw = load_image_native(
                    image_data,
                    patch_size=self.patch_size,
                    downsample_ratio=self.downsample_ratio,
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels,
                )
                img_tensors.append(pixel_values)
                img_grids.append(image_grid_hw)
            else:
                raise Exception("Unsupport input types: {} for {}".format(type(img), img))

            # must devide merge_length
            cur_num = int(img_tensors[-1].shape[0] * (self.downsample_ratio ** 2))
            print(f"cur_num is {cur_num}")
            valid_ids.append([valid_id, valid_id + cur_num])
            valid_id += cur_num

        if len(img_tensors) <= 0:
            return None

        imgs = torch.cat(img_tensors, dim=0)
        grid_hw = torch.cat(img_grids, dim=0)

        pixel_values = imgs.to("cuda", dtype=self.data_type, non_blocking=True)
        image_grid_hw = grid_hw.to("cuda", non_blocking=True)

        all_img_embeds = self.forward(pixel_values, grid_hw=image_grid_hw)

        return all_img_embeds, uuids, valid_ids
