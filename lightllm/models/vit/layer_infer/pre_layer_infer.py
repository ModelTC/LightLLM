import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

from lightllm.models.vit.layer_weights.pre_and_post_layer_weight import ViTPreAndPostLayerWeight
from lightllm.utils.dist_utils import get_current_rank_in_dp, get_dp_world_size


class ViTPreLayerInfer:
    """ """

    def __init__(self, network_config, mode):
        self.tp_rank_ = get_current_rank_in_dp()
        self.tp_world_size_ = get_dp_world_size()
        self.network_config_ = network_config
        self.mode = mode
        self.dynamic_image_size = network_config.get("dynamic_image_size", False)
        print(f"dynamic is {self.dynamic_image_size}")
        if self.dynamic_image_size:
            self.max_pixels = network_config.get("max_pixels", 8192 * 28 * 28)
            self.patch_size = network_config.get("patch_size")
            self.embed_dim = network_config.get("hidden_size")
            ROPE_MAX_COORD_DEFAULT = self.max_pixels / self.patch_size ** 2
            ROPE_BASE_DEFAULT = 10000.0
            self.rope_dim_part = self.embed_dim // 2
            self.cos_x, self.sin_x = self.precompute_rope_freqs_sincos(
                self.rope_dim_part, ROPE_MAX_COORD_DEFAULT, base=ROPE_BASE_DEFAULT
            )
            self.cos_y, self.sin_y = self.precompute_rope_freqs_sincos(
                self.rope_dim_part, ROPE_MAX_COORD_DEFAULT, base=ROPE_BASE_DEFAULT
            )

        return

    def precompute_rope_freqs_sincos(self, dim: int, max_position: int, base: float = 10000.0, device=None):
        """预计算 RoPE 的 cos 和 sin 值 (1D)。"""
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        t = torch.arange(max_position, device=device).type_as(inv_freq)
        freqs = torch.outer(t, inv_freq)
        return torch.cos(freqs), torch.sin(freqs)

    def apply_rotary_emb_1d(
        self,
        x: torch.Tensor,
        cos_cached: torch.Tensor,
        sin_cached: torch.Tensor,
        positions: torch.Tensor,
    ):
        """对输入张量的一部分应用1D RoPE。"""
        # x: (..., seq_len, dim_part)
        # positions: (..., seq_len)
        # cos_cached: (max_pos, dim_part / 2)

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
        self,
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
        rotated_part_1 = self.apply_rotary_emb_1d(x_part_1, cos_cached_x, sin_cached_x, abs_positions_x)
        # 将与 abs_positions_y 相关的旋转应用于 x_part_2
        rotated_part_2 = self.apply_rotary_emb_1d(x_part_2, cos_cached_y, sin_cached_y, abs_positions_y)

        # 将它们重新拼接起来。确保顺序与你分割时一致。
        return torch.cat((rotated_part_1, rotated_part_2), dim=-1)

    def _apply_2d_rotary_pos_emb(self, patch_embeds, grid_hw):
        """
        Apply 2D Rotary Position Embedding to the patch embeddings.
        """
        abs_pos_x, abs_pos_y = self.build_abs_positions_from_grid_hw(grid_hw, device=patch_embeds.device)
        embeddings = self.apply_2d_rotary_pos_emb(
            patch_embeds.to(torch.float32),  # RoPE calculations are often more stable in float32
            self.cos_cached_x,
            self.sin_cached_x,
            self.cos_cached_y,
            self.sin_cached_y,
            abs_pos_x,
            abs_pos_y,
        )
        return embeddings

    def build_abs_positions_from_grid_hw(self, grid_hw: torch.Tensor, device=None):
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

    def forward(self, pixel_values, layer_weight: ViTPreAndPostLayerWeight, grid_hw=None):
        target_dtype = layer_weight.patch_embedding_weight_.dtype
        target_device = layer_weight.patch_embedding_weight_.device
        if self.dynamic_image_size:
            assert pixel_values.dim() == 2, f"pixel_values must be 2D for native resolution, got: {pixel_values.dim()}"
            pixel_values = pixel_values.view(
                -1,
                3,
                self.patch_size,
                self.patch_size,
            )
            patch_embeds = (
                F.conv2d(
                    pixel_values,
                    weight=layer_weight.patch_embedding_weight_,
                    bias=layer_weight.patch_embedding_bias_,
                    stride=self.patch_size,
                )
                .view(-1, self.embed_dim)
                .to(
                    device=target_device,
                    dtype=target_dtype,
                )
            )
            self.cos_cached_x = self.cos_x.to(target_device)
            self.sin_cached_x = self.sin_x.to(target_device)
            self.cos_cached_y = self.cos_y.to(target_device)
            self.sin_cached_y = self.sin_y.to(target_device)
            embeddings = self._apply_2d_rotary_pos_emb(patch_embeds, grid_hw).to(target_dtype)
        else:

            patch_embeds = F.conv2d(
                pixel_values,
                weight=layer_weight.patch_embedding_weight_,
                bias=layer_weight.patch_embedding_bias_,
                stride=layer_weight.patch_size,
            )
            batch_size, _, height, width = patch_embeds.shape
            patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
            class_embeds = layer_weight.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
            embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
            position_embedding = torch.cat(
                [layer_weight.position_embedding[:, :1, :], layer_weight._get_pos_embed(height, width)], dim=1
            )
            embeddings = embeddings + position_embedding.to(target_dtype)

        if self.tp_world_size_ == 1:
            return embeddings
        gather_embedding = torch.empty(
            (embeddings.shape[2] * self.tp_world_size_, batch_size, embeddings.shape[1]),
            device=embeddings.device,
            dtype=target_dtype,
        )
        split_indexes = np.linspace(0, layer_weight.embed_dim, self.tp_world_size_ + 1, dtype=np.int64)
        dist.all_gather(
            [gather_embedding[split_indexes[i] : split_indexes[i + 1], :, :] for i in range(self.tp_world_size_)],
            embeddings.permute(2, 0, 1).contiguous(),
            group=None,
            async_op=False,
        )
        return gather_embedding.permute(1, 2, 0).contiguous()
