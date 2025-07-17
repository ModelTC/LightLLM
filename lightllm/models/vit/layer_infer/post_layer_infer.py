import torch
import torch.functional as F
import torch.distributed as dist
from lightllm.models.vit.layer_weights.pre_and_post_layer_weight import ViTPreAndPostLayerWeight
from lightllm.utils.dist_utils import get_current_rank_in_dp, get_dp_world_size
from lightllm.models.vit.triton_kernel.gelu_vit import gelu_fwd


class ViTPostLayerInfer:
    """ """

    def __init__(self, network_config, mode):
        self.tp_rank_ = get_current_rank_in_dp()
        self.tp_world_size_ = get_dp_world_size()
        self.network_config_ = network_config
        self.mode = mode
        self.llm_hidden_size = network_config["llm_hidden_size"]
        self.downsample_ratio = network_config["downsample_ratio"]
        return

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor), int(c / (scale_factor * scale_factor)))
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def pixel_shuffle_from_cu_seqlens(self, x, cu_seqlens, grid_hw):
        """
        Args:
            x: (N_total, C)
            cu_seqlens: (B + 1,)
            grid_hw: (B, 2), each row is (Hᵢ, Wᵢ)
        """
        _, C = x.shape
        patches_list = []
        for i in range(len(cu_seqlens) - 1):
            start_idx, end_idx = cu_seqlens[i], cu_seqlens[i + 1]
            h, w = grid_hw[i]
            patches_per_img = x[start_idx:end_idx].view(h, w, -1)
            patches_per_img = patches_per_img.unsqueeze(0)  # (1, Hᵢ, Wᵢ, C)
            patches_per_img = self.pixel_shuffle(patches_per_img, scale_factor=self.downsample_ratio)
            patches_per_img = patches_per_img.view(-1, patches_per_img.shape[-1])
            patches_list.append(patches_per_img)

        x = torch.cat(patches_list, dim=0)  # (N_total, C)
        assert x.shape[-1] == int(
            C / self.downsample_ratio ** 2
        ), f"Expected {int(C / self.downsample_ratio**2)} channels, but got {x.shape[-1]} channels after pixel shuffle."
        print(f"x.shape is {x.shape}")
        return x

    def forward(self, vit_embeds, layer_weight: ViTPreAndPostLayerWeight, cu_seqlens, grid_hw):
        if grid_hw is not None:
            vit_embeds = self.pixel_shuffle_from_cu_seqlens(vit_embeds, cu_seqlens, grid_hw)
        else:
            batch_size = vit_embeds.shape[0]
            h = w = int(vit_embeds.shape[1] ** 0.5)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
            vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds_norm = torch.nn.functional.layer_norm(
            vit_embeds,
            (vit_embeds.shape[-1],),
            weight=layer_weight.layernorm_weight_,
            bias=layer_weight.layernorm_bias_,
        )

        vit_embeds_1 = torch.addmm(
            layer_weight.mlp1_1_bias_, vit_embeds_norm.view(-1, vit_embeds_norm.shape[-1]), layer_weight.mlp1_1_weight_
        )

        vit_embeds_gelu = gelu_fwd(vit_embeds_1, use_custom_tensor_mananger=True)

        vit_embeds_out = torch.addmm(
            layer_weight.mlp1_3_bias_,
            vit_embeds_gelu.view(-1, self.llm_hidden_size // self.tp_world_size_),
            layer_weight.mlp1_3_weight_,
            beta=1.0 / self.tp_world_size_,
        )

        if self.tp_world_size_ == 1:
            if grid_hw is not None:
                return vit_embeds_out.view(-1, self.llm_hidden_size)
            else:
                return vit_embeds_out.view(batch_size, -1, self.llm_hidden_size)

        dist.all_reduce(vit_embeds_out, op=dist.ReduceOp.SUM, async_op=False)
        return vit_embeds_out.view(batch_size, -1, self.llm_hidden_size)
