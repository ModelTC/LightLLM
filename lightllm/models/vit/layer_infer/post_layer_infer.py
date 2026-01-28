import torch
import torch.functional as F
import torch.distributed as dist
from lightllm.models.vit.layer_weights.pre_and_post_layer_weight import ViTPreAndPostLayerWeight
from lightllm.utils.dist_utils import get_current_rank_in_dp, get_dp_world_size
from lightllm.models.vit.triton_kernel.gelu_vit import gelu_fwd


class ViTPostLayerInfer:
    """ """

    def __init__(self, network_config):
        self.tp_rank_ = get_current_rank_in_dp()
        self.tp_world_size_ = get_dp_world_size()
        self.network_config_ = network_config
        self.llm_hidden_size = network_config["llm_hidden_size"]
        self.downsample_ratio = network_config["downsample_ratio"]
        self.eps_ = network_config["layer_norm_eps"]
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

    def forward(self, vit_embeds, layer_weight: ViTPreAndPostLayerWeight):
        batch_size = vit_embeds.shape[0]
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds_norm = layer_weight.layernorm_weight_(input=vit_embeds, eps=self.eps_)
        vit_embeds_1 = layer_weight.mlp1_1_.mm(vit_embeds_norm.view(-1, vit_embeds_norm.shape[-1]))

        vit_embeds_gelu = gelu_fwd(vit_embeds_1, use_custom_tensor_mananger=True)

        vit_embeds_out = layer_weight.mlp1_3_.mm(vit_embeds_gelu.view(-1, self.llm_hidden_size // self.tp_world_size_))

        if self.tp_world_size_ == 1:
            return vit_embeds_out.view(batch_size, -1, self.llm_hidden_size)

        dist.all_reduce(vit_embeds_out, op=dist.ReduceOp.SUM, async_op=False)
        return vit_embeds_out.view(batch_size, -1, self.llm_hidden_size)
