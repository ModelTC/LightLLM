import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

from lightllm.models.vit.layer_weights.pre_and_post_layer_weight import ViTPreAndPostLayerWeight
from lightllm.utils.dist_utils import get_current_rank_in_dp, get_dp_world_size


class ViTPreLayerInfer:
    """ """

    def __init__(self, network_config):
        self.tp_rank_ = get_current_rank_in_dp()
        self.tp_world_size_ = get_dp_world_size()
        self.network_config_ = network_config
        return

    def forward(self, pixel_values, layer_weight: ViTPreAndPostLayerWeight):
        target_dtype = layer_weight.patch_embedding_weight_.dtype
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
