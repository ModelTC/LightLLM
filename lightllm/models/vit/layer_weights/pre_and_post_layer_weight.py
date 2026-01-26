import os
import torch
import numpy as np
import torch.nn.functional as F
from lightllm.common.basemodel import PreAndPostLayerWeight
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.common.basemodel.layer_weights.meta_weights import LayerNormWeight, COLMMWeight, ROWMMWeight


class ViTPreAndPostLayerWeight(PreAndPostLayerWeight):
    def __init__(self, data_type, network_config):
        super().__init__(data_type, network_config)
        self.embed_dim = self.network_config_["hidden_size"]
        self.image_size = self.network_config_["image_size"]
        self.patch_size = self.network_config_["patch_size"]
        self.llm_hidden_size = self.network_config_["llm_hidden_size"]
        self.downsample_ratio = self.network_config_["downsample_ratio"]
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self._create_weight()
        return

    def _create_weight(self):
        split_indexes = np.linspace(0, self.embed_dim, self.tp_world_size_ + 1, dtype=np.int64)
        split_start = split_indexes[self.tp_rank_]
        split_end = split_indexes[self.tp_rank_ + 1]
        split_embed_dim = split_end - split_start

        # Pre-allocate memory for vision model weights
        self.class_embedding = torch.empty((1, 1, split_embed_dim), dtype=self.data_type_).cuda()
        self.position_embedding = torch.empty((1, self.num_positions, split_embed_dim), dtype=self.data_type_).cuda()
        self.patch_embedding_weight_ = torch.empty(
            (split_embed_dim, 3, self.patch_size, self.patch_size), dtype=self.data_type_
        ).cuda()
        self.patch_embedding_bias_ = torch.empty(split_embed_dim, dtype=self.data_type_).cuda()

        self.layernorm_weight_ = LayerNormWeight(
            dim=self.embed_dim * int(1 / self.downsample_ratio) ** 2,
            weight_name="mlp1.0.weight",
            data_type=self.data_type_,
            bias_name="mlp1.0.bias",
        )
        self.mlp1_1_ = ROWMMWeight(
            in_dim=self.embed_dim * int(1 / self.downsample_ratio) ** 2,
            out_dims=[self.llm_hidden_size],
            weight_names=["mlp1.1.weight"],
            data_type=self.data_type_,
            bias_names=["mlp1.1.bias"],
        )
        self.mlp1_3_ = COLMMWeight(
            in_dim=self.llm_hidden_size,
            out_dims=[self.llm_hidden_size],
            weight_names=["mlp1.3.weight"],
            data_type=self.data_type_,
            bias_names=["mlp1.3.bias"],
        )
        return

    def _cuda(self, cpu_tensor):
        device_id = get_current_device_id()
        return cpu_tensor.contiguous().to(self.data_type_).cuda(device_id)

    def _get_pos_embed(self, H, W):
        pos_embed = self.position_embedding[:, 1:, :]
        target_dtype = pos_embed.dtype
        pos_embed = (
            pos_embed.float()
            .reshape(1, self.image_size // self.patch_size, self.image_size // self.patch_size, -1)
            .permute(0, 3, 1, 2)
        )
        pos_embed = (
            F.interpolate(pos_embed, size=(H, W), mode="bicubic", align_corners=False)
            .reshape(1, -1, H * W)
            .permute(0, 2, 1)
            .to(target_dtype)
        )
        return pos_embed

    def load_hf_weights(self, weights):
        super().load_hf_weights(weights)
        split_indexes = np.linspace(0, self.embed_dim, self.tp_world_size_ + 1, dtype=np.int64)
        split_start = split_indexes[self.tp_rank_]
        split_end = split_indexes[self.tp_rank_ + 1]
        if "vision_model.embeddings.class_embedding" in weights:
            self.class_embedding.copy_(weights["vision_model.embeddings.class_embedding"][:, :, split_start:split_end])
        if "vision_model.embeddings.position_embedding" in weights:
            self.position_embedding.copy_(
                weights["vision_model.embeddings.position_embedding"][:, :, split_start:split_end]
            )
        if "vision_model.embeddings.patch_embedding.weight" in weights:
            self.patch_embedding_weight_.copy_(
                weights["vision_model.embeddings.patch_embedding.weight"][split_start:split_end, :, :, :]
            )
        if "vision_model.embeddings.patch_embedding.bias" in weights:
            self.patch_embedding_bias_.copy_(
                weights["vision_model.embeddings.patch_embedding.bias"][split_start:split_end]
            )
        return

    def verify_load(self):
        errors = "weights load not ok"
        weights = [
            self.class_embedding,
            self.position_embedding,
            self.patch_embedding_weight_,
            self.patch_embedding_bias_,
        ]
        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors
        return
