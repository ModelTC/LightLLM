import os
import torch
import numpy as np
import torch.nn.functional as F
from lightllm.common.basemodel import PreAndPostLayerWeight
from lightllm.utils.dist_utils import get_current_device_id


class ViTPreAndPostLayerWeight(PreAndPostLayerWeight):
    def __init__(self, data_type, network_config, mode):
        super().__init__(data_type, network_config, mode)
        self.embed_dim = self.network_config_["hidden_size"]
        self.image_size = self.network_config_["image_size"]
        self.patch_size = self.network_config_["patch_size"]
        self.llm_hidden_size = self.network_config_["llm_hidden_size"]
        self._create_weight()
        return

    def _create_weight(self):
        split_indexes = np.linspace(0, self.embed_dim, self.tp_world_size_ + 1, dtype=np.int64)
        split_start = split_indexes[self.tp_rank_]
        split_end = split_indexes[self.tp_rank_ + 1]
        split_embed_dim = split_end - split_start

        # Pre-allocate memory for vision model weights
        self.class_embedding = torch.empty((1, 1, split_embed_dim), dtype=self.data_type_).cuda()
        self.position_embedding = torch.empty(
            (1, 197, split_embed_dim), dtype=self.data_type_
        ).cuda()  # 197 = (224//16)^2 + 1
        self.patch_embedding_weight_ = torch.empty(
            (split_embed_dim, 3, self.patch_size, self.patch_size), dtype=self.data_type_
        ).cuda()
        self.patch_embedding_bias_ = torch.empty(split_embed_dim, dtype=self.data_type_).cuda()

        # Pre-allocate memory for adapter weights
        self.layernorm_weight_ = torch.empty(self.embed_dim, dtype=self.data_type_).cuda()
        self.layernorm_bias_ = torch.empty(self.embed_dim, dtype=self.data_type_).cuda()

        split_indexes_llm = np.linspace(0, self.llm_hidden_size, self.tp_world_size_ + 1, dtype=np.int64)
        split_start_llm = split_indexes_llm[self.tp_rank_]
        split_end_llm = split_indexes_llm[self.tp_rank_ + 1]
        split_llm_hidden_size = split_end_llm - split_start_llm

        self.mlp1_1_weight_ = torch.empty((self.llm_hidden_size, split_llm_hidden_size), dtype=self.data_type_).cuda()
        self.mlp1_1_bias_ = torch.empty(split_llm_hidden_size, dtype=self.data_type_).cuda()
        self.mlp1_3_weight_ = torch.empty((split_llm_hidden_size, self.llm_hidden_size), dtype=self.data_type_).cuda()
        self.mlp1_3_bias_ = torch.empty(self.llm_hidden_size, dtype=self.data_type_).cuda()
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

        if "mlp1.0.weight" in weights:
            self.layernorm_weight_.copy_(weights["mlp1.0.weight"])
        if "mlp1.0.bias" in weights:
            self.layernorm_bias_.copy_(weights["mlp1.0.bias"])

        split_indexes = np.linspace(0, self.llm_hidden_size, self.tp_world_size_ + 1, dtype=np.int64)
        split_start = split_indexes[self.tp_rank_]
        split_end = split_indexes[self.tp_rank_ + 1]

        if "mlp1.1.weight" in weights:
            self.mlp1_1_weight_.copy_(weights["mlp1.1.weight"][split_start:split_end, :].t())
        if "mlp1.1.bias" in weights:
            self.mlp1_1_bias_.copy_(weights["mlp1.1.bias"][split_start:split_end])

        if "mlp1.3.weight" in weights:
            self.mlp1_3_weight_.copy_(weights["mlp1.3.weight"][:, split_start:split_end].t())
        if "mlp1.3.bias" in weights:
            self.mlp1_3_bias_.copy_(weights["mlp1.3.bias"])

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
