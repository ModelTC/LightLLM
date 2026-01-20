from typing import List
from lightllm.models.glm4_moe.model import Glm4MoeTpPartModel
from lightllm.models.glm4_moe_mtp.layer_weights.pre_and_post_layer_weight import Glm4MoeMTPPreAndPostLayerWeight
from lightllm.models.glm4_moe_mtp.layer_weights.transformer_layer_weight import Glm4MoeMTPTransformerLayerWeight
from lightllm.models.glm4_moe_mtp.layer_infer.transformer_layer_infer import Glm4MoeMTPTransformerLayerInfer
from lightllm.models.deepseek_mtp.layer_infer.pre_layer_infer import Deepseek3MTPPreLayerInfer
from lightllm.common.basemodel import TpPartBaseModel


class Glm4MoeMTPModel(Glm4MoeTpPartModel):
    """
    GLM-4.7 MoE Multi-Token Prediction (MTP) Model.

    This model is used for speculative decoding. It reuses the main model's
    memory manager and request manager, and only runs FFN layers (no attention).
    """

    pre_and_post_weight_class = Glm4MoeMTPPreAndPostLayerWeight
    pre_layer_infer_class = Deepseek3MTPPreLayerInfer

    transformer_weight_class = Glm4MoeMTPTransformerLayerWeight
    transformer_layer_infer_class = Glm4MoeMTPTransformerLayerInfer

    def __init__(self, kvargs: dict):
        self._pre_init(kvargs)
        super().__init__(kvargs)
        return

    def _pre_init(self, kvargs: dict):
        """Pre-initialize by extracting main model and previous draft models."""
        self.main_model: TpPartBaseModel = kvargs.pop("main_model")
        self.mtp_previous_draft_models: List[TpPartBaseModel] = kvargs.pop("mtp_previous_draft_models")
        return

    def _init_custom(self):
        """Initialize by reusing main model's RoPE embeddings."""
        self._cos_cached = self.main_model._cos_cached
        self._sin_cached = self.main_model._sin_cached
        return

    def _init_req_manager(self):
        """Reuse main model's request manager."""
        self.req_manager = self.main_model.req_manager
        return

    def _init_mem_manager(self):
        """Reuse main model's memory manager."""
        self.mem_manager = self.main_model.mem_manager
        return

    def _init_weights(self, start_layer_index=None):
        """Initialize weights starting after previous models' layers."""
        assert start_layer_index is None
        mtp_index = len(self.mtp_previous_draft_models)
        super()._init_weights(start_layer_index=mtp_index)
        # Reuse main model's embedding, lm_head, and final norm weights
        self.pre_post_weight.wte_weight_ = self.main_model.pre_post_weight.wte_weight_
        self.pre_post_weight.lm_head_weight_ = self.main_model.pre_post_weight.lm_head_weight_
        self.pre_post_weight.final_norm_weight_ = self.main_model.pre_post_weight.final_norm_weight_
        return

    def _init_infer_layer(self, start_layer_index=None):
        """Initialize inference layers after previous models' layers."""
        assert start_layer_index is None
        total_pre_layers_num = len(self.main_model.layers_infer)
        total_pre_layers_num += sum(
            [len(previous_model.layers_infer) for previous_model in self.mtp_previous_draft_models]
        )
        super()._init_infer_layer(start_layer_index=total_pre_layers_num)
        return
