from typing import List

from lightllm.common.basemodel import TpPartBaseModel
from lightllm.models.glm5_next_mtp.layer_infer.pre_layer_infer import (
    Glm5NextMTPPreLayerInfer,
)
from lightllm.models.glm5_next_mtp.layer_infer.post_layer_infer import (
    Glm5NextMTPPostLayerInfer,
)
from lightllm.models.draft_registry import DraftModelRegistry
from lightllm.models.glm5_next_mtp.layer_weights.pre_and_post_layer_weight import (
    Glm5NextMTPPreAndPostLayerWeight,
)
from lightllm.models.glm5_next.layer_infer.transformer_layer_infer import (
    Glm5NextTransformerLayerInfer,
)
from lightllm.models.glm5_next.layer_weights.transformer_layer_weight import (
    Glm5NextTransformerLayerWeight,
)
from lightllm.models.glm5_next.model import Glm5NextTpPartModel


@DraftModelRegistry(
    model_type=("glm5_next", "glm5_next_text"),
    spec_modes=("vanilla_with_att", "eagle_with_att"),
)
class Glm5NextMTPModel(Glm5NextTpPartModel):
    """GLM-5.3's shared one-layer NextN draft model."""

    is_mtp_draft_model = True
    replicated_attention_ep = False
    pre_and_post_weight_class = Glm5NextMTPPreAndPostLayerWeight
    pre_layer_infer_class = Glm5NextMTPPreLayerInfer
    post_layer_infer_class = Glm5NextMTPPostLayerInfer
    transformer_weight_class = Glm5NextTransformerLayerWeight
    transformer_layer_infer_class = Glm5NextTransformerLayerInfer

    def __init__(self, kvargs: dict):
        self.main_model: TpPartBaseModel = kvargs.pop("main_model")
        self.mtp_previous_draft_models: List[TpPartBaseModel] = kvargs.pop("mtp_previous_draft_models")
        super().__init__(kvargs)

    def _init_custom(self):
        self._cos_cached = self.main_model._cos_cached
        self._sin_cached = self.main_model._sin_cached

    def _init_req_manager(self):
        self.req_manager = self.main_model.req_manager

    def _init_mem_manager(self):
        self.mem_manager = self.main_model.mem_manager

    def _init_weights(self, start_layer_index=None):
        assert start_layer_index is None
        mtp_layer = self.config["num_hidden_layers"]
        self.pre_post_weight = self.pre_and_post_weight_class(
            self.data_type, network_config=self.config, quant_cfg=self.quant_cfg
        )
        self.trans_layers_weight = [
            self.transformer_weight_class(
                mtp_layer,
                self.data_type,
                network_config=self.config,
                quant_cfg=self.quant_cfg,
            )
        ]
        self.pre_post_weight.wte_weight_ = self.main_model.pre_post_weight.wte_weight_
        self.pre_post_weight.lm_head_weight_ = self.main_model.pre_post_weight.lm_head_weight_

    def _init_infer_layer(self, start_layer_index=None):
        assert start_layer_index is None
        self.pre_infer = self.pre_layer_infer_class(network_config=self.config)
        self.post_infer = self.post_layer_infer_class(network_config=self.config)
        logical_layer = len(self.main_model.layers_infer) + sum(
            len(model.layers_infer) for model in self.mtp_previous_draft_models
        )
        self.layers_infer = [self.transformer_layer_infer_class(logical_layer, network_config=self.config)]

    def _init_some_value(self):
        super()._init_some_value()
        self.layers_num = 1

    def autotune_layers(self):
        return 1
