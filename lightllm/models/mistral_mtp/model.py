from typing import List
from lightllm.models.mistral.model import MistralTpPartModel
from lightllm.models.mistral_mtp.layer_weights.pre_and_post_layer_weight import MistralMTPPreAndPostLayerWeight
from lightllm.models.mistral_mtp.layer_infer.pre_layer_infer import MistralMTPPreLayerInfer
from lightllm.models.mistral_mtp.layer_infer.post_layer_infer import MistralMTPPostLayerInfer
from lightllm.models.mistral_mtp.layer_infer.transformer_layer_infer import MistralMTPTransformerLayerInfer
from lightllm.models.mistral_mtp.layer_weights.transformer_layer_weight import MistralMTPTransformerLayerWeight
from lightllm.common.basemodel import TpPartBaseModel


class MistralMTPModel(MistralTpPartModel):

    pre_and_post_weight_class = MistralMTPPreAndPostLayerWeight
    pre_layer_infer_class = MistralMTPPreLayerInfer

    transformer_weight_class = MistralMTPTransformerLayerWeight
    transformer_layer_infer_class = MistralMTPTransformerLayerInfer

    post_layer_infer_class = MistralMTPPostLayerInfer

    def __init__(self, kvargs: dict):
        self._pre_init(kvargs)
        super().__init__(kvargs)
        return

    def _pre_init(self, kvargs: dict):
        self.main_model: TpPartBaseModel = kvargs.pop("main_model")
        self.mtp_previous_draft_models: List[TpPartBaseModel] = kvargs.pop("mtp_previous_draft_models")
        return

    def _init_some_value(self):
        super()._init_some_value()
        self.layers_num = 1
        return

    def _init_custom(self):
        self._cos_cached = self.main_model._cos_cached
        self._sin_cached = self.main_model._sin_cached
        return

    def _init_req_manager(self):
        self.req_manager = self.main_model.req_manager
        return

    def _init_mem_manager(self):
        self.mem_manager = self.main_model.mem_manager
        return

    def _init_weights(self, start_layer_index=None):
        assert start_layer_index is None

        self.config["n_layer"] = 1
        super()._init_weights(start_layer_index=0)
        self.pre_post_weight.wte_weight_ = self.main_model.pre_post_weight.wte_weight_
        self.pre_post_weight.lm_head_weight_ = self.main_model.pre_post_weight.lm_head_weight_
        self.pre_post_weight.final_norm_weight_ = self.main_model.pre_post_weight.final_norm_weight_
        return

    def _init_infer_layer(self, start_layer_index=None):
        assert start_layer_index is None
        self.config["n_layer"] = 1
        super()._init_infer_layer(start_layer_index=0)
        return
