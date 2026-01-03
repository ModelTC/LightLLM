from typing import List
from lightllm.models.qwen3_moe.model import Qwen3MOEModel
from lightllm.models.qwen3_moe_mtp.layer_weights.pre_and_post_layer_weight import Qwen3MOEMTPPreAndPostLayerWeight
from lightllm.models.deepseek_mtp.layer_infer.pre_layer_infer import Deepseek3MTPPreLayerInfer
from lightllm.models.qwen3_moe_mtp.layer_infer.transformer_layer_infer import Qwen3MOEMTPTransformerLayerInfer
from lightllm.models.qwen3_moe_mtp.layer_weights.transformer_layer_weight import Qwen3MOEMTPTransformerLayerWeight
from lightllm.common.basemodel import TpPartBaseModel


class Qwen3MOEMTPModel(Qwen3MOEModel):

    pre_and_post_weight_class = Qwen3MOEMTPPreAndPostLayerWeight
    pre_layer_infer_class = Deepseek3MTPPreLayerInfer

    transformer_weight_class = Qwen3MOEMTPTransformerLayerWeight
    transformer_layer_infer_class = Qwen3MOEMTPTransformerLayerInfer

    def __init__(self, kvargs: dict):
        self._pre_init(kvargs)
        super().__init__(kvargs)
        return

    def _pre_init(self, kvargs: dict):
        self.main_model: TpPartBaseModel = kvargs.pop("main_model")
        self.mtp_previous_draft_models: List[TpPartBaseModel] = kvargs.pop("mtp_previous_draft_models")
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
        mtp_index = len(self.mtp_previous_draft_models)
        super()._init_weights(start_layer_index=mtp_index)
        self.pre_post_weight.wte_weight_ = self.main_model.pre_post_weight.wte_weight_
        self.pre_post_weight.lm_head_weight_ = self.main_model.pre_post_weight.lm_head_weight_
        self.pre_post_weight.final_norm_weight_ = self.main_model.pre_post_weight.final_norm_weight_
        return

    def _init_infer_layer(self, start_layer_index=None):
        assert start_layer_index is None
        total_pre_layers_num = len(self.main_model.layers_infer)
        total_pre_layers_num += sum(
            [len(previous_model.layers_infer) for previous_model in self.mtp_previous_draft_models]
        )
        super()._init_infer_layer(start_layer_index=total_pre_layers_num)
        return
