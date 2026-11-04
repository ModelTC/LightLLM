from lightllm.models.qwen3_moe_mtp.adapter import Qwen3MoEMTPModelAdapter
from copy import deepcopy
from typing import List
from lightllm.models.qwen3_moe.model import Qwen3MOEModel
from lightllm.models.qwen3_moe.adapter import Qwen3MoEModelAdapter, qwen3_moe_specs
from lightllm.models.qwen3_moe_mtp.layer_weights.pre_and_post_layer_weight import Qwen3MOEMTPPreAndPostLayerWeight
from lightllm.models.deepseek_mtp.layer_infer.pre_layer_infer import Deepseek3MTPPreLayerInfer
from lightllm.models.qwen3_moe_mtp.layer_infer.transformer_layer_infer import Qwen3MOEMTPTransformerLayerInfer
from lightllm.models.qwen3_moe_mtp.layer_weights.transformer_layer_weight import Qwen3MOEMTPTransformerLayerWeight
from lightllm.common.basemodel import TpPartBaseModel
from lightllm.models.base.adapter import RecipeModelAdapter


class Qwen3MOEMTPModel(Qwen3MOEModel):
    model_adapter_class = Qwen3MoEMTPModelAdapter

    # MTP draft model marker (consumed by the decode CUDA-graph / padding paths).
    is_mtp_draft_model = True

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

    def _create_model_adapter(self, adapter_class):
        checkpoint_start = len(self.mtp_previous_draft_models)
        execution_start = len(self.main_model.layers_infer) + sum(
            len(model.layers_infer) for model in self.mtp_previous_draft_models
        )
        depth = self.config["num_hidden_layers"]
        weights = self.main_model.pre_post_weight
        return adapter_class(
            self.config,
            dtype=self.data_type,
            layer_ids=range(execution_start, execution_start + depth),
            checkpoint_layer_ids=range(checkpoint_start, checkpoint_start + depth),
            shared_weights=(weights.wte_weight_, weights.lm_head_weight_, weights.final_norm_weight_),
        )
