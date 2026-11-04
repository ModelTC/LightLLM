from lightllm.models.deepseek_mtp.adapter import Deepseek3MTPModelAdapter
from typing import List
from lightllm.models.deepseek2.model import Deepseek2TpPartModel
from lightllm.models.deepseek2.adapter import Deepseek2ModelAdapter
from lightllm.models.deepseek_mtp.layer_infer.pre_layer_infer import Deepseek3MTPPreLayerInfer
from lightllm.models.deepseek_mtp.layer_weights.pre_and_post_layer_weight import Deepseek3MTPPreAndPostLayerWeight
from lightllm.common.basemodel import TpPartBaseModel
from lightllm.models.base.adapter import grouped_moe_specs


class Deepseek3MTPModel(Deepseek2TpPartModel):
    model_adapter_class = Deepseek3MTPModelAdapter

    # MTP draft model marker (consumed by the decode CUDA-graph / padding paths).
    is_mtp_draft_model = True

    pre_and_post_weight_class = Deepseek3MTPPreAndPostLayerWeight
    pre_layer_infer_class = Deepseek3MTPPreLayerInfer

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
        start = len(self.main_model.layers_infer) + sum(len(m.layers_infer) for m in self.mtp_previous_draft_models)
        depth = self.config["num_hidden_layers"]
        checkpoint_start = self.config.get("mtp_checkpoint_layer_start", 0)
        weights = self.main_model.pre_post_weight
        return adapter_class(
            self.config,
            role=self.att_state_role,
            dtype=self.data_type,
            tp_world_size=self.tp_world_size_,
            layer_ids=range(start, start + depth),
            checkpoint_layer_ids=range(checkpoint_start, checkpoint_start + depth),
            shared_weights=(weights.wte_weight_, weights.lm_head_weight_),
        )

    def _init_weights(self, start_layer_index=None):
        assert start_layer_index is None
        if getattr(self, "model_adapter", None) is not None:
            return TpPartBaseModel._init_weights(self)
        self.pre_post_weight = self.pre_and_post_weight_class(
            self.data_type, network_config=self.config, quant_cfg=self.quant_cfg
        )
        self.trans_layers_weight = [
            self.transformer_weight_class(
                i,
                self.data_type,
                network_config=self.config,
                quant_cfg=self.quant_cfg,
            )
            for i in range(0, self.config["n_layer"])
        ]
        self.pre_post_weight.wte_weight_ = self.main_model.pre_post_weight.wte_weight_
        self.pre_post_weight.lm_head_weight_ = self.main_model.pre_post_weight.lm_head_weight_
        return

    def _init_infer_layer(self, start_layer_index=None):
        assert start_layer_index is None
        if getattr(self, "model_adapter", None) is not None:
            return TpPartBaseModel._init_infer_layer(self)
        total_pre_layers_num = len(self.main_model.layers_infer)
        total_pre_layers_num += sum(
            [len(previous_model.layers_infer) for previous_model in self.mtp_previous_draft_models]
        )
        super()._init_infer_layer(start_layer_index=total_pre_layers_num)
        return
