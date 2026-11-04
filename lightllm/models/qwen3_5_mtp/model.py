from typing import List

from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.models.qwen3_5.model import Qwen3_5TpPartModel, Qwen35ModelAdapter
from lightllm.common.execution.model_adapter import DenseModelAdapter
from lightllm.models.qwen3_5.layer_infer.transformer_layer_infer import Qwen35TransformerLayerInfer
from lightllm.models.qwen3_5_mtp.layer_weights.pre_and_post_layer_weight import Qwen3_5MTPPreAndPostLayerWeight
from lightllm.models.qwen3_5_mtp.layer_weights.transformer_layer_weight import Qwen3_5MTPTransformerLayerWeight
from lightllm.models.qwen3_5_mtp.layer_infer.pre_layer_infer import Qwen3_5MTPPreLayerInfer


class Qwen35MTPModelAdapter(DenseModelAdapter):
    pre_and_post_weight_class = Qwen3_5MTPPreAndPostLayerWeight
    transformer_weight_class = Qwen3_5MTPTransformerLayerWeight
    pre_layer_infer_class = Qwen3_5MTPPreLayerInfer
    post_layer_infer_class = Qwen35ModelAdapter.post_layer_infer_class
    transformer_layer_infer_class = Qwen35TransformerLayerInfer

    def __init__(self, config, *, role, dtype, tp_world_size, layer_id, checkpoint_layer_id, shared_weights):
        super().__init__(config, role=role, dtype=dtype, tp_world_size=tp_world_size, layer_ids=(layer_id,))
        self.checkpoint_layer_id = checkpoint_layer_id
        self.shared_weights = shared_weights

    def build_weights(self, quant_config):
        pre_post = self.pre_and_post_weight_class(self.dtype, network_config=self.config, quant_cfg=quant_config)
        pre_post.wte_weight_, pre_post.lm_head_weight_, pre_post.main_norm_weight_ = self.shared_weights
        weight = self.transformer_weight_class(
            self.checkpoint_layer_id, self.dtype, network_config=self.config, quant_cfg=quant_config
        )
        return pre_post, [weight]


class Qwen3_5MTPModel(Qwen3_5TpPartModel):
    model_adapter_class = Qwen35MTPModelAdapter
    pre_and_post_weight_class = Qwen3_5MTPPreAndPostLayerWeight
    pre_layer_infer_class = Qwen3_5MTPPreLayerInfer
    transformer_weight_class = Qwen3_5MTPTransformerLayerWeight
    transformer_layer_infer_class = Qwen35TransformerLayerInfer

    # MTP draft model: reuses the main model's request/attention resources and rope caches, and is
    # marked so the decode CUDA-graph / padding paths detect it (is_mtp_draft_model).
    is_mtp_draft_model = True

    def __init__(self, kvargs: dict):
        self.main_model: TpPartBaseModel = kvargs.pop("main_model")
        self.mtp_previous_draft_models: List[TpPartBaseModel] = kvargs.pop("mtp_previous_draft_models")
        super().__init__(kvargs)
        return

    def _init_custom(self):
        self._cos_cached = self.main_model._cos_cached
        self._sin_cached = self.main_model._sin_cached
        return

    @classmethod
    def normalize_model_config(cls, model_config: dict[str, object]) -> dict[str, object]:
        """原生 MTP 只包含一个 full-attention 层，与 main 共用 checkpoint。"""
        config = super().normalize_model_config(model_config)
        config.update(full_attention_interval=1, num_hidden_layers=1, n_layer=1, layer_types=["full_attention"])
        return config

    def _init_some_value(self):
        super()._init_some_value()
        self.layers_num = 1
        return

    def _create_model_adapter(self, adapter_class):
        main_weights = self.main_model.pre_post_weight
        return adapter_class(
            self.config,
            role=self.att_state_role,
            dtype=self.data_type,
            tp_world_size=self.tp_world_size_,
            layer_id=len(self.main_model.layers_infer)
            + sum(len(m.layers_infer) for m in self.mtp_previous_draft_models),
            checkpoint_layer_id=len(self.mtp_previous_draft_models),
            shared_weights=(main_weights.wte_weight_, main_weights.lm_head_weight_, main_weights.final_norm_weight_),
        )

    def _init_weights(self, start_layer_index=None):
        assert start_layer_index is None
        if getattr(self, "model_adapter", None) is not None:
            return TpPartBaseModel._init_weights(self)
        mtp_index = len(self.mtp_previous_draft_models)
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
            for i in range(mtp_index, mtp_index + self.config["n_layer"])
        ]
        # Shared with the main Qwen3.5 model (mtp_use_dedicated_embeddings: false).
        self.pre_post_weight.wte_weight_ = self.main_model.pre_post_weight.wte_weight_
        self.pre_post_weight.lm_head_weight_ = self.main_model.pre_post_weight.lm_head_weight_
        self.pre_post_weight.main_norm_weight_ = self.main_model.pre_post_weight.final_norm_weight_
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
