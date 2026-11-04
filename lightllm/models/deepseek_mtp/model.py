from typing import List
from lightllm.models.deepseek2.model import Deepseek2TpPartModel, Deepseek2ModelAdapter
from lightllm.models.deepseek_mtp.layer_infer.pre_layer_infer import Deepseek3MTPPreLayerInfer
from lightllm.models.deepseek_mtp.layer_weights.pre_and_post_layer_weight import Deepseek3MTPPreAndPostLayerWeight
from lightllm.common.basemodel import TpPartBaseModel
from lightllm.common.execution.model_adapter import grouped_moe_specs


class Deepseek3MTPModelAdapter(Deepseek2ModelAdapter):
    moe_group_score_topk = 2
    pre_and_post_weight_class = Deepseek3MTPPreAndPostLayerWeight
    pre_layer_infer_class = Deepseek3MTPPreLayerInfer

    def __init__(self, *args, checkpoint_layer_ids, shared_weights, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_layer_ids = tuple(checkpoint_layer_ids)
        ids = self.checkpoint_layer_ids
        if len(ids) != len(self.layer_ids) or len(set(ids)) != len(ids) or min(ids) < 0:
            raise ValueError("MLA checkpoint layer IDs must uniquely describe every model layer")
        self.shared_weights = shared_weights

    def moe_specs(self):
        specs = grouped_moe_specs(
            self.config, self.dtype, layer_ids=self.checkpoint_layer_ids, group_score_topk=self.moe_group_score_topk
        )
        return {
            layer: specs[checkpoint]
            for layer, checkpoint in zip(self.layer_ids, self.checkpoint_layer_ids, strict=True)
            if checkpoint in specs
        }

    def build_weights(self, quant_config):
        pre_post = self.pre_and_post_weight_class(self.dtype, network_config=self.config, quant_cfg=quant_config)
        pre_post.wte_weight_, pre_post.lm_head_weight_ = self.shared_weights
        layers = [
            self.transformer_weight_class(i, self.dtype, network_config=self.config, quant_cfg=quant_config)
            for i in self.checkpoint_layer_ids
        ]
        return pre_post, layers

    def build_recipe(self):
        pre = self.pre_layer_infer_class(network_config=self.config)
        post = self.post_layer_infer_class(network_config=self.config)
        layers = []
        for layer_id, checkpoint_id in zip(self.layer_ids, self.checkpoint_layer_ids, strict=True):
            # The checkpoint selects dense/MoE semantics; the execution domain selects state and routing slots.
            layer = self.transformer_layer_infer_class(checkpoint_id, network_config=self.config)
            layer.layer_num_ = layer_id
            self.validate_layer(layer)
            layers.append(layer)
        return pre, post, layers


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
