from typing import List

from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.models.qwen3_5.model import Qwen3_5TpPartModel
from lightllm.models.qwen3_5.layer_infer.transformer_layer_infer import Qwen35TransformerLayerInfer
from lightllm.models.qwen3_5_mtp.layer_weights.pre_and_post_layer_weight import Qwen3_5MTPPreAndPostLayerWeight
from lightllm.models.qwen3_5_mtp.layer_weights.transformer_layer_weight import Qwen3_5MTPTransformerLayerWeight
from lightllm.models.qwen3_5_mtp.layer_infer.pre_layer_infer import Qwen3_5MTPPreLayerInfer


class Qwen3_5MTPModel(Qwen3_5TpPartModel):
    pre_and_post_weight_class = Qwen3_5MTPPreAndPostLayerWeight
    pre_layer_infer_class = Qwen3_5MTPPreLayerInfer
    transformer_weight_class = Qwen3_5MTPTransformerLayerWeight
    transformer_layer_infer_class = Qwen35TransformerLayerInfer

    # MTP draft model: reuses the main model's req/mem managers and rope caches, and is
    # marked so the decode CUDA-graph / padding paths detect it (is_mtp_draft_model).
    is_mtp_draft_model = True

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

    def _init_config(self):
        super()._init_config()
        self.config["full_attention_interval"] = 1
        self.config["num_hidden_layers"] = 1
        self.config["n_layer"] = 1
        return

    def _init_some_value(self):
        super()._init_some_value()
        self.layers_num = 1
        return

    def _init_weights(self, start_layer_index=None):
        assert start_layer_index is None
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
        return

    def _init_infer_layer(self, start_layer_index=None):
        assert start_layer_index is None
        super()._init_infer_layer(start_layer_index=0)
        draft_idx = len(self.mtp_previous_draft_models)
        draft_kv_slot = self.main_model.mem_manager.model_full_att_layer_num + draft_idx
        self.layers_infer[0].layer_num_ = draft_kv_slot * self.main_model.config["full_attention_interval"]
        return
