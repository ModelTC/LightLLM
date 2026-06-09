from lightllm.models.base_mtp_model import BaseMTPModel
from lightllm.models.deepseek_mtp.layer_infer.pre_layer_infer import Deepseek3MTPPreLayerInfer
from lightllm.models.glm4_moe_lite.model import Glm4MoeLiteTpPartModel
from lightllm.models.glm4_moe_lite_mtp.layer_weights.pre_and_post_layer_weight import (
    Glm4MoeLiteMTPPreAndPostLayerWeight,
)
from lightllm.common.basemodel.basemodel import load_hf_weights


class Glm4MoeLiteMTPModel(BaseMTPModel, Glm4MoeLiteTpPartModel):

    pre_and_post_weight_class = Glm4MoeLiteMTPPreAndPostLayerWeight
    pre_layer_infer_class = Deepseek3MTPPreLayerInfer

    def _init_weights(self, start_layer_index=None):
        assert start_layer_index is None

        mtp_layer_start = self.config["num_hidden_layers"]
        num_mtp_layers = self.config.get("num_nextn_predict_layers", 1)

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
            for i in range(mtp_layer_start, mtp_layer_start + num_mtp_layers)
        ]

        load_hf_weights(
            self.data_type,
            weight_dir=self.weight_dir_,
            pre_post_layer=self.pre_post_weight,
            transformer_layer_list=self.trans_layers_weight,
            weight_dict=self.weight_dict,
        )

        self.pre_post_weight.verify_load()
        [weight.verify_load() for weight in self.trans_layers_weight]

        self.pre_post_weight.wte_weight_ = self.main_model.pre_post_weight.wte_weight_
        self.pre_post_weight.lm_head_weight_ = self.main_model.pre_post_weight.lm_head_weight_

    def _init_infer_layer(self, start_layer_index=None):
        assert start_layer_index is None

        self.pre_infer = self.pre_layer_infer_class(network_config=self.config)
        self.post_infer = self.post_layer_infer_class(network_config=self.config)

        total_pre_layers_num = len(self.main_model.layers_infer)
        total_pre_layers_num += sum(
            [len(previous_model.layers_infer) for previous_model in self.mtp_previous_draft_models]
        )

        num_mtp_layers = self.config.get("num_nextn_predict_layers", 1)
        self.layers_infer = [
            self.transformer_layer_infer_class(i, network_config=self.config)
            for i in range(total_pre_layers_num, total_pre_layers_num + num_mtp_layers)
        ]

    def _init_some_value(self):
        super()._init_some_value()
        self.layers_num = self.config.get("num_nextn_predict_layers", 1)

    def autotune_layers(self):
        return self.config.get("num_nextn_predict_layers", 1)
