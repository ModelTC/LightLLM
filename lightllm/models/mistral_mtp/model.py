from lightllm.models.base_mtp_model import BaseMTPModel
from lightllm.models.mistral.model import MistralTpPartModel
from lightllm.models.mistral_mtp.layer_weights.pre_and_post_layer_weight import MistralMTPPreAndPostLayerWeight
from lightllm.models.mistral_mtp.layer_infer.pre_layer_infer import MistralMTPPreLayerInfer
from lightllm.models.mistral_mtp.layer_infer.post_layer_infer import MistralMTPPostLayerInfer
from lightllm.models.mistral_mtp.layer_infer.transformer_layer_infer import MistralMTPTransformerLayerInfer
from lightllm.models.mistral_mtp.layer_weights.transformer_layer_weight import MistralMTPTransformerLayerWeight


class MistralMTPModel(BaseMTPModel, MistralTpPartModel):

    pre_and_post_weight_class = MistralMTPPreAndPostLayerWeight
    pre_layer_infer_class = MistralMTPPreLayerInfer

    transformer_weight_class = MistralMTPTransformerLayerWeight
    transformer_layer_infer_class = MistralMTPTransformerLayerInfer

    post_layer_infer_class = MistralMTPPostLayerInfer

    def _init_some_value(self):
        super()._init_some_value()
        self.layers_num = 1
        return

    def _init_weights(self, start_layer_index=None):
        assert start_layer_index is None
        self.config["n_layer"] = 1
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
        self.pre_post_weight.final_norm_weight_ = self.main_model.pre_post_weight.final_norm_weight_
        return

    def _init_infer_layer(self, start_layer_index=None):
        assert start_layer_index is None
        self.config["n_layer"] = 1
        super()._init_infer_layer(start_layer_index=0)
        return
