from lightllm.models.qwen3_moe.model import Qwen3MOEModel
from lightllm.models.qwen3_moe_mtp.layer_weights.pre_and_post_layer_weight import Qwen3MOEMTPPreAndPostLayerWeight
from lightllm.models.deepseek_mtp.layer_infer.pre_layer_infer import Deepseek3MTPPreLayerInfer
from lightllm.models.qwen3_moe_mtp.layer_infer.transformer_layer_infer import Qwen3MOEMTPTransformerLayerInfer
from lightllm.models.qwen3_moe_mtp.layer_weights.transformer_layer_weight import Qwen3MOEMTPTransformerLayerWeight
from lightllm.common.basemodel import TpPartBaseModel
from lightllm.common.basemodel.layer_weights.hf_load_utils import load_hf_weights


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
        self.mem_layer_start = kvargs.pop("mem_layer_start", 0)
        self.mtp_index = kvargs.pop("mtp_index")
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

    def _init_weights(self):
        self.pre_post_weight = self.pre_and_post_weight_class(
            self.data_type, network_config=self.config, mode=self.mode
        )
        self.trans_layers_weight = [
            self.transformer_weight_class(
                i,
                self.data_type,
                network_config=self.config,
                mode=self.mode,
                quant_cfg=self.quant_cfg,
            )
            for i in range(self.mtp_index, self.mtp_index + self.config["n_layer"])
        ]
        if self.load_way == "HF":
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
        self.pre_post_weight.final_norm_weight_ = self.main_model.pre_post_weight.final_norm_weight_
        return

    def _init_infer_layer(self):
        super()._init_infer_layer()
        # reset the layer_num_ of the self.layers_infer
        for layer in self.layers_infer:
            layer.layer_num_ = layer.layer_num_ + self.mem_layer_start - self.mtp_index
        return
