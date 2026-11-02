import torch

from lightllm.models.draft_registry import DraftModelRegistry
from lightllm.models.qwen3_5_mtp.model import Qwen3_5MTPModel
from ..model import Qwen4ExpTpPartModel
from ..infer_struct import Qwen4ExpInferStateInfo
from ..layer_infer.post_layer_infer import Qwen4ExpPostLayerInfer
from .weights import Qwen4ExpMTPPreAndPostLayerWeight, Qwen4ExpMTPTransformerLayerWeight
from .infer import Qwen4ExpMTPPreLayerInfer, Qwen4ExpMTPTransformerLayerInfer


class Qwen4ExpMTPModel(Qwen3_5MTPModel):
    pre_and_post_weight_class = Qwen4ExpMTPPreAndPostLayerWeight
    transformer_weight_class = Qwen4ExpMTPTransformerLayerWeight
    pre_layer_infer_class = Qwen4ExpMTPPreLayerInfer
    post_layer_infer_class = Qwen4ExpPostLayerInfer
    transformer_layer_infer_class = Qwen4ExpMTPTransformerLayerInfer
    infer_state_class = Qwen4ExpInferStateInfo
    _init_att_backend = Qwen4ExpTpPartModel._init_att_backend
    create_full_att_states = classmethod(Qwen4ExpTpPartModel.create_full_att_states.__func__)

    @classmethod
    def normalize_model_config(cls, model_config):
        config = Qwen4ExpTpPartModel.normalize_model_config(model_config)
        config.update(
            full_attention_interval=1, num_hidden_layers=1, n_layer=1, ple_layer_ids=[], layer_types=["full_attention"]
        )
        return config

    def _init_config(self):
        Qwen4ExpTpPartModel._init_config(self)
        if self.config.get("mtp_num_hidden_layers", 1) != 1:
            raise ValueError("Qwen4Exp recurrent MTP requires one trained draft layer")
        self.config.update(
            full_attention_interval=1, num_hidden_layers=1, n_layer=1, ple_layer_ids=[], layer_types=["full_attention"]
        )

    def _init_weights(self, start_layer_index=None):
        if start_layer_index is not None or self.mtp_previous_draft_models:
            raise ValueError("Qwen4Exp MTP uses one recurrent predictor")
        self.pre_post_weight = self.pre_and_post_weight_class(self.data_type, self.config)
        self.pre_post_weight.wte_weight_ = self.main_model.pre_post_weight.wte_weight_
        self.pre_post_weight.lm_head_weight_ = self.main_model.pre_post_weight.lm_head_weight_
        self.trans_layers_weight = [self.transformer_weight_class(0, self.data_type, self.config, self.quant_cfg)]

    def _gen_special_model_input(self, token_num):
        return {
            "mtp_draft_input_hiddens": torch.randn(
                token_num, self.config["hidden_size"] * self.config["hc_count"], dtype=self.data_type, device="cuda"
            )
        }
