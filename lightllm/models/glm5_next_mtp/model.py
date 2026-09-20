from copy import deepcopy

from lightllm.common.basemodel import TpPartBaseModel
from lightllm.common.basemodel.attention.nsa.glm5_next import Glm5NextSparseAttBackend
from lightllm.models.glm5_next.layer_infer.post_layer_infer import Glm5NextPostLayerInfer
from lightllm.models.glm5_next.layer_infer.transformer_layer_infer import Glm5NextTransformerLayerInfer
from lightllm.models.glm5_next.layer_weights.transformer_layer_weight import Glm5NextTransformerLayerWeight
from .layer_infer.pre_layer_infer import Glm5NextMTPPreLayerInfer
from .layer_weights.pre_and_post_layer_weight import Glm5NextMTPPreAndPostLayerWeight


class Glm5NextMTPModel(TpPartBaseModel):
    is_mtp_draft_model = True
    pre_and_post_weight_class = Glm5NextMTPPreAndPostLayerWeight
    transformer_weight_class = Glm5NextTransformerLayerWeight
    pre_layer_infer_class = Glm5NextMTPPreLayerInfer
    post_layer_infer_class = Glm5NextPostLayerInfer
    transformer_layer_infer_class = Glm5NextTransformerLayerInfer

    def __init__(self, kvargs):
        self.main_model = kvargs.pop("main_model")
        self.mtp_previous_draft_models = kvargs.pop("mtp_previous_draft_models")
        super().__init__(kvargs)

    def _init_config(self):
        # Reuse the validated target configuration without changing its mHC layout.
        self.config = deepcopy(self.main_model.config)
        assert self.config.get("num_nextn_predict_layers") == 1, "GLM NextN requires one native MTP block"
        self.config["mhc"] = False

    def _verify_params(self):
        assert self.load_way in ("HF", "DS"), "GLM-5.3 Flash only supports HF and DS weight loading"

    def _init_weights(self, start_layer_index=None):
        assert start_layer_index is None
        self.pre_post_weight = self.pre_and_post_weight_class(self.data_type, self.config, self.quant_cfg)
        self.pre_post_weight.wte_weight_ = self.main_model.pre_post_weight.wte_weight_
        self.pre_post_weight.lm_head_weight_ = self.main_model.pre_post_weight.lm_head_weight_
        self.trans_layers_weight = [
            self.transformer_weight_class(self.config["num_hidden_layers"], self.data_type, self.config, self.quant_cfg)
        ]

    def _init_infer_layer(self, start_layer_index=None):
        assert start_layer_index is None
        self.pre_infer = self.pre_layer_infer_class(self.config)
        self.post_infer = self.post_layer_infer_class(self.config)
        # Chained modules reuse the native weights but own distinct cache layers.
        layer_index = len(self.main_model.layers_infer) + len(self.mtp_previous_draft_models)
        self.layers_infer = [self.transformer_layer_infer_class(layer_index, self.config)]

    def _init_some_value(self):
        self.layers_num = 1
        self.vocab_size = self.config["vocab_size"]
        self.tp_k_head_num_ = 1
        self.tp_v_head_num_ = 0
        self.qk_nope_head_dim = self.config["qk_nope_head_dim"]
        self.qk_rope_head_dim = self.config["qk_rope_head_dim"]
        self.q_lora_rank = self.config["q_lora_rank"]
        self.kv_lora_rank = self.config["kv_lora_rank"]
        self.v_head_dim = self.config.get("v_head_dim", self.qk_nope_head_dim)
        self.head_dim_ = self.kv_lora_rank + self.qk_rope_head_dim

    def _init_req_manager(self):
        self.req_manager = self.main_model.req_manager
        self.linear_config = self.main_model.linear_config

    def _init_mem_manager(self):
        self.mem_manager = self.main_model.mem_manager

    def _init_att_backend(self):
        self.prefill_att_backend = Glm5NextSparseAttBackend(model=self)
        self.decode_att_backend = self.prefill_att_backend

    def autotune_layers(self):
        return 1
