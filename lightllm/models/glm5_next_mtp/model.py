from lightllm.models.draft_registry import DraftModelRegistry
from lightllm.models.glm5_next.model import Glm5NextTpPartModel
from .layer_infer.pre_layer_infer import Glm5NextMTPPreLayerInfer
from .layer_weights.pre_and_post_layer_weight import Glm5NextMTPPreAndPostLayerWeight


@DraftModelRegistry(
    model_type=("glm5_next", "glm5_next_text"),
    spec_modes=("vanilla_with_att", "eagle_with_att"),
)
class Glm5NextMTPModel(Glm5NextTpPartModel):
    is_mtp_draft_model = True
    pre_and_post_weight_class = Glm5NextMTPPreAndPostLayerWeight
    pre_layer_infer_class = Glm5NextMTPPreLayerInfer

    def __init__(self, kvargs):
        self.main_model = kvargs.pop("main_model")
        self.mtp_previous_draft_models = kvargs.pop("mtp_previous_draft_models")
        super().__init__(kvargs)

    def _init_config(self):
        super()._init_config()
        assert self.config.get("num_nextn_predict_layers") == 1, "GLM NextN requires one native MTP block"
        self.config["mhc"] = False

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
        super()._init_some_value()
        self.layers_num = 1

    def _init_req_manager(self):
        self.req_manager = self.main_model.req_manager
        self.linear_config = self.main_model.linear_config

    def _init_mem_manager(self):
        self.mem_manager = self.main_model.mem_manager

    def _init_custom(self):
        pass

    def _init_att_backend1(self):
        self.prefill_att_backend1 = self.decode_att_backend1 = None

    def autotune_layers(self):
        return 1
