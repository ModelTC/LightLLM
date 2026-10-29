from lightllm.models.starcoder.layer_infer.transformer_layer_infer import StarcoderTransformerLayerInfer
from lightllm.models.starcoder.layer_infer.pre_layer_infer import StarcoderPreLayerInfer
from lightllm.models.starcoder.layer_weights.transformer_layer_weight import StarcoderTransformerLayerWeight
from lightllm.models.starcoder.layer_weights.pre_and_post_layer_weight import StarcoderPreAndPostLayerWeight
from lightllm.models.bloom.layer_infer.post_layer_infer import BloomPostLayerInfer
from lightllm.common.build_utils import repair_config
from lightllm.common.basemodel import TpPartBaseModel
from lightllm.common.basemodel import InferStateInfo


class StarcoderTpPartModel(TpPartBaseModel):
    # weight class
    pre_and_post_weight_class = StarcoderPreAndPostLayerWeight
    transformer_weight_class = StarcoderTransformerLayerWeight

    # infer class
    pre_layer_infer_class = StarcoderPreLayerInfer
    transformer_layer_infer_class = StarcoderTransformerLayerInfer
    post_layer_infer_class = BloomPostLayerInfer
    infer_state_class = InferStateInfo

    def __init__(self, kvargs):
        super().__init__(kvargs)

    def _init_att_resource_manager(self):
        from lightllm.common.att_resources import AttResourceManager
        from lightllm.common.basemodel.basemodel_utils import init_model_full_attention

        if not self.is_mtp_draft_model:
            self.att_resource_manager = AttResourceManager(request_capacity=self.max_req_num + 1)
        init_model_full_attention(self)

    def _init_config(self):
        super()._init_config()
        # rename key
        # repair_config()
        repair_config(self.config, same_names=["rms_norm_eps", "layer_norm_epsilon"])
        self._reset_num_key_value_heads()
        return

    def _reset_num_key_value_heads(self):
        self.config["num_key_value_heads"] = 1
        return

    def _verify_params(self):
        assert self.load_way == "HF", "StarCoder only support HF format to load Now!"

    @classmethod
    def _full_attention_options(cls, config, args):
        from lightllm.common.basemodel.basemodel_utils import get_full_attention_options

        options = get_full_attention_options(config, args)
        options.update(heads=1, dim=config["hidden_size"] // config["num_attention_heads"], replicated=True)
        return options

    def _init_some_value(self):
        super()._init_some_value()
        self.tp_k_head_num_ = self.config["num_key_value_heads"]
        self.tp_v_head_num_ = self.config["num_key_value_heads"]
        return
