from lightllm.models.registry import ModelRegistry
from lightllm.models.starcoder.layer_infer.transformer_layer_infer import StarcoderTransformerLayerInfer
from lightllm.models.starcoder.layer_infer.pre_layer_infer import StarcoderPreLayerInfer
from lightllm.models.starcoder.layer_weights.transformer_layer_weight import StarcoderTransformerLayerWeight
from lightllm.models.starcoder.layer_weights.pre_and_post_layer_weight import StarcoderPreAndPostLayerWeight
from lightllm.models.bloom.layer_infer.post_layer_infer import BloomPostLayerInfer
from lightllm.common.build_utils import repair_config
from lightllm.common.kv_cache_mem_manager.mem_utils import select_mem_manager_class
from lightllm.utils.envs_utils import get_added_mtp_kv_layer_num
from lightllm.common.basemodel import TpPartBaseModel
from lightllm.common.basemodel import InferStateInfo


@ModelRegistry("gpt_bigcode")
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

    def _init_mem_manager(self):
        self.mem_manager = select_mem_manager_class()(
            self.max_total_token_num,
            dtype=self.data_type,
            head_num=self.config["num_key_value_heads"],
            head_dim=self.config["hidden_size"] // self.config["num_attention_heads"],
            layer_num=self.config["num_hidden_layers"] + get_added_mtp_kv_layer_num(),
            mem_fraction=self.mem_fraction,
        )
        return

    def _init_some_value(self):
        super()._init_some_value()
        self.tp_k_head_num_ = self.config["num_key_value_heads"]
        self.tp_v_head_num_ = self.config["num_key_value_heads"]
        return
