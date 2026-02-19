from lightllm.models.qwen3next.model import Qwen3NextTpPartModel
from lightllm.models.qwen3next_mtp.layer_infer.pre_layer_infer import Qwen3NextMTPPreLayerInfer
from lightllm.models.qwen3next_mtp.layer_infer.transformer_layer_infer import Qwen3NextMTPTransformerLayerInfer
from lightllm.models.qwen3next_mtp.layer_weights.pre_and_post_layer_weight import Qwen3NextMTPPreAndPostLayerWeight
from lightllm.models.qwen3next_mtp.layer_weights.transformer_layer_weight import Qwen3NextMTPTransformerLayerWeight
from lightllm.common.basemodel import TpPartBaseModel
from lightllm.common.basemodel.layer_weights.hf_load_utils import load_hf_weights
from lightllm.models.registry import ModelRegistry


@ModelRegistry("qwen3next_mtp")
class Qwen3NextMTPModel(Qwen3NextTpPartModel):

    pre_and_post_weight_class = Qwen3NextMTPPreAndPostLayerWeight
    pre_layer_infer_class = Qwen3NextMTPPreLayerInfer
    transformer_weight_class = Qwen3NextMTPTransformerLayerWeight
    transformer_layer_infer_class = Qwen3NextMTPTransformerLayerInfer

    def __init__(self, kvargs: dict):
        self.mtp_n_layers = 1
        self._pre_init(kvargs)
        super().__init__(kvargs)
        return

    def _pre_init(self, kvargs: dict):
        """Extract main model and memory layer start from kwargs."""
        self.main_model: TpPartBaseModel = kvargs.pop("main_model")
        self.mem_layer_start = kvargs.pop("mem_layer_start")
        return

    def autotune_layers(self):
        return 1

    def _init_some_value(self):
        self.layers_num = self.mtp_n_layers

    def _init_config(self):
        super()._init_config()
        self.config["n_layers"] = self.mtp_n_layers
        self.config["num_hidden_layers"] = self.mtp_n_layers
        return

    def _init_custom(self):
        """Initialize custom components, sharing cos/sin cache with main model."""
        self._cos_cached = self.main_model._cos_cached
        self._sin_cached = self.main_model._sin_cached
        return

    def _init_req_manager(self):
        """Share request manager with main model."""
        self.req_manager = self.main_model.req_manager
        return

    def _init_mem_manager(self):
        """Share memory manager with main model."""
        self.mem_manager = self.main_model.mem_manager
        return

    def _check_mem_size(self):
        """Skip mem size check for MTP models since they share memory with main model."""
        self.max_total_token_num = self.mem_manager.size
        return

    def _init_weights(self):
        self.pre_post_weight = self.pre_and_post_weight_class(self.data_type, network_config=self.config)
        self.trans_layers_weight = [
            self.transformer_weight_class(
                i,
                self.data_type,
                network_config=self.config,
                quant_cfg=self.quant_cfg,
            )
            for i in range(self.mtp_n_layers)
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
        return

    def _init_infer_layer(self):
        self.pre_infer = self.pre_layer_infer_class(network_config=self.config)
        self.post_infer = self.post_layer_infer_class(network_config=self.config)
        self.layers_infer = [
            self.transformer_layer_infer_class(
                i * self.config["full_attention_interval"] - 1,  # Ensure full attention layer
                network_config=self.config,
            )
            for i in range(self.mtp_n_layers)
        ]
        # Ensure full attention layer
        for i, layer in enumerate(self.layers_infer):
            layer.layer_num_ = i + self.mem_layer_start
        return
