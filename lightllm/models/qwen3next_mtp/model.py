from lightllm.models.qwen3next.model import Qwen3NextTpPartModel
from lightllm.models.qwen3next_mtp.layer_infer.pre_layer_infer import Qwen3NextMTPPreLayerInfer
from lightllm.models.qwen3next_mtp.layer_infer.post_layer_infer import Qwen3NextMTPPostLayerInfer
from lightllm.models.qwen3next_mtp.layer_infer.transformer_layer_infer import Qwen3NextMTPTransformerLayerInfer
from lightllm.models.qwen3next_mtp.layer_weights.pre_and_post_layer_weight import Qwen3NextMTPPreAndPostLayerWeight
from lightllm.models.qwen3next_mtp.layer_weights.transformer_layer_weight import Qwen3NextMTPTransformerLayerWeight
from lightllm.common.basemodel import TpPartBaseModel


class Qwen3NextMTPModel(Qwen3NextTpPartModel):

    # weight classes
    pre_and_post_weight_class = Qwen3NextMTPPreAndPostLayerWeight
    transformer_weight_class = Qwen3NextMTPTransformerLayerWeight

    # infer classes
    pre_layer_infer_class = Qwen3NextMTPPreLayerInfer
    post_layer_infer_class = Qwen3NextMTPPostLayerInfer
    transformer_layer_infer_class = Qwen3NextMTPTransformerLayerInfer

    def __init__(self, kvargs: dict):
        self._pre_init(kvargs)
        super().__init__(kvargs)
        return

    def _pre_init(self, kvargs: dict):
        """Extract main model and memory layer start from kwargs."""
        self.main_model: TpPartBaseModel = kvargs.pop("main_model")
        self.mem_layer_start = kvargs.pop("mem_layer_start", 0)
        return

    def _init_config(self):
        """Initialize config, using mtp_num_hidden_layers if available."""
        super()._init_config()
        # Override layer num for MTP
        if "mtp_num_hidden_layers" in self.config:
            self.config["n_layer"] = self.config["mtp_num_hidden_layers"]
        else:
            # Default to 1 MTP layer if not specified
            self.config["n_layer"] = 1
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

    def _init_weights(self):
        """Initialize weights, sharing embedding and lm_head with main model."""
        super()._init_weights()
        # Share embedding weights with main model
        self.pre_post_weight.wte_weight_ = self.main_model.pre_post_weight.wte_weight_
        # Share lm_head weights with main model
        self.pre_post_weight.lm_head_weight_ = self.main_model.pre_post_weight.lm_head_weight_
        return

    def _init_infer_layer(self):
        """Initialize inference layers, adjusting layer numbers for KV cache offset."""
        super()._init_infer_layer()
        # Adjust layer_num_ for KV cache indexing
        # MTP layers' KV cache is stored after main model's layers
        for layer in self.layers_infer:
            layer.layer_num_ = layer.layer_num_ + self.mem_layer_start
        return
