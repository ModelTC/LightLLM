import torch
import math

from lightllm.common.mem_utils import select_mem_manager_class
from lightllm.models.qwen2.model import Qwen2TpPartModel
from lightllm.models.qwen2_wquant.layer_infer.transformer_layer_infer import Qwen2TransformerLayerInferWQuant
from lightllm.models.qwen2_wquant.layer_weights.transformer_layer_weight import Qwen2TransformerLayerWeightQuantized


class QWen2TpPartModelWQuant(Qwen2TpPartModel):
    # weight class
    transformer_weight_class = Qwen2TransformerLayerWeightQuantized
    # infer class
    transformer_layer_infer_class = Qwen2TransformerLayerInferWQuant

    def __init__(self, kvargs):
        super().__init__(kvargs)

    def _verify_params(self):
        super()._verify_params()
        assert any(
            "w6a16" in mode_ or "w4a16" in mode_ or "w8a16" in mode_ for mode_ in self.mode
        ), "only for weight quant model"
        assert not any("int8kv" in mode_ or "int4kv" in mode_ for mode_ in self.mode), "not support kv cache quant"
        assert self.data_type == torch.float16, f"not support {self.data_type}, only support torch.float16"
        assert self.config["num_key_value_heads"] % self.world_size_ == 0
        assert self.config["num_attention_heads"] % self.world_size_ == 0
        return

    def _init_mem_manager(self):
        self.mem_manager = select_mem_manager_class(self.mode)(
            self.max_total_token_num,
            dtype=self.data_type,
            head_num=self.config["num_key_value_heads"] // self.world_size_,
            head_dim=self.config["hidden_size"] // self.config["num_attention_heads"],
            layer_num=self.config["num_hidden_layers"],
            always_copy=True,
            mem_fraction=self.mem_fraction,
        )
        return
