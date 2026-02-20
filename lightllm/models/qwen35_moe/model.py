import os
import json

from lightllm.models.qwen3_vl.model import QWen3VLTokenizer
from lightllm.models.registry import ModelRegistry
from lightllm.models.qwen3next.model import Qwen3NextTpPartModel
from lightllm.common.build_utils import repair_config
from lightllm.common.basemodel.layer_weights.hf_load_utils import load_hf_weights


class QWen35Tokenizer(QWen3VLTokenizer):
    def __init__(self, tokenizer=None, image_processor=None, **kwargs):
        super().__init__(tokenizer, image_processor, **kwargs)


@ModelRegistry(["qwen3_5"], is_multimodal=True)
class Qwen35MoeTpPartModel(Qwen3NextTpPartModel):
    def _init_config(self):
        with open(os.path.join(self.weight_dir_, "config.json"), "r") as json_file:
            all_config = json.load(json_file)
            self.config = all_config["text_config"]

        repair_config(self.config, same_names=["num_attention_heads", "n_head"])
        repair_config(self.config, same_names=["hidden_size", "n_embd", "n_embed"])
        repair_config(self.config, same_names=["num_hidden_layers", "n_layer"])
        repair_config(self.config, same_names=["intermediate_size", "moe_intermediate_size"])

        # Handle fine-tuning config if present
        if self.finetune_config:
            self.config["vocab_size"] = self.finetune_config.vocab_size

    def _load_hf_weights(self):
        load_hf_weights(
            self.data_type,
            weight_dir=self.weight_dir_,
            pre_post_layer=self.pre_post_weight,
            transformer_layer_list=self.trans_layers_weight,
            weight_dict=self.weight_dict,
        )
        self.pre_post_weight.verify_load()
        [weight.verify_load() for weight in self.trans_layers_weight]
        return
