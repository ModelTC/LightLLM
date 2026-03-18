import os
import json
from lightllm.common.build_utils import repair_config
from lightllm.models.registry import ModelRegistry, llm_model_type_is
from lightllm.models.qwen3_vl.infer_struct import Qwen3VLInferStateInfo
from lightllm.models.qwen3_vl.layer_infer.pre_layer_infer import Qwen3VLMultimodalPreLayerInfer
from lightllm.models.qwen3_vl.layer_infer.transformer_layer_infer import Qwen3VLTransformerLayerInfer
from lightllm.models.qwen3_vl.layer_weights.pre_and_post_layer_weight import Qwen3VLPreAndPostLayerWeight
from lightllm.models.qwen2_vl.model import QWen2VLTokenizer
from lightllm.models.qwen3.model import Qwen3TpPartModel
from lightllm.server.core.objs import SamplingParams
from lightllm.models.qwen3_moe.model import Qwen3MOEModel
from lightllm.server.multimodal_params import AudioItem, MultimodalParams, ImageItem
from lightllm.models.neo_chat_moe.vision_process import smart_resize
from lightllm.models.internvl.model import InternvlTokenizer
from lightllm.models.qwen_vl.layer_infer.pre_layer_infer import LlamaMultimodalPreLayerInfer
from lightllm.models.neo_chat.layer_infer.transformer_layer_infer import NeoChatTransformerLayerInfer
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.neo_chat.layer_weights.transformer_layer_weight import NeoChatTransformerLayerWeight
from lightllm.models.neo_chat.layer_weights.pre_and_post_layer_weight import NeoChatPreAndPostLayerWeight
from lightllm.common.basemodel.multimodal_tokenizer import BaseMultiModalTokenizer
from lightllm.models.neo_chat_moe.infer_struct import NeoChatInferStateInfo


@ModelRegistry(["neo_chat"], is_multimodal=True, condition=llm_model_type_is("qwen3"))
class NeoTpPartModel(Qwen3TpPartModel):

    pre_layer_infer_class = LlamaMultimodalPreLayerInfer
    transformer_layer_infer_class = NeoChatTransformerLayerInfer

    pre_and_post_weight_class = NeoChatPreAndPostLayerWeight
    transformer_weight_class = NeoChatTransformerLayerWeight

    infer_state_class = NeoChatInferStateInfo

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return

    def _init_inferstate_cls(self):
        pass

    def _init_config(self):
        with open(os.path.join(self.weight_dir_, "config.json"), "r") as json_file:
            all_config = json.load(json_file)
            self.config = all_config["llm_config"]
        # rename keys
        repair_config(self.config, same_names=["num_attention_heads", "n_head"])
        repair_config(self.config, same_names=["hidden_size", "n_embd", "n_embed"])
        repair_config(self.config, same_names=["num_hidden_layers", "n_layer"])
        if self.finetune_config:
            self.config["vocab_size"] = self.finetune_config.vocab_size
        return
