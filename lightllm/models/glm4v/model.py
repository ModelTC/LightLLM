import os
import json
import numpy as np
from lightllm.common.build_utils import repair_config
from lightllm.models.registry import ModelRegistry
from lightllm.models.qwen2_vl.infer_struct import Qwen2VLInferStateInfo
from lightllm.models.qwen_vl.layer_infer.pre_layer_infer import LlamaMultimodalPreLayerInfer
from lightllm.models.qwen2_vl.layer_infer.transformer_layer_infer import Qwen2VLTransformerLayerInfer
from lightllm.models.glm4v.layer_infer.transformer_layer_infer import Glm4VTransformerLayerInfer
from lightllm.models.glm4v.layer_weight.pre_and_post_layer_weight import Glm4VPreAndPostLayerWeight
from lightllm.models.glm4v.layer_weight.transformer_layer_weight import Glm4VTransformerLayerWeight
from lightllm.server.multimodal_params import MultimodalParams
from lightllm.models.qwen2_vl.model import QWen2VLTokenizer
from lightllm.models.qwen2.model import Qwen2TpPartModel


class GLM4VTokenizer(QWen2VLTokenizer):
    def __init__(self, tokenizer=None, image_processor=None, **kwargs):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.min_pixel = self.image_processor.size["shortest_edge"]
        self.max_pixel = self.image_processor.size["longest_edge"]
        self.patch_size = self.image_processor.patch_size
        self.merge_size = self.image_processor.merge_size
        self.image_start_id = kwargs["model_cfg"]["image_start_token_id"]
        self.image_end_id = kwargs["model_cfg"]["image_end_token_id"]
        self.image_token_id = kwargs["model_cfg"]["image_token_id"]

    def encode(self, prompt, multimodal_params: MultimodalParams = None, **kwargs):
        origin_ids = self.tokenizer.encode(prompt)

        # <img><image_pad></img> -> <img></img>
        origin_ids = [token for token in origin_ids if token != self.image_token_id]
        # <img></img> --> <img>id,id+1...id+num</img>
        input_ids = []
        image_id = 0
        while True:
            try:
                start_idx = origin_ids.index(self.image_start_id)
                if start_idx + 1 >= len(origin_ids):
                    break
                if origin_ids[start_idx + 1] == self.image_end_id:
                    input_ids.extend(origin_ids[: start_idx + 1])
                    token_id = multimodal_params.images[image_id].token_id
                    token_num = multimodal_params.images[image_id].token_num
                    multimodal_params.images[image_id].start_idx = len(input_ids)
                    input_ids.extend(range(token_id, token_id + token_num))
                    input_ids.append(self.image_end_id)
                    origin_ids = origin_ids[start_idx + 2 :]
                    image_id += 1
                else:
                    raise ValueError("image token error")
            except ValueError:
                break
        input_ids.extend(origin_ids)
        return input_ids


@ModelRegistry(["glm4v"], is_multimodal=True)
class GLM4VTpPartModel(Qwen2TpPartModel):

    pre_layer_infer_class = LlamaMultimodalPreLayerInfer
    transformer_layer_infer_class = Glm4VTransformerLayerInfer

    pre_and_post_weight_class = Glm4VPreAndPostLayerWeight
    transformer_weight_class = Glm4VTransformerLayerWeight

    infer_state_class = Qwen2VLInferStateInfo

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return

    def _init_inferstate_cls(self):
        pass

    def _init_config(self):
        with open(os.path.join(self.weight_dir_, "config.json"), "r") as json_file:
            all_config = json.load(json_file)
            self.config = all_config["text_config"]
        # rename keys
        repair_config(self.config, same_names=["num_attention_heads", "n_head"])
        repair_config(self.config, same_names=["hidden_size", "n_embd", "n_embed"])
        repair_config(self.config, same_names=["num_hidden_layers", "n_layer"])
        if self.finetune_config:
            self.config["vocab_size"] = self.finetune_config.vocab_size
        return
