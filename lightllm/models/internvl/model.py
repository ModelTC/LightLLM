import os
import json
from lightllm.models.internlm2.model import Internlm2TpPartModel
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.phi3.model import Phi3TpPartModel
from lightllm.models.qwen_vl.layer_infer.pre_layer_infer import LlamaMultimodalPreLayerInfer
from lightllm.server.multimodal_params import MultimodalParams, ImageItem
from lightllm.common.build_utils import repair_config
from lightllm.models.internvl.layer_weights.pre_and_post_layer_weight import (
    InternVLLlamaPreAndPostLayerWeight,
    InternVLPhi3PreAndPostLayerWeight,
)
from lightllm.models.internvl.layer_weights.pre_and_post_layer_weight import InternVLInternlm2PreAndPostLayerWeight
from lightllm.models.llava.llava_visual import LlavaVisionModel
from lightllm.models.internvl.img_process import get_image_patch
from typing import Dict
import lightllm.models.internvl.internvl_visual
import torch
import numpy

IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_TOKEN = "<image>"

# Warp of the origal tokenizer
class InternvlTokenizer:
    def __init__(self, tokenizer, model_cfg, **kwargs):

        self.llm_model_type = model_cfg.get("llm_config").get("model_type")
        self.tokenizer = tokenizer
        self.image_length = 256

        self.image_start_tag = IMG_START_TOKEN
        self.image_start_id = tokenizer.convert_tokens_to_ids(self.image_start_tag)

        self.image_end_tag = IMG_END_TOKEN
        self.image_end_id = tokenizer.convert_tokens_to_ids(self.image_end_tag)

    def get_image_token_length(self, img: ImageItem):
        return get_image_patch(img.image_w, img.image_h, use_thumbnail=True) * self.image_length

    # only change the impl of the encode func:
    def encode(self, prompt, multimodal_params: MultimodalParams = None, **kwargs):
        # TEXT<image>TEXT<image>TEXT --> TEXT<img></img>TEXT<img></img>TEXT
        image_tokens = IMG_START_TOKEN + IMG_END_TOKEN
        image_count = len(multimodal_params.images)
        prompt = prompt.replace(IMG_TOKEN, image_tokens, image_count)

        origin_ids = self.tokenizer.encode(prompt)
        # <img></img> --> <img>id,id+1...id+num</img>
        input_ids = []
        image_id = 0
        start_idx = 0
        while True:
            try:
                start_idx = origin_ids.index(self.image_start_id, start_idx)
                if start_idx + 1 >= len(origin_ids):
                    break
                if origin_ids[start_idx + 1] == self.image_end_id:
                    input_ids.extend(origin_ids[: start_idx + 1])
                    token_id = multimodal_params.images[image_id].token_id
                    token_num = multimodal_params.images[image_id].token_num
                    input_ids.extend(range(token_id, token_id + token_num))
                    input_ids.append(self.image_end_id)
                    origin_ids = origin_ids[start_idx + 2 :]
                    start_idx = 0
                    image_id += 1
                else:
                    raise ValueError("image token error")
            except ValueError:
                break
        input_ids.extend(origin_ids[start_idx:])
        return input_ids

    def __getattr__(self, name):
        if name != "encode":
            return getattr(self.tokenizer, name)
        return self.encode


class InternVLPhi3TpPartModel(Phi3TpPartModel):
    # weight class
    pre_and_post_weight_class = InternVLPhi3PreAndPostLayerWeight

    # infer class
    pre_layer_infer_class = LlamaMultimodalPreLayerInfer

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return

    def _init_config(self):
        with open(os.path.join(self.weight_dir_, "config.json"), "r") as json_file:
            self.config = json.load(json_file)["llm_config"]
        # rename keys
        repair_config(self.config, same_names=["num_attention_heads", "n_head"])
        repair_config(self.config, same_names=["hidden_size", "n_embd", "n_embed"])
        repair_config(self.config, same_names=["num_hidden_layers", "n_layer"])
        if self.finetune_config:
            self.config["vocab_size"] = self.finetune_config.vocab_size
        return


class InternVLInternlm2TpPartModel(Internlm2TpPartModel):
    # weight class
    pre_and_post_weight_class = InternVLInternlm2PreAndPostLayerWeight

    # infer class
    pre_layer_infer_class = LlamaMultimodalPreLayerInfer

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return

    def _init_config(self):
        with open(os.path.join(self.weight_dir_, "config.json"), "r") as json_file:
            self.config = json.load(json_file)["llm_config"]
        # rename keys
        repair_config(self.config, same_names=["num_attention_heads", "n_head"])
        repair_config(self.config, same_names=["hidden_size", "n_embd", "n_embed"])
        repair_config(self.config, same_names=["num_hidden_layers", "n_layer"])
        if self.finetune_config:
            self.config["vocab_size"] = self.finetune_config.vocab_size
        return


class InternVLLlamaTpPartModel(LlamaTpPartModel):
    # weight class
    pre_and_post_weight_class = InternVLLlamaPreAndPostLayerWeight

    # infer class
    pre_layer_infer_class = LlamaMultimodalPreLayerInfer

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return

    def _init_config(self):
        with open(os.path.join(self.weight_dir_, "config.json"), "r") as json_file:
            self.config = json.load(json_file)["llm_config"]
        # rename keys
        repair_config(self.config, same_names=["num_attention_heads", "n_head"])
        repair_config(self.config, same_names=["hidden_size", "n_embd", "n_embed"])
        repair_config(self.config, same_names=["num_hidden_layers", "n_layer"])
        if self.finetune_config:
            self.config["vocab_size"] = self.finetune_config.vocab_size
        return
