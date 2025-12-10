import json
import numpy as np
import unicodedata
from lightllm.common.basemodel.multimodal_tokenizer import BaseMultiModalTokenizer
from lightllm.models.qwen.model import QWenTpPartModel
from lightllm.models.qwen_vl.layer_infer.pre_layer_infer import LlamaMultimodalPreLayerInfer
from lightllm.server.multimodal_params import AudioItem, MultimodalParams, ImageItem
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessorMixin
from lightllm.server.core.objs import SamplingParams
from transformers.tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from typing import List, Optional, Union
from transformers.utils import TensorType, logging
from lightllm.models.qwen2_vl.flashattention_infer_struct import Qwen2VLFlashAttentionStateInfo
from lightllm.common.build_utils import repair_config
from lightllm.models.registry import ModelRegistry
from lightllm.models.qwen3_vl.infer_struct import Qwen3VLInferStateInfo
from lightllm.models.qwen3_vl.layer_infer.pre_layer_infer import Qwen3VLMultimodalPreLayerInfer
from lightllm.models.qwen3_vl.layer_infer.transformer_layer_infer import Qwen3VLTransformerLayerInfer
from lightllm.models.qwen3_vl.layer_weights.pre_and_post_layer_weight import Qwen3VLPreAndPostLayerWeight
from lightllm.models.qwen3_vl.layer_weights.transformers_layer_weight import Qwen3VLTransformerLayerWeight
from lightllm.models.qwen3_vl_moe.layer_weights.transformers_layer_weight import Qwen3VLMOETransformerLayerWeight
from lightllm.models.qwen3_vl_moe.layer_infer.transformer_layer_infer import Qwen3VLMOETransformerLayerInfer

import torch
from PIL import Image
from lightllm.models.qwen2_vl.vision_process import smart_resize
from lightllm.utils.envs_utils import enable_env_vars, get_env_start_args
from lightllm.models.qwen3.model import Qwen3TpPartModel
from lightllm.models.qwen3_moe.model import Qwen3MOEModel
import os


class QWen3VLTokenizer(BaseMultiModalTokenizer):
    def __init__(self, tokenizer=None, image_processor=None, **kwargs):
        super().__init__(tokenizer)
        self.image_processor = image_processor
        self.min_pixel = self.image_processor.size["shortest_edge"]
        self.max_pixel = self.image_processor.size["longest_edge"]
        self.patch_size = self.image_processor.patch_size
        self.merge_size = self.image_processor.merge_size
        self.image_start_id = kwargs["model_cfg"]["vision_start_token_id"]
        self.image_end_id = kwargs["model_cfg"]["vision_end_token_id"]
        self.image_token_id = kwargs["model_cfg"]["image_token_id"]

    def init_imageitem_extral_params(
        self, img: ImageItem, multi_params: MultimodalParams, sampling_params: SamplingParams
    ):
        return

    def init_audioitem_extral_params(
        self, audio: AudioItem, multi_params: MultimodalParams, sampling_params: SamplingParams
    ):
        raise NotImplementedError

    def get_image_token_length(self, img: ImageItem):
        width, height = img.image_w, img.image_h
        factor = self.patch_size * self.merge_size
        resized_height, resized_width = smart_resize(
            height=height, width=width, factor=factor, min_pixels=self.min_pixel, max_pixels=self.max_pixel
        )
        grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size
        token_num = (grid_h * grid_w) // (self.merge_size ** 2)
        print(f"token_num is {token_num}")
        return token_num

    def get_audio_token_length(self, audio: AudioItem):
        raise NotImplementedError

    def encode(self, prompt, multimodal_params: MultimodalParams = None, **kwargs):

        origin_ids = self.tokenizer.encode(prompt)

        # <img><image_pad></img> -> <img></img>
        origin_ids = [token for token in origin_ids if token != self.image_token_id]
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


@ModelRegistry(["qwen3_vl"], is_multimodal=True)
class Qwen3VLTpPartModel(Qwen3TpPartModel):

    pre_layer_infer_class = Qwen3VLMultimodalPreLayerInfer
    transformer_layer_infer_class = Qwen3VLTransformerLayerInfer

    pre_and_post_weight_class = Qwen3VLPreAndPostLayerWeight
    transformer_weight_class = Qwen3VLTransformerLayerWeight

    infer_state_class = Qwen3VLInferStateInfo

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return

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


@ModelRegistry(["qwen3_vl_moe"], is_multimodal=True)
class Qwen3VLMOETpPartModel(Qwen3MOEModel):

    pre_layer_infer_class = Qwen3VLMultimodalPreLayerInfer
    transformer_layer_infer_class = Qwen3VLMOETransformerLayerInfer

    pre_and_post_weight_class = Qwen3VLPreAndPostLayerWeight
    transformer_weight_class = Qwen3VLMOETransformerLayerWeight

    infer_state_class = Qwen3VLInferStateInfo

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return

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
