import json
import numpy as np
from lightllm.common.basemodel.multimodal_tokenizer import BaseMultiModalTokenizer
from lightllm.models.qwen_vl.layer_infer.pre_layer_infer import LlamaMultimodalPreLayerInfer
from lightllm.server.multimodal_params import AudioItem, MultimodalParams, ImageItem
from lightllm.server.core.objs import SamplingParams
from lightllm.common.build_utils import repair_config
from lightllm.models.registry import ModelRegistry
from lightllm.models.qwen2_vl.infer_struct import Qwen2VLInferStateInfo
from lightllm.models.qwen2_vl.layer_infer.transformer_layer_infer import Qwen2VLTransformerLayerInfer

from .vision_process import smart_resize
from lightllm.models.qwen2.model import Qwen2TpPartModel
import os

# Warp of the origal tokenizer
class QWen2VLTokenizer(BaseMultiModalTokenizer):
    def __init__(self, tokenizer=None, image_processor=None, **kwargs):
        super().__init__(tokenizer)
        self.image_processor = image_processor
        self.min_pixel = self.image_processor.min_pixels
        self.max_pixel = self.image_processor.max_pixels
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
        position_delta = max(grid_h // self.merge_size, grid_w // self.merge_size) - token_num
        # delta 是为了mrope准备的，记录由于图片引入，position_id 产生的偏移量
        img.grid_thwd = (1, grid_h // self.merge_size, grid_w // self.merge_size, position_delta)
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


@ModelRegistry(["qwen2_vl", "qwen2_5_vl"], is_multimodal=True)
class Qwen2VLTpPartModel(Qwen2TpPartModel):

    pre_layer_infer_class = LlamaMultimodalPreLayerInfer
    transformer_layer_infer_class = Qwen2VLTransformerLayerInfer

    infer_state_class = Qwen2VLInferStateInfo

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return

    def _init_inferstate_cls(self):
        pass

    def _init_config(self):
        with open(os.path.join(self.weight_dir_, "config.json"), "r") as json_file:
            self.config = json.load(json_file)
        # rename keys
        repair_config(self.config, same_names=["num_attention_heads", "n_head"])
        repair_config(self.config, same_names=["hidden_size", "n_embd", "n_embed"])
        repair_config(self.config, same_names=["num_hidden_layers", "n_layer"])
        if self.finetune_config:
            self.config["vocab_size"] = self.finetune_config.vocab_size
        return
