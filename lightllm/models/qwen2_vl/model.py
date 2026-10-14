import json
import numpy as np
from lightllm.common.basemodel.multimodal_tokenizer import BaseMultiModalTokenizer
from lightllm.models.qwen_vl.layer_infer.pre_layer_infer import LlamaMultimodalPreLayerInfer
from lightllm.server.multimodal_params import AudioItem, MultimodalParams, ImageItem
from lightllm.server.core.objs import SamplingParams
from lightllm.common.build_utils import repair_config
from lightllm.models.qwen2_vl.infer_struct import Qwen2VLInferStateInfo
from lightllm.models.qwen2_vl.layer_infer.transformer_layer_infer import Qwen2VLTransformerLayerInfer

from .vision_process import smart_resize
from lightllm.models.qwen2.model import Qwen2TpPartModel
import os
from typing import Union, List
from lightllm.utils.error_utils import InvalidRequestError

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

    def encode(self, prompt: Union[str, List[int]], multimodal_params: MultimodalParams = None, **kwargs):
        if isinstance(prompt, str):
            origin_ids = self.tokenizer.encode(prompt, **kwargs)
        elif isinstance(prompt, list):
            origin_ids = prompt
        else:
            raise ValueError(f"Unsupported prompt type: {type(prompt)}")

        image_spans = self._find_image_spans(origin_ids)

        if multimodal_params is not None and len(multimodal_params.images) != len(image_spans):
            raise InvalidRequestError(
                f"invalid image tag num: {len(multimodal_params.images)} images vs {len(image_spans)} tags"
            )

        input_ids = []
        offset = 0
        for image_id, (start, end) in enumerate(image_spans):
            input_ids.extend(origin_ids[offset : start + 1])
            if multimodal_params is not None:
                image = multimodal_params.images[image_id]
                image.start_idx = len(input_ids)
                input_ids.extend(range(image.token_id, image.token_id + image.token_num))
            input_ids.append(self.image_end_id)
            offset = end + 1
        input_ids.extend(origin_ids[offset:])
        return input_ids

    def _find_image_spans(self, token_ids):
        spans = []
        offset = 0
        while True:
            try:
                start = token_ids.index(self.image_start_id, offset)
            except ValueError:
                return spans
            try:
                end = token_ids.index(self.image_end_id, start + 1)
            except ValueError:
                raise InvalidRequestError("invalid image token sequence") from None
            for index in range(start + 1, end):
                if token_ids[index] != self.image_token_id:
                    raise InvalidRequestError("invalid image token sequence")
            spans.append((start, end))
            offset = end + 1


class Qwen2VLTpPartModel(Qwen2TpPartModel):

    pre_layer_infer_class = LlamaMultimodalPreLayerInfer
    transformer_layer_infer_class = Qwen2VLTransformerLayerInfer

    infer_state_class = Qwen2VLInferStateInfo

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return

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
