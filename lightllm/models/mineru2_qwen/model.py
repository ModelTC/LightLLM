import re
import os
import json

from lightllm.common.basemodel.multimodal_tokenizer import BaseMultiModalTokenizer
from lightllm.server.multimodal_params import AudioItem, MultimodalParams, ImageItem
from lightllm.server.core.objs import SamplingParams
from lightllm.models.registry import ModelRegistry
from lightllm.models.qwen2.model import Qwen2TpPartModel
from lightllm.models.qwen_vl.layer_infer.pre_layer_infer import LlamaMultimodalPreLayerInfer
from lightllm.models.internvl.layer_weights.pre_and_post_layer_weight import InternVLLlamaPreAndPostLayerWeight
from lightllm.models.internvl.img_process import get_image_patch

from ..mineru2_qwen.image_processing_mineru2 import Mineru2ImageProcessor
from .image_processing_mineru2 import get_anyres_image_grid_shape

IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_TOKEN = "<image>"


class Mineru2QwenTokenizer(BaseMultiModalTokenizer):
    def __init__(self, tokenizer, model_cfg):
        super().__init__(tokenizer)

        self.image_token = model_cfg.get("image_token", IMG_TOKEN)
        self.img_token_index = model_cfg.get("image_token_index", 151646)

        self.image_start_tag = IMG_START_TOKEN
        self.image_start_id = tokenizer.convert_tokens_to_ids(self.image_start_tag)

        self.image_end_tag = IMG_END_TOKEN
        self.image_end_id = tokenizer.convert_tokens_to_ids(self.image_end_tag)

        if "text_config" in model_cfg:
            patch_size = model_cfg["vision_config"]["patch_size"]
            image_size = model_cfg["vision_config"]["image_size"]
        else:
            mm_vision_tower = model_cfg.get("mm_vision_tower", "google/siglip-so400m-patch14-384")
            if isinstance(mm_vision_tower, list):
                mm_vision_tower = mm_vision_tower[0]
            mm_vision_tower = mm_vision_tower.split("/")[-1]
            vision_tower_match = re.match(r"^siglip-(\w+)-patch(\d+)-(\d+)$", mm_vision_tower)
            patch_size = int(vision_tower_match.group(2))
            default_img_size = int(vision_tower_match.group(3))
            image_size = model_cfg.get("img_size", default_img_size)
            image_size = model_cfg.get("mm_image_size", image_size)

        self.image_processor = Mineru2ImageProcessor(
            image_aspect_ratio=(model_cfg.get("image_aspect_ratio", None)),
            image_grid_pinpoints=(model_cfg.get("image_grid_pinpoints", None)),
        )
        self.image_length = (image_size // patch_size) ** 2

    def init_imageitem_extral_params(
        self, img: ImageItem, multi_params: MultimodalParams, sampling_params: SamplingParams
    ):
        return

    def init_audioitem_extral_params(
        self, audio: AudioItem, multi_params: MultimodalParams, sampling_params: SamplingParams
    ):
        raise NotImplementedError

    def get_image_token_length(self, img: ImageItem):
        # 对于 Mineru2 集成，视觉塔返回的是每个裁剪的一条 pooled 向量。
        # token 数应与裁剪数量一致：anyres 模式为 1（原图）+ 网格裁剪数，且每块含双视图（factor=2）。
        aspect = getattr(self.image_processor, "image_aspect_ratio", None)
        try:
            if aspect and (aspect == "anyres" or (isinstance(aspect, str) and "anyres_max" in aspect)):
                crop_size = self.image_processor.crop_size["height"]
                grid_w, grid_h = get_anyres_image_grid_shape(
                    (img.image_w, img.image_h), self.image_processor.image_grid_pinpoints, crop_size
                )
                base = int(grid_w * grid_h + 1)
                token_num = base
                print(
                    f"[debug] mineru2_tokenizer anyres img_size=({img.image_w},{img.image_h}) "
                    f"crop={crop_size} grid=({grid_w},{grid_h}) base={base} token_num={token_num}"
                )
                return token_num
            else:
                print(f"[debug] mineru2_tokenizer non-anyres token_num=1 aspect={aspect}")
                return 1
        except Exception as e:
            print(f"[debug] mineru2_tokenizer token_num_fallback due to {e}, return 1")
            return 1

    def get_audio_token_length(self, audio: AudioItem):
        raise NotImplementedError

    # only change the impl of the encode func:
    def encode(self, prompt, multimodal_params: MultimodalParams = None, add_special_tokens: bool = True):
        # TEXT<image>TEXT<image>TEXT --> TEXT<img></img>TEXT<img></img>TEXT
        image_tokens = IMG_START_TOKEN + IMG_END_TOKEN
        if multimodal_params is None:
            return self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        image_count = len(multimodal_params.images)
        prompt = prompt.replace(IMG_TOKEN, image_tokens, image_count)

        origin_ids = self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
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


@ModelRegistry("mineru2_qwen", is_multimodal=True)
class Mineru2QwenForCausalLM(Qwen2TpPartModel):
    # weight class
    # pre_and_post_weight_class = InternVLLlamaPreAndPostLayerWeight

    # infer class
    pre_layer_infer_class = LlamaMultimodalPreLayerInfer

    def __init__(self, kvargs):
        super().__init__(kvargs)
