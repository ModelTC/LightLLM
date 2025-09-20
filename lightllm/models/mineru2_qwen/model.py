import re
import math

from lightllm.common.basemodel.multimodal_tokenizer import BaseMultiModalTokenizer
from lightllm.server.multimodal_params import AudioItem, MultimodalParams, ImageItem
from lightllm.server.core.objs import SamplingParams
from lightllm.models.registry import ModelRegistry
from lightllm.models.qwen2.model import Qwen2TpPartModel
from lightllm.models.qwen_vl.layer_infer.pre_layer_infer import LlamaMultimodalPreLayerInfer

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
        # 非 anyres：单视图，仅 base patch 序列
        patch_len = int(self.image_length)
        aspect_ratio = getattr(self.image_processor, "image_aspect_ratio", None)
        if not aspect_ratio or ("anyres" not in str(aspect_ratio)):
            return patch_len

        # anyres：按 ref 的 spatial + unpad + anyres_max 逻辑计数
        crop_size = self.image_processor.crop_size["height"]
        grid_w, grid_h = get_anyres_image_grid_shape(
            (img.image_w, img.image_h), self.image_processor.image_grid_pinpoints, crop_size
        )
        # base 视图（原图等比到 crop）
        base_tokens = patch_len
        patch_side = int(math.sqrt(patch_len))
        # h, w 为拼接后的整体网格尺寸（单位：patch）
        h = int(grid_h * patch_side)
        w = int(grid_w * patch_side)

        new_h, new_w = h, w
        max_num_patches = None
        m = re.search(r"anyres_max_(\d+)", str(aspect_ratio))
        if m:
            max_num_patches = int(m.group(1))
            times = math.sqrt((h * w) / (max_num_patches * patch_len))
            if times > 1.1:
                new_h = int(new_h // times)
                new_w = int(new_w // times)
        # 每行追加换行 token，数量等于行数 new_h
        extra_tokens = int(new_h * (new_w + 1))
        total_tokens = int(base_tokens + extra_tokens)

        print(f"[debug][spatial] P={patch_side}, N={patch_len}, Nx={grid_w}, Ny={grid_h}, crops={grid_w*grid_h}")
        if max_num_patches is not None:
            times = math.sqrt((h * w) / (max_num_patches * patch_len))
            print(
                f"[debug][spatial+unpad+anyres_max] h={h}, w={w}, "
                f"times={times:.4f}, h'={new_h}, w'={new_w}, newline={new_h}, extra_tokens~={extra_tokens}"
            )
        print(f"[debug][spatial] base_tokens={base_tokens}, extra_tokens={extra_tokens}, total_tokens={total_tokens}")
        return total_tokens

    def get_audio_token_length(self, audio: AudioItem):
        raise NotImplementedError

    # only change the impl of the encode func:
    def encode(self, prompt, multimodal_params: MultimodalParams = None, add_special_tokens: bool = True):
        if multimodal_params is None:
            return self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)

        prompt = prompt.replace(IMG_START_TOKEN + IMG_END_TOKEN, IMG_TOKEN)

        origin_ids = self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)

        # 单标记<image>：遇到img_token_index，直接展开K个占位token
        input_ids = []
        image_id = 0
        i = 0
        while i < len(origin_ids):
            tok = origin_ids[i]
            if tok == self.img_token_index:
                if image_id >= len(multimodal_params.images):
                    input_ids.append(tok)
                    i += 1
                    continue
                token_id = multimodal_params.images[image_id].token_id
                token_num = multimodal_params.images[image_id].token_num
                input_ids.extend(range(token_id, token_id + token_num))
                image_id += 1
                i += 1
            else:
                input_ids.append(tok)
                i += 1

        return input_ids


@ModelRegistry("mineru2_qwen", is_multimodal=True)
class Mineru2QwenForCausalLM(Qwen2TpPartModel):
    # infer class
    pre_layer_infer_class = LlamaMultimodalPreLayerInfer

    def __init__(self, kvargs):
        super().__init__(kvargs)
