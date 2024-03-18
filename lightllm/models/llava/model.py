import re
import json
import numpy as np
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.qwen_vl.layer_infer.pre_layer_infer import LlamaMultimodalPreLayerInfer
from lightllm.server.multimodal_params import MultimodalParams


# Warp of the origal tokenizer
class LlavaTokenizer:
    def __init__(self, tokenizer, model_cfg):
        self.tokenizer = tokenizer
        self.image_token = model_cfg.get("image_token", "<image>")
        mm_vision_tower = model_cfg.get("mm_vision_tower", "openai/clip-vit-large-patch14-336")
        if isinstance(mm_vision_tower, list):
            mm_vision_tower = mm_vision_tower[0]
        mm_vision_tower = mm_vision_tower.split("/")[-1]
        vision_tower_match = re.match(r"^clip-vit-large-patch(\d+)-(\d+)$", mm_vision_tower)
        patch_size = int(vision_tower_match.group(1))
        default_img_size = int(vision_tower_match.group(2))
        image_size = model_cfg.get("img_size", default_img_size)
        image_size = model_cfg.get("mm_image_size", image_size)
        # (image_size // patch_size) ** 2: (336 // 14) ** 2 = 576
        self.image_length = (image_size // patch_size) ** 2
        self.skip_start = model_cfg.get("skip_start", True)

    # only change the impl of the encode func:
    def encode(self, prompt, multimodal_params: MultimodalParams = None):

        # split prompt by <image>, and merge parts by [pad_id] * 576
        ids_chunks = [self.tokenizer(x).input_ids for x in prompt.split(self.image_token)]
        input_ids = ids_chunks[0]
        image_id = 0

        for ids in ids_chunks[1:]:
            # skip the start token
            if len(ids) > 0 and ids[0] == self.tokenizer.bos_token_id and self.skip_start:
                ids = ids[1:]

            token_id = multimodal_params.images[image_id].token_id
            token_num = multimodal_params.images[image_id].token_num
            assert token_num == self.image_length, "invalid token num: {} vs {}!".format(token_num, self.image_length)

            input_ids.extend(range(token_id, token_id + token_num))
            input_ids.extend(ids)
            image_id += 1
        if multimodal_params:
            image_cnt = len(multimodal_params.images)
            assert image_cnt == image_id, "invalid image tag num: {} vs {}!".format(image_cnt, image_id)
        return input_ids

    def __getattr__(self, name):
        if name != "encode":
            return getattr(self.tokenizer, name)
        return self.encode


class LlavaTpPartModel(LlamaTpPartModel):

    # infer class
    pre_layer_infer_class = LlamaMultimodalPreLayerInfer

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return
