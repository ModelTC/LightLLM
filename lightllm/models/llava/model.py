import os
import re
import json
import numpy as np
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.qwen_vl.layer_infer.pre_layer_infer import LlamaMultimodalPreLayerInfer
from lightllm.models.llava.layer_weights.pre_and_post_layer_weight import LlavaPreAndPostLayerWeight
from lightllm.server.multimodal_params import MultimodalParams, ImageItem
from lightllm.server.core.objs import SamplingParams
from lightllm.common.build_utils import repair_config
from transformers import AutoConfig


# Warp of the origal tokenizer
class LlavaTokenizer:
    def __init__(self, tokenizer, model_cfg):
        self.tokenizer = tokenizer
        self.image_token = model_cfg.get("image_token", "<image>")
        # for llava-v1.5-7b-hf model
        if "text_config" in model_cfg:
            patch_size = model_cfg["vision_config"]["patch_size"]
            image_size = model_cfg["vision_config"]["image_size"]
        else:
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

    def init_imageItem_extral_params(
        self, img: ImageItem, multi_params: MultimodalParams, sampling_params: SamplingParams
    ):
        return

    def get_image_token_length(self, img: ImageItem):
        return self.image_length

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
    # weight class
    pre_and_post_weight_class = LlavaPreAndPostLayerWeight

    # infer class
    pre_layer_infer_class = LlamaMultimodalPreLayerInfer

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return

    def _init_config(self):
        with open(os.path.join(self.weight_dir_, "config.json"), "r") as json_file:
            self.config = json.load(json_file)
        # for llava-v1.5-7b-hf model, should load config from transformers
        if "text_config" in self.config:
            config = AutoConfig.from_pretrained(self.weight_dir_, trust_remote_code=True)
            self.config = config.text_config.to_dict()
        # rename keys
        repair_config(self.config, same_names=["num_attention_heads", "n_head"])
        repair_config(self.config, same_names=["hidden_size", "n_embd", "n_embed"])
        repair_config(self.config, same_names=["num_hidden_layers", "n_layer"])
        if self.finetune_config:
            self.config["vocab_size"] = self.finetune_config.vocab_size
        return
