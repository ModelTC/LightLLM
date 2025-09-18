import os
import json
from lightllm.models.registry import ModelRegistry, llm_model_type_is
from lightllm.common.basemodel.multimodal_tokenizer import BaseMultiModalTokenizer
from lightllm.common.build_utils import repair_config
from lightllm.server.core.objs import SamplingParams
from lightllm.server.multimodal_params import AudioItem, MultimodalParams, ImageItem
from lightllm.models.qwen3_moe.model import Qwen3MOEModel
from lightllm.models.qwen_vl.layer_infer.pre_layer_infer import LlamaMultimodalPreLayerInfer
from lightllm.models.interns1.layer_weights.pre_and_post_layer_weight import (
    InternS1PreAndPostLayerWeight,
)


IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_TOKEN = "<IMG_CONTEXT>"



# Warp of the origal tokenizer
class InternS1Tokenizer(BaseMultiModalTokenizer):
    def __init__(self, tokenizer, model_cfg, **kwargs):
        super().__init__(tokenizer)
        self.llm_model_type = model_cfg.get("text_config").get("model_type")
        self.image_length = int(os.environ.get("INTERNVL_IMAGE_LENGTH", 256))

        self.image_start_tag = IMG_START_TOKEN
        self.image_start_id = tokenizer.convert_tokens_to_ids(self.image_start_tag)

        self.image_end_tag = IMG_END_TOKEN
        self.image_end_id = tokenizer.convert_tokens_to_ids(self.image_end_tag)


    def init_imageitem_extral_params(
        self, img: ImageItem, multi_params: MultimodalParams, sampling_params: SamplingParams
    ):
        img.extra_params["image_patch_max_num"] = 12 # 好丑的写法，后面改动
        return

    def init_audioitem_extral_params(
        self, audio: AudioItem, multi_params: MultimodalParams, sampling_params: SamplingParams
    ):
        return

    def get_image_token_length(self, img: ImageItem):
        return self.image_length

    def get_audio_token_length(self, audio: AudioItem):
        return

    # only change the impl of the encode func:
    def encode(self, prompt, multimodal_params: MultimodalParams = None, **kwargs):
        # TEXT<IMG_CONTEXT>TEXT<IMG_CONTEXT>TEXT --> TEXT<img></img>TEXT<img></img>TEXT
        image_tokens = IMG_START_TOKEN + IMG_END_TOKEN
        if multimodal_params is None:
            add_special_tokens = kwargs.get("add_special_tokens", True)
            return self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        image_count = len(multimodal_params.images)
        prompt = prompt.replace(IMG_TOKEN, image_tokens, image_count)
        origin_ids = self.tokenizer.encode(prompt, add_special_tokens=kwargs["add_special_tokens"])
        
        # print("[debug] prompt: ", prompt)
        # print("[debug] origin_ids: ", origin_ids)
        # import copy
        # origin_ids_ = copy.deepcopy(origin_ids)

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

        # print("[debug] input_ids: ", input_ids)
        # data = {
        #     "origin_ids": origin_ids_,
        #     "input_ids": input_ids
        # }  
        # with open("input_ids_lightllm.json", "w") as f:
        #     json.dump(data, f)

        return input_ids



@ModelRegistry(["interns1"], is_multimodal=True, condition=llm_model_type_is("qwen3_moe"))
class InternS1Qwen3MOETpPartModel(Qwen3MOEModel):
    # weight class
    pre_and_post_weight_class = InternS1PreAndPostLayerWeight

    # infer class
    pre_layer_infer_class = LlamaMultimodalPreLayerInfer

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return

    def _init_config(self):
        with open(os.path.join(self.weight_dir_, "config.json"), "r") as json_file:
            self.config = json.load(json_file)["text_config"]
        # rename keys
        repair_config(self.config, same_names=["num_attention_heads", "n_head"])
        repair_config(self.config, same_names=["hidden_size", "n_embd", "n_embed"])
        repair_config(self.config, same_names=["num_hidden_layers", "n_layer"])
        if self.finetune_config:
            self.config["vocab_size"] = self.finetune_config.vocab_size
        return


