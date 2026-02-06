import os
import json
from lightllm.common.build_utils import repair_config
from lightllm.models.registry import ModelRegistry, llm_model_type_is
from lightllm.models.qwen3_vl.infer_struct import Qwen3VLInferStateInfo
from lightllm.models.qwen3_vl.layer_infer.pre_layer_infer import Qwen3VLMultimodalPreLayerInfer
from lightllm.models.qwen3_vl.layer_infer.transformer_layer_infer import Qwen3VLTransformerLayerInfer
from lightllm.models.qwen3_vl.layer_weights.pre_and_post_layer_weight import Qwen3VLPreAndPostLayerWeight
from lightllm.models.qwen2_vl.model import QWen2VLTokenizer
from lightllm.models.qwen3.model import Qwen3TpPartModel
from lightllm.server.core.objs import SamplingParams
from lightllm.models.qwen3_moe.model import Qwen3MOEModel
from lightllm.server.multimodal_params import AudioItem, MultimodalParams, ImageItem
from lightllm.models.neo_chat_moe.vision_process import smart_resize
from lightllm.models.internvl.model import InternvlTokenizer
from lightllm.models.qwen_vl.layer_infer.pre_layer_infer import LlamaMultimodalPreLayerInfer
from lightllm.models.neo_chat_moe.layer_infer.transformer_layer_infer import NeoChatMOETransformerLayerInfer
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.neo_chat_moe.layer_weights.transformer_layer_weight import NeoChatMOETransformerLayerWeight
from lightllm.models.neo_chat_moe.layer_weights.pre_and_post_layer_weight import NeoChatMOEPreAndPostLayerWeight
from lightllm.common.basemodel.multimodal_tokenizer import BaseMultiModalTokenizer
from lightllm.models.neo_chat_moe.infer_struct import NeoChatInferStateInfo

IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_TOKEN = "<image>"
AUDIO_START_TOKEN = "<audio>"
AUDIO_END_TOKEN = "</audio>"


class NeoChatTokenizer(BaseMultiModalTokenizer):
    def __init__(self, tokenizer, model_cfg, **kwargs):
        super().__init__(tokenizer)
        self.tokenizer = tokenizer
        self.min_pixel = model_cfg.get("vision_config").get("min_pixels")
        self.max_pixel = model_cfg.get("vision_config").get("max_pixels")
        self.patch_size = model_cfg.get("vision_config").get("patch_size")
        self.downsample_ratio = model_cfg.get("vision_config").get("downsample_ratio")

        self.image_token_id = model_cfg.get("image_token_id")
        self.image_start_tag = IMG_START_TOKEN
        self.image_start_id = tokenizer.convert_tokens_to_ids(self.image_start_tag)
        self.image_end_tag = IMG_END_TOKEN
        self.image_end_id = tokenizer.convert_tokens_to_ids(self.image_end_tag)

    def init_imageitem_extral_params(
        self, img: ImageItem, multi_params: MultimodalParams, sampling_params: SamplingParams
    ):
        img.extra_params["min_pixels"] = (
            sampling_params.min_pixels if sampling_params.min_pixels > 0 else self.min_pixel
        )
        img.extra_params["max_pixels"] = (
            sampling_params.max_pixels if sampling_params.max_pixels > 0 else self.max_pixel
        )
        assert (
            img.extra_params["min_pixels"] <= img.extra_params["max_pixels"]
        ), "min_pixels should be less than or equal to max_pixels"
        return

    def init_audioitem_extral_params(
        self, audio: AudioItem, multi_params: MultimodalParams, sampling_params: SamplingParams
    ):
        raise NotImplementedError

    def get_audio_token_length(self, audio: AudioItem):
        raise NotImplementedError

    def get_image_token_length(self, img: ImageItem):
        width, height = img.image_w, img.image_h
        resized_height, resized_width = smart_resize(
            height=height,
            width=width,
            factor=int(self.patch_size // self.downsample_ratio),
            min_pixels=img.extra_params["min_pixels"],
            max_pixels=img.extra_params["max_pixels"],
        )
        grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size
        token_num = int((grid_h * grid_w) * (self.downsample_ratio ** 2))
        # 这里的grid_h和grid_w需要* self.downsample_ratio么？再仔细看下代码
        img.grid_thwd = (1, int(grid_h * self.downsample_ratio), int(grid_w * self.downsample_ratio), 1 - token_num)
        return token_num

    # only change the impl of the encode func:
    def encode(self, prompt, multimodal_params: MultimodalParams = None, **kwargs):
        # TEXT<image>TEXT<image>TEXT --> TEXT<img></img>TEXT<img></img>TEXT
        image_tokens = IMG_START_TOKEN + IMG_END_TOKEN
        if multimodal_params is None:
            add_special_tokens = kwargs.get("add_special_tokens", True)
            return self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        image_count = len(multimodal_params.images)
        if not kwargs.get("already_tokenized", False):
            prompt = prompt.replace(IMG_TOKEN, image_tokens, image_count)
            origin_ids = self.tokenizer.encode(prompt, add_special_tokens=kwargs["add_special_tokens"])
        else:
            origin_ids = prompt
        # <img></img> --> <img>id,id+1...id+num</img>
        input_ids = []
        image_id = 0
        start_idx = 0
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


@ModelRegistry(["neo_chat"], is_multimodal=True, condition=llm_model_type_is("qwen3_moe"))
class NeoTpMOEPartModel(Qwen3MOEModel):

    pre_layer_infer_class = LlamaMultimodalPreLayerInfer
    transformer_layer_infer_class = NeoChatMOETransformerLayerInfer

    pre_and_post_weight_class = NeoChatMOEPreAndPostLayerWeight
    transformer_weight_class = NeoChatMOETransformerLayerWeight

    infer_state_class = NeoChatInferStateInfo

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return

    def _init_inferstate_cls(self):
        pass

    def _init_config(self):
        with open(os.path.join(self.weight_dir_, "config.json"), "r") as json_file:
            all_config = json.load(json_file)
            self.config = all_config["llm_config"]
        # rename keys
        repair_config(self.config, same_names=["num_attention_heads", "n_head"])
        repair_config(self.config, same_names=["hidden_size", "n_embd", "n_embed"])
        repair_config(self.config, same_names=["num_hidden_layers", "n_layer"])
        if self.finetune_config:
            self.config["vocab_size"] = self.finetune_config.vocab_size
        return
