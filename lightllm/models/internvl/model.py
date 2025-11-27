import os
import json
from lightllm.models.registry import ModelRegistry, llm_model_type_is
from lightllm.common.basemodel.multimodal_tokenizer import BaseMultiModalTokenizer
from lightllm.common.build_utils import repair_config
from lightllm.server.core.objs import SamplingParams
from lightllm.server.multimodal_params import AudioItem, MultimodalParams, ImageItem
from lightllm.models.internlm2.model import Internlm2TpPartModel
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.phi3.model import Phi3TpPartModel
from lightllm.models.qwen2.model import Qwen2TpPartModel
from lightllm.models.qwen3.model import Qwen3TpPartModel
from lightllm.models.qwen3_moe.model import Qwen3MOEModel
from lightllm.models.deepseek2.model import Deepseek2TpPartModel
from lightllm.models.qwen_vl.layer_infer.pre_layer_infer import LlamaMultimodalPreLayerInfer
from lightllm.models.internvl.layer_weights.pre_and_post_layer_weight import (
    InternVLLlamaPreAndPostLayerWeight,
    InternVLPhi3PreAndPostLayerWeight,
)
from lightllm.models.internvl.layer_weights.pre_and_post_layer_weight import InternVLInternlm2PreAndPostLayerWeight
from lightllm.models.vit import get_image_patch_func
from lightllm.models.whisper.defaults import MIN_AUDIO_LEN

IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_TOKEN = "<image>"
AUDIO_START_TOKEN = "<audio>"
AUDIO_END_TOKEN = "</audio>"


# Warp of the origal tokenizer
class InternvlTokenizer(BaseMultiModalTokenizer):
    def __init__(self, tokenizer, model_cfg, **kwargs):
        super().__init__(tokenizer)
        self.llm_model_type = model_cfg.get("llm_config").get("model_type")
        self.image_length = int(os.environ.get("INTERNVL_IMAGE_LENGTH", 256))

        self.image_start_tag = IMG_START_TOKEN
        self.image_start_id = tokenizer.convert_tokens_to_ids(self.image_start_tag)

        self.image_end_tag = IMG_END_TOKEN
        self.image_end_id = tokenizer.convert_tokens_to_ids(self.image_end_tag)

        self.audio_start_tag = AUDIO_START_TOKEN
        self.audio_start_id = tokenizer.convert_tokens_to_ids(self.audio_start_tag)

        self.audio_end_tag = AUDIO_END_TOKEN
        self.audio_end_id = tokenizer.convert_tokens_to_ids(self.audio_end_tag)
        self.get_image_patch_func = get_image_patch_func(kwargs["weight_dir"])

        self.audio_min_length = MIN_AUDIO_LEN
        self.audio_max_length = 16000 * 30

    def init_imageitem_extral_params(
        self, img: ImageItem, multi_params: MultimodalParams, sampling_params: SamplingParams
    ):
        if sampling_params.image_max_patch_num > 0:
            img.extra_params["image_patch_max_num"] = sampling_params.image_max_patch_num
            return
        elif os.getenv("MAX_PATCH_NUM"):
            img.extra_params["image_patch_max_num"] = int(os.getenv("MAX_PATCH_NUM"))
            return
        else:
            num_images = len(multi_params.images)
            if num_images == 1:
                img.extra_params["image_patch_max_num"] = 12
            elif num_images > 1 and num_images <= 6:
                img.extra_params["image_patch_max_num"] = 6
            elif num_images > 6:
                img.extra_params["image_patch_max_num"] = 0
        return

    def init_audioitem_extral_params(
        self, audio: AudioItem, multi_params: MultimodalParams, sampling_params: SamplingParams
    ):
        return

    def get_image_token_length(self, img: ImageItem):
        return (
            self.get_image_patch_func(
                img.image_w, img.image_h, max_num=img.extra_params["image_patch_max_num"], use_thumbnail=True
            )
            * self.image_length
        )

    def get_audio_token_length(self, audio: AudioItem):
        L = audio.audio_length
        audio_token_num = 0
        chunk_lens = []
        if L <= self.audio_max_length:
            cur_len = L
            if cur_len < self.audio_min_length:
                cur_len = self.audio_min_length
            chunk_lens.append(cur_len)
        else:
            start = 0
            while start < L:
                end = min(start + self.audio_max_length, L)
                cur_len = end - start

                if cur_len < self.audio_min_length:
                    cur_len = self.audio_min_length

                chunk_lens.append(cur_len)
                start = end
        for chunk_len in chunk_lens:
            mel_len = chunk_len // 160
            dilation = 1
            L_in = mel_len
            for (padding, kernel_size, stride) in eval("[(1,3,1)] + [(1,3,2)] "):
                L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
                L_out = 1 + L_out // stride
                L_in = L_out
            audio_len_after_cnn = L_out
            chunk_token_num = (audio_len_after_cnn - 2) // 2 + 1
            audio_token_num += int(chunk_token_num)
        return audio_token_num

    # only change the impl of the encode func:
    def encode(self, prompt, multimodal_params: MultimodalParams = None, **kwargs):
        # TEXT<image>TEXT<image>TEXT --> TEXT<img></img>TEXT<img></img>TEXT
        image_tokens = IMG_START_TOKEN + IMG_END_TOKEN
        if multimodal_params is None:
            add_special_tokens = kwargs.get("add_special_tokens", True)
            return self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        image_count = len(multimodal_params.images)
        prompt = prompt.replace(IMG_TOKEN, image_tokens, image_count)

        origin_ids = self.tokenizer.encode(prompt, add_special_tokens=kwargs["add_special_tokens"])
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

        # audio
        origin_ids = input_ids
        input_ids = []
        audio_id = 0
        start_idx = 0
        while True:
            try:
                start_idx = origin_ids.index(self.audio_start_id, start_idx)
                if start_idx + 1 >= len(origin_ids):
                    break
                if origin_ids[start_idx + 1] == self.audio_end_id:
                    input_ids.extend(origin_ids[: start_idx + 1])
                    token_id = multimodal_params.audios[audio_id].token_id
                    token_num = multimodal_params.audios[audio_id].token_num
                    input_ids.extend(range(token_id, token_id + token_num))
                    input_ids.append(self.audio_end_id)
                    origin_ids = origin_ids[start_idx + 2 :]
                    start_idx = 0
                    audio_id += 1
                else:
                    raise ValueError("audio token error")
            except ValueError:
                break
        input_ids.extend(origin_ids[start_idx:])
        return input_ids


@ModelRegistry(["internvl_chat"], is_multimodal=True, condition=llm_model_type_is("phi3"))
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


@ModelRegistry(["internvl_chat"], is_multimodal=True, condition=llm_model_type_is("internlm2"))
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


@ModelRegistry(["internvl_chat"], is_multimodal=True, condition=llm_model_type_is("llama"))
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


@ModelRegistry(["internvl_chat"], is_multimodal=True, condition=llm_model_type_is("qwen2"))
class InternVLQwen2TpPartModel(Qwen2TpPartModel):
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


@ModelRegistry(["internvl_chat"], is_multimodal=True, condition=llm_model_type_is(["deepseek_v2", "deepseek_v3"]))
class InternVLDeepSeek2TpPartModel(Deepseek2TpPartModel):
    # support Deepseek2,3,R1
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


@ModelRegistry(["internvl_chat"], is_multimodal=True, condition=llm_model_type_is("qwen3"))
class InternVLQwen3TpPartModel(Qwen3TpPartModel):
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


@ModelRegistry(["internvl_chat"], is_multimodal=True, condition=llm_model_type_is("qwen3_moe"))
class InternVLQwen3MOETpPartModel(Qwen3MOEModel):
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
