import os
import json
import torch

from lightllm.common.build_utils import repair_config
from lightllm.models.registry import ModelRegistry, llm_model_type_is
from lightllm.models.qwen3.model import Qwen3TpPartModel
from lightllm.server.core.objs import SamplingParams
from lightllm.server.multimodal_params import AudioItem, MultimodalParams, ImageItem
from lightllm.models.neo_chat_moe.vision_process import smart_resize
from lightllm.models.qwen_vl.layer_infer.pre_layer_infer import LlamaMultimodalPreLayerInfer
from lightllm.models.neo_chat.layer_infer.transformer_layer_infer import NeoChatTransformerLayerInfer
from lightllm.models.neo_chat.layer_weights.transformer_layer_weight import NeoChatTransformerLayerWeight
from lightllm.models.neo_chat.layer_weights.pre_and_post_layer_weight import NeoChatPreAndPostLayerWeight
from lightllm.common.basemodel.multimodal_tokenizer import BaseMultiModalTokenizer
from lightllm.models.neo_chat_moe.infer_struct import NeoChatInferStateInfo


IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_TOKEN = "<image>"


class NeoChatTokenizer(BaseMultiModalTokenizer):
    """Tokenizer wrapper shared by the dense Neo model and the X2I API."""

    def __init__(self, tokenizer, model_cfg, **kwargs):
        super().__init__(tokenizer)
        vision_config = model_cfg["vision_config"]
        self.tokenizer = tokenizer
        self.min_pixel = vision_config["min_pixels"]
        self.max_pixel = vision_config["max_pixels"]
        self.patch_size = vision_config["patch_size"]
        self.downsample_ratio = vision_config["downsample_ratio"]

        self.image_token_id = model_cfg.get("image_token_id")
        self.image_start_tag = IMG_START_TOKEN
        self.image_start_id = tokenizer.convert_tokens_to_ids(self.image_start_tag)
        self.image_end_tag = IMG_END_TOKEN
        self.image_end_id = tokenizer.convert_tokens_to_ids(self.image_end_tag)
        self.image_tag = IMG_TOKEN

    def init_imageitem_extral_params(
        self, img: ImageItem, multi_params: MultimodalParams, sampling_params: SamplingParams
    ):
        img.extra_params["min_pixels"] = (
            sampling_params.min_pixels if sampling_params.min_pixels > 0 else self.min_pixel
        )
        img.extra_params["max_pixels"] = (
            sampling_params.max_pixels if sampling_params.max_pixels > 0 else self.max_pixel
        )
        assert img.extra_params["min_pixels"] <= img.extra_params["max_pixels"], (
            "min_pixels should be less than or equal to max_pixels"
        )

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
            min_pixels=img.extra_params.get("min_pixels", self.min_pixel),
            max_pixels=img.extra_params.get("max_pixels", self.max_pixel),
        )
        grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size
        token_num = int((grid_h * grid_w) * (self.downsample_ratio**2))
        img.grid_thwd = (
            1,
            int(grid_h * self.downsample_ratio),
            int(grid_w * self.downsample_ratio),
            1 - token_num,
        )
        return token_num

    def get_image_token_length_by_size(self, width: int, height: int):
        assert width > 0 and height > 0 and width % self.patch_size == 0 and height % self.patch_size == 0, (
            "width and height should be greater than 0 and divisible by patch_size"
        )
        grid_h, grid_w = height // self.patch_size, width // self.patch_size
        return int((grid_h * grid_w) * (self.downsample_ratio**2))

    def encode(self, prompt, multimodal_params: MultimodalParams = None, **kwargs):
        if multimodal_params is None:
            return self.tokenizer.encode(prompt, add_special_tokens=kwargs.get("add_special_tokens", True))

        image_count = len(multimodal_params.images)
        if not kwargs.get("already_tokenized", False):
            prompt = prompt.replace(IMG_TOKEN, IMG_START_TOKEN + IMG_END_TOKEN, image_count)
            origin_ids = self.tokenizer.encode(prompt, add_special_tokens=kwargs.get("add_special_tokens", True))
        else:
            origin_ids = list(prompt)

        input_ids = []
        image_id = 0
        while True:
            try:
                start_idx = origin_ids.index(self.image_start_id)
            except ValueError:
                break
            if start_idx + 1 >= len(origin_ids):
                break
            if origin_ids[start_idx + 1] != self.image_end_id:
                raise ValueError("image token error")
            if image_id >= image_count:
                raise ValueError("not enough images for image placeholders")

            input_ids.extend(origin_ids[: start_idx + 1])
            image = multimodal_params.images[image_id]
            image.start_idx = len(input_ids)
            input_ids.extend(range(image.token_id, image.token_id + image.token_num))
            input_ids.append(self.image_end_id)
            origin_ids = origin_ids[start_idx + 2 :]
            image_id += 1

        input_ids.extend(origin_ids)
        return input_ids

    def _build_t2i_query(self, msg, thinking_content=""):
        prompt = self.tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True
        )
        return prompt + thinking_content + IMG_START_TOKEN

    def fix_prompt(self, prompt: str, img_len: int):
        prompt_img_len = prompt.count(IMG_TOKEN)
        assert prompt_img_len <= img_len, f"not enough images provided, need {prompt_img_len}, given {img_len}"
        if prompt_img_len < img_len:
            return f"{IMG_TOKEN}\n" * (img_len - prompt_img_len) + prompt
        return prompt

    def get_query_for_it2i(self, prompt: str):
        image_len = prompt.count(IMG_TOKEN)
        query_condition = prompt + IMG_START_TOKEN if not prompt.endswith(IMG_START_TOKEN) else prompt
        query_text_uncondition = self._build_t2i_query(IMG_TOKEN * image_len)
        question_img_uncondition = self._build_t2i_query("")
        return query_condition, query_text_uncondition, question_img_uncondition

    def get_query_for_t2i(self, prompt: str, input_image_num: int = 0):
        image_len = prompt.count(IMG_TOKEN)
        query_condition = prompt + IMG_START_TOKEN if not prompt.endswith(IMG_START_TOKEN) else prompt
        query_uncondition = self._build_t2i_query(
            IMG_TOKEN * input_image_num,
            thinking_content=IMG_TOKEN * (image_len - input_image_num),
        )
        return query_condition, query_uncondition


@ModelRegistry(["neo_chat"], is_multimodal=True, condition=llm_model_type_is("qwen3"))
class NeoTpPartModel(Qwen3TpPartModel):

    pre_layer_infer_class = LlamaMultimodalPreLayerInfer
    transformer_layer_infer_class = NeoChatTransformerLayerInfer

    pre_and_post_weight_class = NeoChatPreAndPostLayerWeight
    transformer_weight_class = NeoChatTransformerLayerWeight

    infer_state_class = NeoChatInferStateInfo

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return

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

    def _init_custom(self):
        """Build Neo's independent T/H/W RoPE caches on the selected platform."""
        if self.head_dim_ % 4 != 0:
            raise ValueError(f"Neo head_dim must be divisible by 4, got {self.head_dim_}")

        rope_scaling = self.config.get("rope_scaling") or {}
        rope_scaling_factor = float(rope_scaling.get("factor", 1.0))
        if rope_scaling_factor <= 0:
            raise ValueError(f"rope scaling factor must be positive, got {rope_scaling_factor}")

        time_dim = self.head_dim_ // 2
        spatial_dim = self.head_dim_ // 4
        self._cos_cached, self._sin_cached = self._build_neo_rope_cache(
            rotary_dim=time_dim,
            base=float(self.config.get("rope_theta", 10000.0)),
            max_position_embeddings=int(self.config.get("max_position_embeddings", 16384)),
            scaling_factor=rope_scaling_factor,
        )
        self._hw_cos_cached, self._hw_sin_cached = self._build_neo_rope_cache(
            rotary_dim=spatial_dim,
            base=float(self.config.get("rope_theta_hw", 10000.0)),
            max_position_embeddings=int(self.config.get("max_position_embeddings_hw", 10000)),
            scaling_factor=rope_scaling_factor,
        )

    def _build_neo_rope_cache(self, rotary_dim, base, max_position_embeddings, scaling_factor):
        cache_len = max(
            int(max_position_embeddings * scaling_factor) + 1024 * 128,
            int(self.max_seq_length),
        )
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, rotary_dim, 2, device="cpu", dtype=torch.float32)
                / rotary_dim
            )
        )
        positions = torch.arange(cache_len, device="cpu", dtype=torch.float32) / scaling_factor
        freqs = torch.outer(positions, inv_freq)
        cos = torch.cos(freqs).to(dtype=self.data_type, device=self.target_device)
        sin = torch.sin(freqs).to(dtype=self.data_type, device=self.target_device)
        return cos, sin
