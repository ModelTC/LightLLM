import os
import re
import json
import numpy as np
import torch
from lightllm.models.registry import ModelRegistry
from lightllm.common.basemodel.multimodal_tokenizer import BaseMultiModalTokenizer
from lightllm.common.kv_cache_mem_manager.mem_utils import select_mem_manager_class
from lightllm.models.gemma3.infer_struct import Gemma3InferStateInfo
from lightllm.models.gemma3.layer_infer.post_layer_infer import Gemma3PostLayerInfer
from lightllm.models.gemma3.layer_infer.pre_layer_infer import Gemma3PreLayerInfer
from lightllm.models.gemma3.layer_infer.transformer_layer_infer import Gemma3TransformerLayerInfer
from lightllm.models.gemma3.layer_weights.pre_and_post_layer_weight import Gemma3PreAndPostLayerWeight
from lightllm.models.gemma3.layer_weights.transformer_layer_weight import Gemma3TransformerLayerWeight
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.qwen_vl.layer_infer.pre_layer_infer import LlamaMultimodalPreLayerInfer
from lightllm.models.llava.layer_weights.pre_and_post_layer_weight import LlavaPreAndPostLayerWeight
from lightllm.server.multimodal_params import AudioItem, MultimodalParams, ImageItem
from lightllm.server.core.objs import SamplingParams
from lightllm.common.build_utils import repair_config
from transformers import AutoConfig
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


# Warp of the origal tokenizer
class Gemma3Tokenizer(BaseMultiModalTokenizer):
    def __init__(self, tokenizer, model_cfg):
        super().__init__(tokenizer)
        self.image_token = model_cfg.get("image_token", "<start_of_image>")
        self.boi_token = "<start_of_image>"
        self.eoi_token = "<end_of_image>"
        self.boi_token_index: int = model_cfg.get("boi_token_index", 255_999)
        self.eoi_token_index: int = model_cfg.get("eoi_token_index", 256_000)
        self.image_token_index: int = model_cfg.get("image_token_index", 262_144)
        self.mm_tokens_per_image: int = model_cfg.get("mm_tokens_per_image", 256)

        self.image_length = self.mm_tokens_per_image

    def init_imageitem_extral_params(
        self, img: ImageItem, multi_params: MultimodalParams, sampling_params: SamplingParams
    ):
        return

    def init_audioitem_extral_params(
        self, audio: AudioItem, multi_params: MultimodalParams, sampling_params: SamplingParams
    ):
        raise NotImplementedError

    def get_image_token_length(self, img: ImageItem):
        return self.image_length

    def get_audio_token_length(self, audio: AudioItem):
        raise NotImplementedError

    # only change the impl of the encode func:
    def encode(self, prompt, multimodal_params: MultimodalParams = None, add_special_tokens=False):
        if multimodal_params is None:
            return self.tokenizer(prompt).input_ids

        ids_chunks = [self.tokenizer(x).input_ids for x in prompt.split(self.boi_token)]
        input_ids = ids_chunks[0]
        image_id = 0

        for ids in ids_chunks[1:]:
            token_id = multimodal_params.images[image_id].token_id
            token_num = multimodal_params.images[image_id].token_num
            assert token_num == self.image_length, "invalid token num: {} vs {}!".format(token_num, self.image_length)

            input_ids.append(self.boi_token_index)
            input_ids.extend(range(token_id, token_id + self.mm_tokens_per_image))
            input_ids.append(self.eoi_token_index)
            input_ids.extend(ids)
            image_id += 1
        if multimodal_params:
            image_cnt = len(multimodal_params.images)
            assert image_cnt == image_id, "invalid image tag num: {} vs {}!".format(image_cnt, image_id)
        return input_ids


@ModelRegistry("gemma3")
class Gemma3TpPartModel(LlamaTpPartModel):
    # weight class
    pre_and_post_weight_class = Gemma3PreAndPostLayerWeight
    transformer_weight_class = Gemma3TransformerLayerWeight

    # infer class
    pre_layer_infer_class = Gemma3PreLayerInfer
    transformer_layer_infer_class = Gemma3TransformerLayerInfer
    post_layer_infer_class = Gemma3PostLayerInfer

    infer_state_class = Gemma3InferStateInfo

    def __init__(self, kvargs):
        self.head_dim_ = 256
        super().__init__(kvargs)
        return

    def _init_to_get_rotary(self, default_base=10000.0):
        partial_head_dim = int(self.config.get("partial_rotary_factor", 1) * self.head_dim_)
        if self.config.get("rope_scaling", {}) is None:
            rope_scaling_factor = 1.0
        else:
            rope_scaling_factor = self.config.get("rope_scaling", {}).get("factor", 1.0)

        if "max_sequence_length" in self.config:
            max_seq_len = self.config["max_sequence_length"]
        else:
            max_position_embeddings = self.config.get("max_position_embeddings", 16384)
            max_seq_len = max_position_embeddings * rope_scaling_factor

        inv_freq_local = 1.0 / (
            10000.0 ** (torch.arange(0, partial_head_dim, 2, dtype=torch.int64).float().cuda() / partial_head_dim)
        )
        inv_freq_global = (
            1.0
            / (1000000.0 ** (torch.arange(0, partial_head_dim, 2, dtype=torch.int64).float().cuda() / partial_head_dim))
            / rope_scaling_factor
        )
        # local default
        # global linear
        # print(inv_freq_local, inv_freq_global, partial_head_dim)
        t = torch.arange(max(max_seq_len + 1024 * 128, self.max_seq_length), dtype=torch.float32).to(
            inv_freq_local.device
        )

        freqs_global = torch.outer(t, inv_freq_global)
        freqs_local = torch.outer(t, inv_freq_local)

        self._cos_cached = torch.cos(freqs_global).to(torch.float32).cuda()
        self._sin_cached = torch.sin(freqs_global).to(torch.float32).cuda()

        self._cos_cached_global = torch.cos(freqs_global).to(torch.float32).cuda()
        self._sin_cached_global = torch.sin(freqs_global).to(torch.float32).cuda()

        self._cos_cached_local = torch.cos(freqs_local).to(torch.float32).cuda()
        self._sin_cached_local = torch.sin(freqs_local).to(torch.float32).cuda()
        return

    def _init_custom(self):
        self.head_dim_ = 256
        self._init_to_get_rotary()

    def _init_mem_manager(self):
        self.mem_manager = select_mem_manager_class()(
            self.max_total_token_num,
            dtype=torch.bfloat16,
            head_num=self.config["num_key_value_heads"] // self.tp_world_size_,
            head_dim=256,
            layer_num=self.config["num_hidden_layers"],
            mem_fraction=self.mem_fraction,
        )
        return

    def _init_config(self):
        with open(os.path.join(self.weight_dir_, "config.json"), "r") as json_file:
            self.config = json.load(json_file)
        # rename keys
        if "text_config" in self.config:
            config = AutoConfig.from_pretrained(self.weight_dir_, trust_remote_code=True)
            self.config = config.text_config.to_dict()

        repair_config(self.config, same_names=["num_attention_heads", "n_head"])
        repair_config(self.config, same_names=["hidden_size", "n_embd", "n_embed"])
        repair_config(self.config, same_names=["num_hidden_layers", "n_layer"])
        return
