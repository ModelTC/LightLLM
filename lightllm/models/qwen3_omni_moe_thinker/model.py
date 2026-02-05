import os
import json
from lightllm.common.build_utils import repair_config
from lightllm.models.registry import ModelRegistry
from lightllm.models.qwen3_moe.model import Qwen3MOEModel
from lightllm.models.qwen3_vl.layer_infer.pre_layer_infer import Qwen3VLMultimodalPreLayerInfer

from lightllm.models.qwen3_omni_moe_thinker.layer_infer.transformer_layer_infer import Qwen3OmniMOETransformerLayerInfer
from lightllm.models.qwen3_omni_moe_thinker.layer_weights.pre_and_post_layer_weight import (
    Qwen3OmniMOEThinkerPreAndPostLayerWeight,
)
from lightllm.models.qwen3_omni_moe_thinker.layer_weights.transformers_layer_weight import (
    Qwen3OmniMOEThinkerTransformerLayerWeight,
)

from lightllm.models.qwen3_vl_moe.model import Qwen3VLMOETpPartModel
from lightllm.models.qwen3_omni_moe_thinker.infer_struct import Qwen3OmniMOEInferStateInfo
from lightllm.models.qwen3_vl.model import QWen3VLTokenizer
from lightllm.server.core.objs import SamplingParams
from lightllm.server.multimodal_params import AudioItem, MultimodalParams, ImageItem


# <|audio_start|><|audio_pad|><|audio_end|>
AUDIO_START_TOKEN = "<|audio_start|>"
AUDIO_END_TOKEN = "<|audio_end|>"

MIN_AUDIO_LEN = 480


class QWen3OmniTokenizer(QWen3VLTokenizer):
    def __init__(self, tokenizer=None, image_processor=None, **kwargs):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.min_pixel = self.image_processor.min_pixels
        self.max_pixel = self.image_processor.max_pixels
        self.patch_size = self.image_processor.patch_size
        self.merge_size = self.image_processor.merge_size
        self.image_start_id = kwargs["model_cfg"]["vision_start_token_id"]
        self.image_end_id = kwargs["model_cfg"]["vision_end_token_id"]
        self.image_token_id = kwargs["model_cfg"]["image_token_id"]

        self.audio_start_tag = AUDIO_START_TOKEN
        self.audio_start_id = tokenizer.convert_tokens_to_ids(self.audio_start_tag)

        self.audio_end_tag = AUDIO_END_TOKEN
        self.audio_end_id = tokenizer.convert_tokens_to_ids(self.audio_end_tag)

        self.audio_min_length = MIN_AUDIO_LEN
        self.audio_max_length = 16000 * 30

    def init_audioitem_extral_params(
        self, audio: AudioItem, multi_params: MultimodalParams, sampling_params: SamplingParams
    ):
        return

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


@ModelRegistry(["qwen3_omni_moe"], is_multimodal=True)
class Qwen3OmniMOETpPartModel(Qwen3VLMOETpPartModel):

    pre_layer_infer_class = Qwen3VLMultimodalPreLayerInfer
    transformer_layer_infer_class = Qwen3OmniMOETransformerLayerInfer

    pre_and_post_weight_class = Qwen3OmniMOEThinkerPreAndPostLayerWeight
    transformer_weight_class = Qwen3OmniMOEThinkerTransformerLayerWeight

    infer_state_class = Qwen3OmniMOEInferStateInfo

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return

    def _init_config(self):
        with open(os.path.join(self.weight_dir_, "config.json"), "r") as json_file:
            all_config = json.load(json_file)
            self.config = all_config["thinker_config"]["text_config"]
        # rename keys
        print(f"self.config is {self.config}")
        repair_config(self.config, same_names=["num_attention_heads", "n_head"])
        repair_config(self.config, same_names=["hidden_size", "n_embd", "n_embed"])
        repair_config(self.config, same_names=["num_hidden_layers", "n_layer"])
        if self.finetune_config:
            self.config["vocab_size"] = self.finetune_config.vocab_size
        return
