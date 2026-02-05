import os
import json
import librosa
from io import BytesIO
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


def _get_feat_extract_output_lengths(input_lengths):
    """
    Computes the output length of the convolutional layers and the output length of the audio encoder
    """

    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    return output_lengths


# <|audio_start|><|audio_pad|><|audio_end|>
AUDIO_START_TOKEN = "<|audio_start|>"
AUDIO_END_TOKEN = "<|audio_end|>"
AUDIO_TOKEN_TOKEN = "<|audio_pad|>"
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

        self.audio_token_tag = AUDIO_TOKEN_TOKEN
        self.audio_token_id = tokenizer.convert_tokens_to_ids(self.audio_token_tag)

        # 这些太hard了, 后面改一下,可以直接从audio_processor里取?
        self.sampling_rate = 16000
        self.chunk_length = 30
        self.n_samples = self.chunk_length * self.sampling_rate
        self.hop_length = 160

    def init_audioitem_extral_params(
        self, audio: AudioItem, multi_params: MultimodalParams, sampling_params: SamplingParams
    ):
        return

    def get_audio_token_length(self, audio: AudioItem):
        # audio_bytes = audio._preload_data
        # audio_values, _ = librosa.load(BytesIO(audio_bytes), sr=self.sampling_rate)
        # length = max(int(audio_values.shape[0]), int(MIN_AUDIO_LEN)) #这个最短还有必要吗?稍等再检查一下
        # L_eff = min(length, int(self.n_samples))
        # num_frames = L_eff // int(self.hop_length)

        return 290

    def encode(self, prompt, multimodal_params: MultimodalParams = None, **kwargs):
        origin_ids = self.tokenizer.encode(prompt)

        # <img><image_pad></img> -> <img></img>
        origin_ids = [token for token in origin_ids if token not in (self.image_token_id, self.audio_token_id)]
        # <img></img> --> <img>id,id+1...id+num</img>
        input_ids = []
        image_id = 0
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
        if multimodal_params:
            image_cnt = len(multimodal_params.images)
            if image_cnt != image_id:
                raise ValueError(image_cnt == image_id, f"invalid image tag num: {image_cnt} vs {image_id}!")
        input_ids.extend(origin_ids)

        # audio
        origin_ids = input_ids
        input_ids = []
        audio_id = 0
        start_idx = 0
        while True:
            try:
                start_idx = origin_ids.index(self.audio_start_id)
                if start_idx + 1 >= len(origin_ids):
                    break
                if origin_ids[start_idx + 1] == self.audio_end_id:
                    input_ids.extend(origin_ids[: start_idx + 1])
                    token_id = multimodal_params.audios[audio_id].token_id
                    token_num = multimodal_params.audios[audio_id].token_num
                    input_ids.extend(range(token_id, token_id + token_num))
                    input_ids.append(self.audio_end_id)
                    origin_ids = origin_ids[start_idx + 2 :]
                    audio_id += 1
                else:
                    raise ValueError("audio token error")
            except ValueError:
                break
        if multimodal_params:
            audio_cnt = len(multimodal_params.audios)
            if audio_cnt != audio_id:
                raise ValueError(audio_cnt == audio_id, f"invalid audio tag num: {audio_cnt} vs {audio_id}!")
        input_ids.extend(origin_ids)

        return input_ids


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
