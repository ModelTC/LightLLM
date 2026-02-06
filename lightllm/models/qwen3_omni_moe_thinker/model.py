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


MIN_AUDIO_LEN = 480


class QWen3OmniTokenizer(QWen3VLTokenizer):
    def __init__(self, tokenizer=None, processor=None, **kwargs):
        self.tokenizer = tokenizer
        # image
        self.image_processor = processor.image_processor
        self.min_pixel = self.image_processor.min_pixels
        self.max_pixel = self.image_processor.max_pixels
        self.patch_size = self.image_processor.patch_size
        self.merge_size = self.image_processor.merge_size

        # audio
        self.audio_processor = processor.feature_extractor
        self.sampling_rate = self.audio_processor.sampling_rate
        self.n_samples = self.audio_processor.n_samples
        self.hop_length = self.audio_processor.hop_length

        self.image_start_id = kwargs["model_cfg"]["vision_start_token_id"]
        self.image_end_id = kwargs["model_cfg"]["vision_end_token_id"]
        self.image_token_id = kwargs["model_cfg"]["image_token_id"]

        self.audio_start_id = kwargs["model_cfg"]["audio_start_token_id"]
        self.audio_end_id = kwargs["model_cfg"]["audio_end_token_id"]
        self.audio_token_id = kwargs["model_cfg"]["audio_token_id"]

    def init_audioitem_extral_params(
        self, audio: AudioItem, multi_params: MultimodalParams, sampling_params: SamplingParams
    ):
        return

    def get_audio_token_length(self, audio: AudioItem):
        # 这里得处理对应奖语音长度按照 30 进行限制，后续处理中，超过30的会被截断。
        length = min(audio.audio_length, int(self.n_samples))
        token_num = self._caclu_audio_token_num(length)
        # print(f"token_num is {token_num}  n_samples is {self.n_samples} hop_length is {self.hop_length}")
        return token_num

    def _caclu_audio_token_num(self, input_audio_len: int):
        _mel_len = input_audio_len // int(self.hop_length)
        input_lengths_leave = _mel_len % 100
        feat_lengths = (input_lengths_leave - 1) // 2 + 1
        output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (_mel_len // 100) * 13
        return output_lengths

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
