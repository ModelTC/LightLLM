import os
import json
import rpyc
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from io import BytesIO
from typing import List, Union
from safetensors.torch import load_file
from transformers.processing_utils import ProcessorMixin
from lightllm.server.embed_cache.utils import tensor2bytes, read_shm, create_shm, get_shm_name_data, get_shm_name_embed
from lightllm.server.multimodal_params import AudioItem


# tokenizer_class removed
class WhisperProcessor(ProcessorMixin):
    r"""
    Constructs a Whisper processor which wraps a Whisper feature extractor and a Whisper tokenizer into a single
    processor.

    [`WhisperProcessor`] offers all the functionalities of [`WhisperFeatureExtractor`] and [`WhisperTokenizer`]. See
    the [`~WhisperProcessor.__call__`] and [`~WhisperProcessor.decode`] for more information.

    Args:
        feature_extractor (`WhisperFeatureExtractor`):
            An instance of [`WhisperFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`WhisperTokenizer`):
            An instance of [`WhisperTokenizer`]. The tokenizer is a required input.
    """
    attributes = ["feature_extractor"]
    feature_extractor_class = "WhisperFeatureExtractor"

    def __init__(self, feature_extractor):
        super().__init__(feature_extractor)
        self.current_processor = self.feature_extractor
        self._in_target_context_manager = False

    def get_decoder_prompt_ids(self, task=None, language=None, no_timestamps=True):
        return self.tokenizer.get_decoder_prompt_ids(task=task, language=language, no_timestamps=no_timestamps)

    def get_T_after_cnn(self, L_in, dilation=1):
        for (padding, kernel_size, stride) in eval("[(1,3,1)] + [(1,3,2)] "):
            L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
            L_out = 1 + L_out // stride
            L_in = L_out
        return L_out

    def __call__(self, audios, audio_lens, *args, **kwargs):
        """
        Forwards the `audios` argument to WhisperFeatureExtractor's [`~WhisperFeatureExtractor.__call__`] and the `text`
        argument to [`~WhisperTokenizer.__call__`]. Please refer to the doctsring of the above two methods for more
        information.
        """
        # For backward compatibility
        if self._in_target_context_manager:
            return self.current_processor(*args, **kwargs)

        sampling_rate = kwargs.pop("sampling_rate", 16000)

        audio_lens = np.where(audio_lens <= 480000, audio_lens, 480000)
        audio_lens = audio_lens // 160
        audio_lens_after_cnn = self.get_T_after_cnn(audio_lens)
        padded_inputs = self.feature_extractor(audios, *args, sampling_rate=sampling_rate, **kwargs)

        return padded_inputs["input_features"], audio_lens_after_cnn

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to WhisperTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to WhisperTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    def get_prompt_ids(self, text: str, return_tensors="np"):
        return self.tokenizer.get_prompt_ids(text, return_tensors=return_tensors)


class WhisperAudioModel:
    def __init__(self, kvargs):
        self.max_seconds = 30
        self.sampling_rate = 16000
        self.max_length = self.max_seconds * self.sampling_rate
        self.cache_port = kvargs["cache_port"]
        self.cache_client = rpyc.connect("localhost", self.cache_port)
        data_type = kvargs["data_type"]
        if data_type in ["bf16", "bfloat16"]:
            self.data_type = torch.bfloat16
        else:
            self.data_type = torch.float16

    def cuda(self):
        self.audio = self.audio.cuda()
        for k, v in self.projector_weights.items():
            self.projector_weights[k] = v.cuda()
        self.device = torch.device("cuda")
        return self

    def load_model(self, weight_dir, config):
        self.audio_processor = WhisperProcessor.from_pretrained(weight_dir)
        from transformers.models.whisper.modeling_whisper import WhisperEncoder, WhisperConfig

        self.audio = WhisperEncoder(WhisperConfig(**config["audio_config"])).to(self.data_type)
        self.device = torch.device("cpu")
        self.projector_weights = {}
        self.load_weight(weight_dir)

    def load_weight(self, weight_dir):
        weight_path = os.path.join(weight_dir, "model.safetensors.index.json")
        weight_map = json.load(open(weight_path, "r"))["weight_map"]
        params_map = {}
        audio_weight = {}
        for k, v in weight_map.items():
            if "mlp2" not in k and "audio_model" not in k:
                continue
            filename = weight_map[k]
            if filename not in params_map:
                tensor_data = load_file(os.path.join(weight_dir, filename))
                params_map[filename] = tensor_data
            if "mlp2" in k:
                self.projector_weights[k] = params_map[filename][k].to(self.data_type)
            if "audio_model" in k:
                audio_weight[k[len("audio_model.encoder.") :]] = params_map[filename][k].to(self.data_type)

        self.audio.load_state_dict(audio_weight)

        assert "mlp2.0.bias" in self.projector_weights
        assert "mlp2.0.weight" in self.projector_weights
        assert "mlp2.1.bias" in self.projector_weights
        assert "mlp2.1.weight" in self.projector_weights
        assert "mlp2.3.bias" in self.projector_weights
        assert "mlp2.3.weight" in self.projector_weights

    def forward(self, audio_values, audio_lens_after_cnn):
        audio_values = audio_values.to(self.data_type).to(device=self.device)
        audio_values = audio_values.squeeze(1)
        audio_lens_after_cnn = torch.tensor(audio_lens_after_cnn).cuda()
        max_len_in_batch = torch.max(audio_lens_after_cnn).item()

        padding_mask = torch.ones([audio_values.size(0), max_len_in_batch]).to(
            dtype=audio_values.dtype, device=audio_values.device
        )
        for index in range(len(audio_values)):
            padding_mask[index, : audio_lens_after_cnn[index].item()] = 0
        last_hidden_state = self.audio(audio_values, padding_mask).last_hidden_state
        x = F.layer_norm(
            last_hidden_state,
            normalized_shape=(last_hidden_state.shape[-1],),
            weight=self.projector_weights["mlp2.0.weight"],
            bias=self.projector_weights["mlp2.0.bias"],
        )
        x = F.linear(x, weight=self.projector_weights["mlp2.1.weight"], bias=self.projector_weights["mlp2.1.bias"])
        x = F.gelu(x)
        x = F.linear(x, weight=self.projector_weights["mlp2.3.weight"], bias=self.projector_weights["mlp2.3.bias"])
        return x

    def encode(self, audio_items: List[AudioItem]):
        batch_audios = []
        batch_audio_lens = np.zeros(len(audio_items), dtype=np.int32)
        uuids = []
        for i, item in enumerate(audio_items):
            if isinstance(item, AudioItem):
                uuids.append(item.uuid)
                audio_data = read_shm(get_shm_name_data(item.uuid))
                audio = BytesIO(audio_data)
                audio, _ = librosa.load(audio, sr=16000)
            else:
                raise ValueError(f"cannot read audio which type is {type(item)}!")

            # padding to min audio len
            from .defaults import MIN_AUDIO_LEN

            if audio.shape[0] < MIN_AUDIO_LEN:
                audio = np.pad(audio, (0, MIN_AUDIO_LEN - len(audio)), mode="constant", constant_values=0.0)

            batch_audio_lens[i] = min(audio.shape[0], self.max_length)
            batch_audios.append(audio)

        audios, audio_lens_after_cnn = self.audio_processor(
            batch_audios, batch_audio_lens, sampling_rate=16000, return_tensors="pt"
        )
        audios = self.forward(audios, audio_lens_after_cnn)
        audio_lens_after_cnn = np.array(audio_lens_after_cnn, dtype=np.int32)
        audio_token_num = (audio_lens_after_cnn - 2) // 2 + 1

        for i in range(len(uuids)):
            if not self.cache_client.root.get_item_embed(uuids[i]):
                cur_embed_bytes = tensor2bytes(audios[i][: audio_token_num[i]])
                create_shm(get_shm_name_embed(uuids[i]), cur_embed_bytes)
                self.cache_client.root.set_item_embed(uuids[i])
