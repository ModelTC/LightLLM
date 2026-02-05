import torch
import numpy as np
from typing import TYPE_CHECKING, Any, Optional, Union, Tuple
from transformers.audio_utils import mel_filter_bank, spectrogram, window_function
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature
from transformers.utils import TensorType


class WhisperFeatureExtractor(SequenceFeatureExtractor):

    model_input_names = ["input_features"]

    def __init__(
        self,
        feature_size=80,
        sampling_rate=16000,
        hop_length=160,
        chunk_length=30,
        n_fft=400,
        padding_value=0.0,
        dither=0.0,
        return_attention_mask=False,  # pad inputs to max length with silence token (zero) and no attention mask
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.n_samples = chunk_length * sampling_rate
        self.nb_max_frames = self.n_samples // hop_length
        self.sampling_rate = sampling_rate
        self.dither = dither
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=1 + n_fft // 2,
            num_mel_filters=feature_size,
            min_frequency=0.0,
            max_frequency=8000.0,
            sampling_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )

    def _torch_extract_fbank_features(self, waveform: np.ndarray, device: str = "cpu") -> np.ndarray:
        waveform = torch.from_numpy(waveform).to(device, torch.float32)
        window = torch.hann_window(self.n_fft, device=device)

        if self.dither != 0.0:
            waveform += self.dither * torch.randn(waveform.shape, dtype=waveform.dtype, device=waveform.device)

        stft = torch.stft(waveform, self.n_fft, self.hop_length, window=window, return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2

        mel_filters = torch.from_numpy(self.mel_filters).to(device, torch.float32)
        mel_spec = mel_filters.T @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        if waveform.dim() == 2:
            max_val = log_spec.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
            log_spec = torch.maximum(log_spec, max_val - 8.0)
        else:
            log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        if device != "cpu":
            log_spec = log_spec.detach().cpu()
        return log_spec.numpy()

    # Copied from transformers.models.wav2vec2.feature_extraction_wav2vec2.
    # Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm
    def zero_mean_unit_var_norm(
        self, input_values: list[np.ndarray], attention_mask: list[np.ndarray], padding_value: float = 0.0
    ) -> list[np.ndarray]:
        if attention_mask is not None:
            attention_mask = np.array(attention_mask, np.int32)
            normed_input_values = []

            for vector, length in zip(input_values, attention_mask.sum(-1)):
                normed_slice = (vector - vector[:length].mean()) / np.sqrt(vector[:length].var() + 1e-7)
                if length < normed_slice.shape[0]:
                    normed_slice[length:] = padding_value

                normed_input_values.append(normed_slice)
        else:
            normed_input_values = [(x - x.mean()) / np.sqrt(x.var() + 1e-7) for x in input_values]

        return normed_input_values

    def _preprocess(
        self,
        raw_speech: Union[np.ndarray, list[float], list[np.ndarray], list[list[float]]],
        truncation: bool = True,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_attention_mask: Optional[bool] = None,
        padding: Optional[str] = "max_length",
        max_length: Optional[int] = None,
        sampling_rate: Optional[int] = None,
        do_normalize: Optional[bool] = None,
        device: Optional[str] = "cpu",
        return_token_timestamps: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        is_batched_numpy = isinstance(raw_speech, np.ndarray) and len(raw_speech.shape) > 1
        if is_batched_numpy and len(raw_speech.shape) > 2:
            raise ValueError(f"Only mono-channel audio is supported for input to {self}")
        is_batched = is_batched_numpy or (
            isinstance(raw_speech, (list, tuple)) and (isinstance(raw_speech[0], (np.ndarray, tuple, list)))
        )

        if is_batched:
            raw_speech = [np.asarray([speech], dtype=np.float32).T for speech in raw_speech]
        elif not is_batched and not isinstance(raw_speech, np.ndarray):
            raw_speech = np.asarray(raw_speech, dtype=np.float32)
        elif isinstance(raw_speech, np.ndarray) and raw_speech.dtype is np.dtype(np.float64):
            raw_speech = raw_speech.astype(np.float32)

        # always return batch
        if not is_batched:
            raw_speech = [np.asarray([raw_speech]).T]

        batched_speech = BatchFeature({"input_features": raw_speech})

        # convert into correct format for padding

        padded_inputs = self.pad(
            batched_speech,
            padding=padding,
            max_length=max_length if max_length else self.n_samples,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask or do_normalize,
        )

        # zero-mean and unit-variance normalization
        if do_normalize:
            padded_inputs["input_features"] = self.zero_mean_unit_var_norm(
                padded_inputs["input_features"],
                attention_mask=padded_inputs["attention_mask"],
                padding_value=self.padding_value,
            )
            padded_inputs["input_features"] = np.stack(padded_inputs["input_features"], axis=0)

        # make sure list is in array format
        input_features = padded_inputs.get("input_features").transpose(2, 0, 1)

        input_features = self._torch_extract_fbank_features(input_features[0], device)

        if isinstance(input_features[0], list):
            padded_inputs["input_features"] = [np.asarray(feature, dtype=np.float32) for feature in input_features]

        else:
            padded_inputs["input_features"] = input_features

        if return_attention_mask:
            # rescale from sample (48000) to feature (3000)
            rescaled_attention_mask = padded_inputs["attention_mask"][:, :: self.hop_length]

            # The STFT computation produces L//hop_length + 1 frames,
            # but we skip the last frame (see `_torch_extract_fbank_features`).
            # This means we need to trim the rescaled attention mask to match
            # the actual number of frames (L//hop_length) when the input length
            # is not perfectly divisible by the hop length.
            if padded_inputs["attention_mask"].shape[1] % self.hop_length != 0:
                rescaled_attention_mask = rescaled_attention_mask[:, :-1]
            padded_inputs["attention_mask"] = rescaled_attention_mask

        if return_token_timestamps is not None:
            padded_inputs["num_frames"] = [len(raw_speech_i) // self.hop_length for raw_speech_i in raw_speech]

        if return_tensors is not None:
            padded_inputs = padded_inputs.convert_to_tensors(return_tensors)

        return padded_inputs["input_features"], padded_inputs["attention_mask"]
