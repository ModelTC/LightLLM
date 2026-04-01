"""Multimodal parameters for text generation."""
import os
import wave
import time
import librosa
import base64
import hashlib
import numpy as np
from typing import List
from io import BytesIO
from PIL import Image
from fastapi import Request
from lightllm.utils.multimodal_utils import fetch_resource
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)
RAW_AUDIO_SHM_FORMAT = "raw_audio_bytes"
WAVEFORM_F32_SHM_FORMAT = "waveform_f32"
AUDIO_SHM_USE_RAW_ENV = "LIGHTLLM_AUDIO_SHM_USE_RAW"
DEFAULT_AUDIO_SAMPLE_RATE = 16000
DEFAULT_AUDIO_HOP_LENGTH = 160
DEFAULT_MIN_AUDIO_LEN = 480


def generate_silence_wav_bytes(sample_rate: int = 16000, duration_seconds: float = 1.0) -> bytes:
    num_samples = max(1, int(sample_rate * duration_seconds))
    with BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(b"\x00\x00" * num_samples)
        return buffer.getvalue()


def load_audio_from_shm_payload(audio_data: bytes, extra_params: dict, sample_rate: int) -> np.ndarray:
    audio_shm_format = extra_params.get("audio_shm_format", RAW_AUDIO_SHM_FORMAT)
    if audio_shm_format == WAVEFORM_F32_SHM_FORMAT:
        num_samples = int(extra_params.get("audio_num_samples", 0))
        if num_samples > 0:
            return np.frombuffer(audio_data, dtype=np.float32, count=num_samples)
        return np.frombuffer(audio_data, dtype=np.float32)

    audio, _ = librosa.load(BytesIO(audio_data), sr=sample_rate)
    return np.asarray(audio, dtype=np.float32)


def should_use_raw_audio_shm() -> bool:
    return os.getenv(AUDIO_SHM_USE_RAW_ENV, "0") == "1"


class AudioItem:
    def __init__(self, **kwargs):
        self._type = kwargs["type"]
        self._data = kwargs["data"]
        # the unique id for the image
        self.uuid = None
        # the start audio token id
        self.token_id = None
        # the start index in embed cache
        self.start_index_in_embed_cache = None
        # the audio token num
        self.token_num = None
        # the audio length
        self.audio_length = None

        self._preload_data = None
        self.extra_params = {}

    async def preload(self, request: Request, audio_preload_config: dict = None):
        try:
            req_id = getattr(getattr(request, "state", None), "lightllm_req_id", None)
            preload_start = time.time()
            source_ready_start = preload_start
            if self._type == "url":
                timeout = int(os.getenv("REQUEST_TIMEOUT", "5"))
                proxy = os.getenv("REQUEST_PROXY", None)
                audio_data = await fetch_resource(self._data, request, timeout=timeout, proxy=proxy)
            elif self._type == "base64":
                audio_data = base64.b64decode(self._data)
            else:
                raise ValueError(f"cannot read audio which type is {self._type}!")
            source_ready_cost_ms = (time.time() - source_ready_start) * 1000.0

            audio_preload_config = audio_preload_config or {}
            target_sample_rate = int(audio_preload_config.get("sampling_rate", DEFAULT_AUDIO_SAMPLE_RATE))
            hop_length = int(audio_preload_config.get("hop_length", DEFAULT_AUDIO_HOP_LENGTH))
            min_audio_len = int(audio_preload_config.get("min_audio_len", DEFAULT_MIN_AUDIO_LEN))

            # check if valid audio bytes
            decode_start = time.time()
            audio_values, _ = librosa.load(BytesIO(audio_data), sr=target_sample_rate)
            audio_values = np.asarray(audio_values, dtype=np.float32)
            decode_cost_ms = (time.time() - decode_start) * 1000.0
            effective_audio_len = max(audio_values.shape[0], min_audio_len)
            padded_audio_len = ((effective_audio_len + hop_length - 1) // hop_length) * hop_length
            if padded_audio_len > audio_values.shape[0]:
                audio_values = np.pad(
                    audio_values,
                    (0, padded_audio_len - audio_values.shape[0]),
                    mode="constant",
                    constant_values=0.0,
                )

            self.audio_length = effective_audio_len
            if should_use_raw_audio_shm():
                self.extra_params["audio_shm_format"] = RAW_AUDIO_SHM_FORMAT
                self.extra_params.pop("audio_sample_rate", None)
                self.extra_params.pop("audio_num_samples", None)
                self.extra_params.pop("audio_num_frames", None)
                self._preload_data = audio_data
            else:
                self.extra_params["audio_shm_format"] = WAVEFORM_F32_SHM_FORMAT
                self.extra_params["audio_sample_rate"] = target_sample_rate
                self.extra_params["audio_num_samples"] = int(audio_values.shape[0])
                self.extra_params["audio_num_frames"] = int(effective_audio_len // hop_length)
                self._preload_data = audio_values.tobytes()
            self.extra_params["audio_payload_md5"] = hashlib.md5(self._preload_data).hexdigest()
            logger.info(
                f"lightllm_req_id:{req_id} stage:audio_preload_done "
                f"elapsed_ms:{(time.time() - preload_start) * 1000.0:.3f} "
                f"source_type:{self._type} source_ready_ms:{source_ready_cost_ms:.3f} "
                f"decode_ms:{decode_cost_ms:.3f} audio_length:{self.audio_length} "
                f"shm_format:{self.extra_params['audio_shm_format']}"
            )
            return

        except Exception as e:
            raise ValueError(f"Failed to read audio type={self._type}, data[:100]={self._data[:100]}: {e}!")

    def read(self):
        assert self._preload_data is not None
        ans = self._preload_data
        self._preload_data = None
        self._data = None
        return ans

    def to_dict(self):
        ret = {}
        ret["uuid"] = self.uuid
        ret["token_id"] = self.token_id
        ret["token_num"] = self.token_num
        ret["start_index_in_embed_cache"] = self.start_index_in_embed_cache
        return ret

    def to_origin_dict(self):
        """
        将内容转换为原始请求的形式，主要用于请求转发
        """
        ret = {}
        ret["type"] = self._type
        ret["data"] = self._data
        return ret


class ImageItem:
    def __init__(self, **kwargs):
        self._type = kwargs["type"]
        self._data = kwargs["data"]
        # the unique id for the image
        self.uuid = None
        # the start image token id
        self.token_id = None
        # the start index in embed cache
        self.start_index_in_embed_cache = None
        # the image token num
        self.token_num = None
        # the start index of the image in the input_ids
        # used for mrope position id calculation
        self.start_idx = None
        self.grid_thwd = None
        self.image_w = 0
        self.image_h = 0

        self._preload_data = None
        self.extra_params = {}

    async def preload(self, request: Request):
        try:
            if self._type == "url":
                timeout = int(os.getenv("REQUEST_TIMEOUT", "5"))
                proxy = os.getenv("REQUEST_PROXY", None)
                img_data = await fetch_resource(self._data, request, timeout=timeout, proxy=proxy)
            elif self._type == "base64":
                img_data = base64.b64decode(self._data)
            elif self._type == "image_size":
                # image_size 代表直接传入图片的 width，height，主要是用于一些场景
                # 的 token 计数判断, 所以只需要图片长宽信息，不需要具体图片的内容信息
                self.image_w = self._data[0]
                self.image_h = self._data[1]
                return
            else:
                raise ValueError(f"cannot read image which type is {self._type}!")

            with Image.open(BytesIO(img_data)) as image:
                self.image_w, self.image_h = image.size
                image.verify()  # verify后会失效

            self._preload_data = img_data
            return

        except Exception as e:
            raise ValueError(f"Failed to read image type={self._type}, data[:100]={self._data[:100]}: {e}!")

    def read(self):
        assert self._preload_data is not None
        ans = self._preload_data
        self._preload_data = None
        self._data = None
        return ans

    def to_dict(self):
        ret = {}
        ret["uuid"] = self.uuid
        ret["token_id"] = self.token_id
        ret["start_index_in_embed_cache"] = self.start_index_in_embed_cache
        ret["token_num"] = self.token_num
        ret["grid_thwd"] = self.grid_thwd
        ret["start_idx"] = self.start_idx
        return ret

    def to_origin_dict(self):
        """
        将内容转换为原始请求的形式，主要用于请求转发
        """
        ret = {}
        ret["type"] = self._type
        ret["data"] = self._data
        return ret


class MultimodalParams:
    def __init__(
        self,
        images: List[dict] = [],
        audios: List[dict] = [],
    ) -> None:
        self.images = [ImageItem(**i) for i in images]
        self.audios = [AudioItem(**a) for a in audios]
        return

    async def verify_and_preload(self, request: Request, audio_preload_config: dict = None):
        for image in self.images:
            await image.preload(request)
        for audio in self.audios:
            await audio.preload(request, audio_preload_config=audio_preload_config)
        return

    def to_dict(self):
        ret = {}
        ret["images"] = [i.to_dict() for i in self.images]
        ret["audios"] = [a.to_dict() for a in self.audios]
        return ret

    def to_origin_dict(self):
        """
        将内容转换为原始请求的形式，主要用于请求转发
        """
        ret = {}
        ret["images"] = [i.to_origin_dict() for i in self.images]
        ret["audios"] = [a.to_origin_dict() for a in self.audios]
        return ret


async def warmup_audio_preload():
    warmup_audio = AudioItem(
        type="base64",
        data=base64.b64encode(generate_silence_wav_bytes()).decode("utf-8"),
    )
    await warmup_audio.preload(None)
    warmup_audio.read()
    return
