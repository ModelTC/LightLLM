"""Multimodal parameters for text generation."""
import os
import librosa
import base64
from typing import List
from io import BytesIO
from PIL import Image
from fastapi import Request
from lightllm.utils.multimodal_utils import fetch_resource
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class AudioItem:
    def __init__(self, **kwargs):
        self._type = kwargs["type"]
        self._data = kwargs["data"]
        # the unique id for the image
        self.uuid = None
        # the start audio token id
        self.token_id = None
        # the audio token num
        self.token_num = None
        # the audio length
        self.audio_length = None

        self._preload_data = None
        self.extra_params = {}

    async def preload(self, request: Request):
        try:
            if self._type == "url":
                timeout = int(os.getenv("REQUEST_TIMEOUT", "5"))
                proxy = os.getenv("REQUEST_PROXY", None)
                audio_data = await fetch_resource(self._data, request, timeout=timeout, proxy=proxy)
            elif self._type == "base64":
                audio_data = base64.b64decode(self._data)
            else:
                raise ValueError(f"cannot read audio which type is {self._type}!")

            # check if valid audio bytes
            audio_values, _ = librosa.load(BytesIO(audio_data), sr=16000)
            from lightllm.models.whisper.defaults import MIN_AUDIO_LEN

            self.audio_length = max(audio_values.shape[0], MIN_AUDIO_LEN)  # 如果音频过短，会被pad到480的长度
            self._preload_data = audio_data
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
        return ret


class ImageItem:
    def __init__(self, **kwargs):
        self._type = kwargs["type"]
        self._data = kwargs["data"]
        # the unique id for the image
        self.uuid = None
        # the start image token id
        self.token_id = None
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

    async def verify_and_preload(self, request: Request):
        for image in self.images:
            await image.preload(request)
        for audio in self.audios:
            await audio.preload(request)
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
        return ret
