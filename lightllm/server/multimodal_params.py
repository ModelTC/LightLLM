"""Multimodal parameters for text generation."""
from typing import List
import os
import requests
from io import BytesIO
from PIL import Image
from lightllm.utils.image_utils import fetch_image
import base64


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
        self.image_w = 0
        self.image_h = 0

        self._preload_data = None
        self.extra_params = {}

    async def preload(self):
        try:
            if self._type == "url":
                timeout = int(os.getenv("REQUEST_TIMEOUT", "5"))
                proxy = os.getenv("REQUEST_PROXY", None)
                img_data = await fetch_image(self._data, timeout=timeout, proxy=proxy)
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

            # check if valid image bytes
            image = Image.open(BytesIO(img_data))
            self.image_w, self.image_h = image.size
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
    ) -> None:
        self.images = [ImageItem(**i) for i in images]
        return

    async def verify_and_preload(self):
        for image in self.images:
            await image.preload()
        return

    def to_dict(self):
        ret = {}
        ret["images"] = [i.to_dict() for i in self.images]
        return ret

    def to_origin_dict(self):
        """
        将内容转换为原始请求的形式，主要用于请求转发
        """
        ret = {}
        ret["images"] = [i.to_origin_dict() for i in self.images]
        return ret
