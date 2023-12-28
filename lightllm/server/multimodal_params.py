"""Multimodal parameters for text generation."""
from typing import List
import os
import requests
from io import BytesIO
from PIL import Image
import base64


class ImageItem:

    def __init__(self, **kwargs):
        self._type = kwargs["type"]
        self._data = kwargs["data"]
        # the unique id for the image 
        self.uuid = None
        # where should the image fill into the text embeds
        self.offset = -1
        # the length of the image embeds
        self.length = -1

    def read(self):
        try:
            if self._type == "url":
                timeout = int(os.getenv("REQUEST_TIMEOUT", "3"))
                ret = requests.get(self._data, timeout=timeout)
                img_data = ret.content
            elif self._type == "base64":
                img_data = base64.b64decode(self._data)
            else:
                raise ValueError(f"cannot read image which type is {self._type}!")
    
            # check if valid image bytes
            image = Image.open(BytesIO(img_data))
            return img_data
    
        except Exception as e:
            raise ValueError(f"Failed to read image type={self._type}, data[:100]={self._data[:100]}: {e}!")

    def to_dict(self):
        ret = {}
        ret["uuid"] = self.uuid
        ret["offset"] = self.offset
        ret["length"] = self.length
        return ret


class MultimodalParams:

    def __init__(
        self,
        images: List[dict] = [],
    ) -> None:
        self.images = [ImageItem(**i) for i in images]
        return

    def after_tokenize(self, prompt_ids):
        if not isinstance(prompt_ids, dict):
            return prompt_ids
        # fill offsets and lengths for images
        offsets = prompt_ids.pop("offsets")
        lengths = prompt_ids.pop("lengths")
        i_num, o_num, l_num = len(self.images), len(offsets), len(lengths)
        assert i_num == o_num, "Invalid image offsets: {} vs {}".format(i_num, o_num)
        assert i_num == l_num, "Invalid image lengths: {} vs {}".format(i_num, l_num)
        for i, o, l in zip(self.images, offsets, lengths):
            i.offset = o
            i.length = l
        return prompt_ids["input_ids"]

    def verify(self):
        return

    def to_dict(self):
        ret = {}
        ret["images"] = [i.to_dict() for i in self.images]
        return ret
