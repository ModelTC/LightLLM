import ctypes
from dataclasses import dataclass
from typing import Dict, List
from enum import IntEnum
from .token_chunck_hash_list import PastKVCachePageList

class CfgNormType(IntEnum):
    NONE = 0
    CFG_ZERO_STAR = 1
    GLOBAL = 2


class X2IParams(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("steps", ctypes.c_int),
        ("guidance_scale", ctypes.c_float),
        ("image_guidance_scale", ctypes.c_float),
        ("seed", ctypes.c_int),
        ("num_images", ctypes.c_int),
        ("cfg_norm", ctypes.c_int),
        ("past_kvcache", PastKVCachePageList),
        ("past_kvcache_text", PastKVCachePageList),
        ("past_kvcache_img", PastKVCachePageList),
        ("total_prompt_tokens", ctypes.c_int),
        ("request_id", ctypes.c_int64),
    ]

    _width: int = 512
    _height: int = 512
    _steps: int = 30
    _guidance_scale: float = 7.0
    _image_guidance_scale: float = 7.0
    _seed: int = 42
    _num_images: int = 1
    _cfg_norm: CfgNormType = CfgNormType.NONE

    def init(self,  **kwargs):
        def _get(key, default):
            v = kwargs.get(key)
            return v if v is not None else default
        self.width = _get("width", X2IParams._width)
        self.height = _get("height", X2IParams._height)
        self.steps = _get("steps", X2IParams._steps)
        self.guidance_scale = _get("guidance_scale", X2IParams._guidance_scale)
        self.image_guidance_scale = _get("image_guidance_scale", X2IParams._image_guidance_scale)
        self.seed = _get("seed", X2IParams._seed)
        self.num_images = _get("num_images", X2IParams._num_images)
        self.cfg_norm = _get("cfg_norm", X2IParams._cfg_norm)
        self.past_kvcache = PastKVCachePageList()
        self.past_kvcache_text = PastKVCachePageList()
        self.past_kvcache_img = PastKVCachePageList()
        self.total_prompt_tokens = 0
        self.request_id = 0

    def update(self, past_kv: PastKVCachePageList, meta: Dict):
        past_kv.token_len = meta.get("prompt_tokens")
        past_kv.fill(meta.get("kv_cache_pages"))
        self.total_prompt_tokens += past_kv.token_len

    def update_t2i(self, meta, meta_uncond):
        self.update(self.past_kvcache, meta)
        self.update(self.past_kvcache_text, meta_uncond)

    def update_it2i(self, meta, meta_text_uncond, meta_img_uncond):
        self.update(self.past_kvcache, meta)
        self.update(self.past_kvcache_text, meta_text_uncond)
        self.update(self.past_kvcache_img, meta_img_uncond)


@dataclass
class X2IResponse:
    request_id: int
    images: List[bytes]

@dataclass
class X2ICacheRelease:
    request_id: int