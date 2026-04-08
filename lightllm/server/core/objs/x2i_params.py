import ctypes
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from enum import IntEnum
from .token_chunck_hash_list import PastKVCachePageList


class CfgNormType(IntEnum):
    NONE = 0
    CFG_ZERO_STAR = 1
    GLOBAL = 2
    TEXT_CHANNEL = 3
    CHANNEL = 4

    def as_str(self) -> str:
        mapping = {
            CfgNormType.NONE: "none",
            CfgNormType.CFG_ZERO_STAR: "cfg_zero_star",
            CfgNormType.GLOBAL: "global",
            CfgNormType.TEXT_CHANNEL: "text_channel",
            CfgNormType.CHANNEL: "channel",
        }
        return mapping[self]

    def __repr__(self):
        return self.as_str()


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
        ("cfg_interval", ctypes.c_float * 2),
        ("timestep_shift", ctypes.c_float),
        ("cfg_norm", ctypes.c_int),
        ("past_kvcache", PastKVCachePageList),
        ("past_kvcache_text", PastKVCachePageList),
        ("past_kvcache_img", PastKVCachePageList),
        ("total_prompt_tokens", ctypes.c_int),
        ("request_id", ctypes.c_int64),
    ]

    _width: int = 1024
    _height: int = 1024
    _steps: int = 50
    _guidance_scale: float = 4.0
    _image_guidance_scale: float = 1.0
    _seed: int = 42
    _num_images: int = 1
    _cfg_norm: CfgNormType = CfgNormType.GLOBAL
    _cfg_interval: float = (-1, 2)
    _timestep_shift: float = 3.0

    def init(self, **kwargs):
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
        self.cfg_interval = _get("cfg_interval", X2IParams._cfg_interval)
        self.timestep_shift = _get("timestep_shift", X2IParams._timestep_shift)
        self.past_kvcache = PastKVCachePageList()
        self.past_kvcache_text = PastKVCachePageList()
        self.past_kvcache_img = PastKVCachePageList()
        self.total_prompt_tokens = 0
        self.request_id = 0

    def init_from_image_config(self, image_config: Any) -> None:
        """从 HTTP `image_config`（api_models.ImageConfig）填充，与 `init(**kwargs)` 共用默认值逻辑。"""
        from lightllm.server.api_models import ImageConfig

        if not isinstance(image_config, ImageConfig):
            raise TypeError(f"expected ImageConfig, got {type(image_config)!r}")
        w, h = image_config.get_resolution()
        kwargs: Dict[str, Any] = {"width": w, "height": h}
        if image_config.steps is not None:
            kwargs["steps"] = image_config.steps
        if image_config.guidance_scale is not None:
            kwargs["guidance_scale"] = image_config.guidance_scale
        if image_config.image_guidance_scale is not None:
            kwargs["image_guidance_scale"] = image_config.image_guidance_scale
        if image_config.seed is not None:
            kwargs["seed"] = image_config.seed
        if image_config.num_images is not None:
            kwargs["num_images"] = image_config.num_images
        if image_config.cfg_norm is not None:
            for e in CfgNormType:
                if e.as_str() == image_config.cfg_norm:
                    kwargs["cfg_norm"] = e
                    break
        self.init(**kwargs)

    def update(self, past_kv: PastKVCachePageList, meta: Dict):
        item: PastKVCacheItem = meta.get("kv_cache_item")
        past_kv.token_len = item.token_len
        past_kv.img_tokens = item.img_tokens
        past_kv.img_len = item.img_len
        past_kv.fill(item.page_indexes)
        self.total_prompt_tokens += past_kv.token_len

    def update_t2i(self, meta, meta_uncond):
        self.update(self.past_kvcache, meta)
        self.update(self.past_kvcache_text, meta_uncond)

    def update_it2i(self, meta, meta_text_uncond, meta_img_uncond):
        self.update(self.past_kvcache, meta)
        self.update(self.past_kvcache_text, meta_text_uncond)
        self.update(self.past_kvcache_img, meta_img_uncond)

    def get_cfg_norm(self):
        return CfgNormType(self.cfg_norm).as_str()

    def to_string(self):
        parts = []
        for field_name, _ in self._fields_:
            value = getattr(self, field_name)
            parts.append(f"{field_name}={value}")

        return "X2IParams(" + ", ".join(parts) + ")"

    def __repr__(self):
        return self.to_string()


@dataclass
class X2IResponse:
    request_id: int
    images: Optional[List[bytes]]


@dataclass
class X2ICacheRelease:
    request_id: int


@dataclass
class PastKVCacheItem:
    req_id: int
    token_len: int
    img_tokens: int
    img_len: int
    page_indexes: List[int]
