import inspect
import time
import base64
import httpx
from PIL import Image
from io import BytesIO
from fastapi import Request
from functools import lru_cache
from collections import OrderedDict
from typing import Optional, Tuple
from lightllm.utils.error_utils import ClientDisconnected
from lightllm.utils.envs_utils import get_env_start_args, get_lightllm_url_pool_maxsize
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class UrlResourcePool:
    """Cache image, video, and audio URL content to avoid repeated downloads."""

    def __init__(self, maxsize: int = 256):
        self._maxsize = maxsize
        self._cache: "OrderedDict[Tuple[str, Optional[str]], bytes]" = OrderedDict()

    def get(self, url: str, proxy: Optional[str]) -> Optional[bytes]:
        if not get_env_start_args().enable_multimodal_url_cache:
            return None

        key = (url.strip(), proxy)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            logger.info("url_pool hit")
        return cached

    def put(self, url: str, proxy: Optional[str], content: bytes) -> None:
        if not get_env_start_args().enable_multimodal_url_cache or self._maxsize <= 0:
            return

        key = (url.strip(), proxy)
        self._cache[key] = content
        self._cache.move_to_end(key)
        while len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)


URL_RESOURCE_POOL = UrlResourcePool(maxsize=get_lightllm_url_pool_maxsize())


def _httpx_async_client_proxy_kwargs(proxy) -> dict:
    """
    httpx 0.28+ 使用 AsyncClient(proxy=...)；更早版本使用 proxies=...
    用签名检测避免写死版本号。
    """
    if proxy is None:
        return {}
    params = inspect.signature(httpx.AsyncClient.__init__).parameters

    if "proxy" in params:
        return {"proxy": proxy}
    if "proxies" in params:
        return {"proxies": proxy}
    return {}


def image2base64(img_str: str):
    image_obj = Image.open(img_str)
    if image_obj.format is None:
        raise ValueError("No image format found.")
    buffer = BytesIO()
    image_obj.save(buffer, format=image_obj.format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@lru_cache(maxsize=256)
def _get_xhttp_client(proxy=None):
    kvargs = _httpx_async_client_proxy_kwargs(proxy)
    kvargs["limits"] = httpx.Limits(max_connections=10000, max_keepalive_connections=20)
    return httpx.AsyncClient(**kvargs)


async def fetch_resource(url, request: Request, timeout, proxy=None):
    logger.info(f"Begin to download resource from url: {url}")
    cached = URL_RESOURCE_POOL.get(url, proxy)
    if cached is not None:
        return cached

    start_time = time.time()
    client = _get_xhttp_client(proxy)
    async with client.stream("GET", url, timeout=timeout) as response:
        response.raise_for_status()
        ans_bytes = []
        async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
            if request is not None and await request.is_disconnected():
                await response.aclose()
                raise ClientDisconnected(reason=f"client disconnected during download of {url}")
            ans_bytes.append(chunk)
            # 接收的数据不能大于128M
            if len(ans_bytes) > 128:
                raise Exception(f"url {url} recv data is too big")

    content = b"".join(ans_bytes)
    end_time = time.time()
    cost_time = end_time - start_time
    logger.info(f"Download url {url} resource cost time: {cost_time} seconds")
    URL_RESOURCE_POOL.put(url, proxy, content)
    return content
