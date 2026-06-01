import inspect
import time
import base64
import httpx
from PIL import Image
from io import BytesIO
from fastapi import Request
from functools import lru_cache
from collections import OrderedDict
from typing import Awaitable, Callable, Dict, Optional, Tuple
import asyncio
from lightllm.utils.error_utils import ClientDisconnected
from lightllm.utils.envs_utils import get_lightllm_url_pool_maxsize
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class UrlResourcePool:
    def __init__(self, maxsize: int = 256):
        self._maxsize = maxsize
        self._cache: "OrderedDict[Tuple[str, Optional[str]], bytes]" = OrderedDict()
        self._inflight: Dict[Tuple[str, Optional[str]], asyncio.Task] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _normalize_url(url: str) -> str:
        return url.strip()

    async def get_or_create(
        self, url: str, proxy: Optional[str], loader: Callable[[], Awaitable[bytes]]
    ) -> bytes:
        key = (self._normalize_url(url), proxy)

        async with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                logger.info(f"url_pool hit")
                return cached

            task = self._inflight.get(key)
            if task is None:

                async def _run_and_cache() -> bytes:
                    try:
                        content = await loader()
                        async with self._lock:
                            self._cache[key] = content
                            self._cache.move_to_end(key)
                            while len(self._cache) > self._maxsize:
                                self._cache.popitem(last=False)
                        return content
                    finally:
                        async with self._lock:
                            self._inflight.pop(key, None)

                task = asyncio.create_task(_run_and_cache())

                def _consume_task_exception(completed_task: asyncio.Task) -> None:
                    if completed_task.cancelled():
                        return
                    try:
                        completed_task.exception()
                    except BaseException:
                        return

                task.add_done_callback(_consume_task_exception)
                self._inflight[key] = task

        return await asyncio.shield(task)


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
    if request is not None and await request.is_disconnected():
        raise ClientDisconnected(reason=f"client disconnected during url download")

    start_time = time.time()

    async def _load() -> bytes:
        client = _get_xhttp_client(proxy)
        async with client.stream("GET", url, timeout=timeout) as response:
            response.raise_for_status()
            ans_bytes = []
            async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                ans_bytes.append(chunk)
                # 接收的数据不能大于128M
                if len(ans_bytes) > 128:
                    raise Exception("url data is too big")

        content = b"".join(ans_bytes)
        end_time = time.time()
        cost_time = end_time - start_time
        logger.info(f"url download done, cost={cost_time:.3f}s")
        return content

    return await URL_RESOURCE_POOL.get_or_create(url, proxy, _load)
