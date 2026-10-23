import asyncio
import base64
import inspect
import ipaddress
import os
import socket
import time
from urllib.parse import urlsplit

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


def _validate_resolved_host(hostname: str, port: Optional[int]) -> str:
    """Resolve a host, reject non-public results, and return a pinned address."""
    try:
        addresses = list(
            dict.fromkeys(result[4][0] for result in socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM))
        )
    except (OSError, UnicodeError) as exc:
        raise ValueError("resource URL host could not be resolved") from exc

    if not addresses:
        raise ValueError("resource URL host must resolve to a public address")
    try:
        has_non_public_address = any(not ipaddress.ip_address(address).is_global for address in addresses)
    except ValueError as exc:
        raise ValueError("resource URL host must resolve to a public address") from exc
    if has_non_public_address:
        raise ValueError("resource URL host must resolve to a public address")
    return next(iter(addresses))


def _normalize_resource_hostname(hostname: str) -> str:
    try:
        return str(ipaddress.ip_address(hostname))
    except ValueError:
        return hostname.rstrip(".").encode("idna").decode("ascii").lower()


def _resource_host_is_allowed(hostname: str) -> bool:
    configured_hosts = os.getenv("LIGHTLLM_MULTIMODAL_URL_ALLOWLIST")
    if configured_hosts is None:
        return True

    try:
        allowed_hosts = {
            _normalize_resource_hostname(allowed_host.strip())
            for allowed_host in configured_hosts.split(",")
            if allowed_host.strip()
        }
        return _normalize_resource_hostname(hostname) in allowed_hosts
    except UnicodeError as exc:
        raise ValueError("configured multimodal URL allowlist is invalid") from exc


async def validate_resource_url(url: str, timeout: Optional[float] = 5.0) -> Tuple[str, str, Optional[int], str]:
    """Validate a URL and return its scheme, host, port, and pinned public IP.

    LIGHTLLM_MULTIMODAL_URL_ALLOWLIST can contain comma-separated hostnames.
    """
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        port = parsed.port
    except (TypeError, ValueError) as exc:
        raise ValueError("resource URL is invalid") from exc

    if parsed.scheme not in {"http", "https"} or not hostname:
        raise ValueError("resource URL must use http or https")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("resource URL must not include credentials")
    if not _resource_host_is_allowed(hostname):
        raise ValueError("resource URL host is not in the configured allowlist")

    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        try:
            resolution = asyncio.to_thread(_validate_resolved_host, hostname, port)
            resolved_address = (
                await asyncio.wait_for(resolution, timeout=timeout) if timeout is not None else await resolution
            )
        except asyncio.TimeoutError as exc:
            raise ValueError("resource URL host could not be resolved") from exc
    else:
        if not address.is_global:
            raise ValueError("resource URL host must resolve to a public address")
        resolved_address = str(address)
    return parsed.scheme, hostname, port, resolved_address


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
    kvargs["follow_redirects"] = False
    return httpx.AsyncClient(**kvargs)


async def fetch_resource(url, request: Request, timeout, proxy=None):
    scheme, hostname, port, address = await validate_resource_url(url, timeout=timeout)
    logger.info(f"Begin to download resource from url: {url}")
    cached = URL_RESOURCE_POOL.get(url, proxy)
    if cached is not None:
        return cached

    start_time = time.time()
    client = _get_xhttp_client(proxy)
    parsed_url = urlsplit(url)
    pinned_host = f"[{address}]" if ":" in address else address
    if port is not None:
        pinned_host = f"{pinned_host}:{port}"
    pinned_url = parsed_url._replace(netloc=pinned_host).geturl()
    original_host = hostname.encode("idna").decode("ascii")
    host_header = f"[{original_host}]" if ":" in original_host else original_host
    default_port = 443 if scheme == "https" else 80
    if port is not None and port != default_port:
        host_header = f"{host_header}:{port}"
    http_request = client.build_request(
        "GET",
        pinned_url,
        headers={"Host": host_header},
        timeout=timeout,
        extensions={"sni_hostname": original_host},
    )
    response = await client.send(http_request, stream=True)
    try:
        response.raise_for_status()
        ans_bytes = []
        async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
            if request is not None and await request.is_disconnected():
                raise ClientDisconnected(reason=f"client disconnected during download of {url}")
            ans_bytes.append(chunk)
            # 接收的数据不能大于128M
            if len(ans_bytes) > 128:
                raise Exception(f"url {url} recv data is too big")
    finally:
        await response.aclose()

    content = b"".join(ans_bytes)
    end_time = time.time()
    cost_time = end_time - start_time
    logger.info(f"Download url {url} resource cost time: {cost_time} seconds")
    URL_RESOURCE_POOL.put(url, proxy, content)
    return content
