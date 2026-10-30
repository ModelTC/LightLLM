import asyncio
import importlib
import socket
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from lightllm.utils import multimodal_utils


def _load_multimodal_params(monkeypatch):
    server_path = Path(multimodal_utils.__file__).resolve().parents[1] / "server"
    server_package = types.ModuleType("lightllm.server")
    server_package.__path__ = [str(server_path)]
    embed_cache_package = types.ModuleType("lightllm.server.embed_cache")
    embed_cache_package.__path__ = [str(server_path / "embed_cache")]
    monkeypatch.setitem(sys.modules, "lightllm.server", server_package)
    monkeypatch.setitem(sys.modules, "lightllm.server.embed_cache", embed_cache_package)
    return importlib.import_module("lightllm.server.multimodal_params")


def _run_validation(url: str) -> None:
    asyncio.run(multimodal_utils.validate_resource_url(url))


def _resolved_address(address: str):
    return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (address, 80))]


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/image.png",
        "http://10.0.0.1/image.png",
        "http://169.254.169.254/latest/meta-data",
        "http://[::1]/image.png",
        "http://[fe80::1]/image.png",
        "http://[fc00::1]/image.png",
        "file:///etc/passwd",
        "ftp://8.8.8.8/image.png",
    ],
)
def test_validate_resource_url_rejects_unsafe_urls(url):
    with pytest.raises(ValueError):
        _run_validation(url)


def test_validate_resource_url_rejects_private_dns_result(monkeypatch):
    monkeypatch.delenv("LIGHTLLM_MULTIMODAL_URL_ALLOWLIST", raising=False)
    monkeypatch.setattr(multimodal_utils.socket, "getaddrinfo", lambda *args, **kwargs: _resolved_address("10.0.0.8"))

    with pytest.raises(ValueError, match="public address") as error:
        _run_validation("https://internal.example/image.png")

    assert "internal.example" not in str(error.value)


def test_validate_resource_url_allows_public_dns_result(monkeypatch):
    monkeypatch.delenv("LIGHTLLM_MULTIMODAL_URL_ALLOWLIST", raising=False)
    monkeypatch.setattr(multimodal_utils.socket, "getaddrinfo", lambda *args, **kwargs: _resolved_address("8.8.8.8"))

    result = asyncio.run(multimodal_utils.validate_resource_url("https://public.example/image.png"))

    assert result == ("https", "public.example", None, "8.8.8.8")


def test_validate_resource_url_enforces_configured_host_allowlist(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_MULTIMODAL_URL_ALLOWLIST", "images.example,media.example")
    monkeypatch.setattr(
        multimodal_utils.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: pytest.fail("disallowed host was resolved"),
    )

    with pytest.raises(ValueError, match="configured allowlist"):
        _run_validation("https://other.example/image.png")


def test_validate_resource_url_allows_configured_hostname(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_MULTIMODAL_URL_ALLOWLIST", "PUBLIC.EXAMPLE.,media.example")
    monkeypatch.setattr(multimodal_utils.socket, "getaddrinfo", lambda *args, **kwargs: _resolved_address("8.8.8.8"))

    result = asyncio.run(multimodal_utils.validate_resource_url("https://public.example/image.png"))

    assert result == ("https", "public.example", None, "8.8.8.8")


def test_configured_allowlist_does_not_allow_private_dns_result(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_MULTIMODAL_URL_ALLOWLIST", "internal.example")
    monkeypatch.setattr(multimodal_utils.socket, "getaddrinfo", lambda *args, **kwargs: _resolved_address("10.0.0.8"))

    with pytest.raises(ValueError, match="public address"):
        _run_validation("https://internal.example/image.png")


def test_validate_resource_url_rejects_mixed_dns_results(monkeypatch):
    monkeypatch.delenv("LIGHTLLM_MULTIMODAL_URL_ALLOWLIST", raising=False)
    monkeypatch.setattr(
        multimodal_utils.socket,
        "getaddrinfo",
        lambda *args, **kwargs: _resolved_address("8.8.8.8") + _resolved_address("192.168.1.1"),
    )

    with pytest.raises(ValueError, match="public address"):
        _run_validation("https://mixed.example/image.png")


def test_fetch_resource_validates_before_opening_http_client(monkeypatch):
    monkeypatch.setattr(
        multimodal_utils,
        "_get_xhttp_client",
        lambda *_args, **_kwargs: pytest.fail("unsafe URL reached the HTTP client"),
    )

    with pytest.raises(ValueError):
        asyncio.run(multimodal_utils.fetch_resource("http://127.0.0.1/image.png", None, timeout=1))


def test_http_client_does_not_follow_redirects(monkeypatch):
    monkeypatch.setattr(multimodal_utils.httpx, "AsyncClient", lambda **kwargs: kwargs)
    multimodal_utils._get_xhttp_client.cache_clear()
    try:
        client_options = multimodal_utils._get_xhttp_client()
    finally:
        multimodal_utils._get_xhttp_client.cache_clear()

    assert client_options["follow_redirects"] is False


def test_fetch_resource_pins_dns_result_and_preserves_host(monkeypatch):
    monkeypatch.delenv("LIGHTLLM_MULTIMODAL_URL_ALLOWLIST", raising=False)
    monkeypatch.setattr(multimodal_utils.socket, "getaddrinfo", lambda *args, **kwargs: _resolved_address("8.8.8.8"))
    monkeypatch.setattr(multimodal_utils.URL_RESOURCE_POOL, "get", lambda *_args: None)
    monkeypatch.setattr(multimodal_utils.URL_RESOURCE_POOL, "put", lambda *_args: None)

    class FakeResponse:
        def raise_for_status(self):
            pass

        async def aiter_bytes(self, chunk_size):
            yield b"image bytes"

        async def aclose(self):
            pass

    class FakeClient:
        def build_request(self, method, url, **kwargs):
            self.request = type("Request", (), {"method": method, "url": url, **kwargs})()
            return self.request

        async def send(self, request, stream):
            self.sent_request = request
            return FakeResponse()

    client = FakeClient()
    monkeypatch.setattr(multimodal_utils, "_get_xhttp_client", lambda *_args: client)

    content = asyncio.run(
        multimodal_utils.fetch_resource("https://public.example/image.png?size=small", None, timeout=1)
    )

    assert content == b"image bytes"
    assert client.sent_request.url == "https://8.8.8.8/image.png?size=small"
    assert client.sent_request.headers["Host"] == "public.example"
    assert client.sent_request.extensions["sni_hostname"] == "public.example"


@pytest.mark.parametrize(
    ("item_type", "expected_message"),
    [
        ("image", "Failed to read image resource"),
        ("audio", "Failed to read audio resource"),
    ],
)
def test_remote_resource_errors_do_not_expose_fetch_details(monkeypatch, item_type, expected_message):
    multimodal_params = _load_multimodal_params(monkeypatch)

    async def fail_fetch(*args, **kwargs):
        raise RuntimeError("Client error '404 Not Found' for http://127.0.0.1/private.png")

    monkeypatch.setattr(
        multimodal_params,
        "fetch_resource",
        fail_fetch,
    )
    monkeypatch.setattr(
        multimodal_params,
        "get_env_start_args",
        lambda: SimpleNamespace(max_image_pixels=1_000_000, disable_image_resize=False),
    )

    item_class = multimodal_params.ImageItem if item_type == "image" else multimodal_params.AudioItem
    item = item_class(type="url", data="http://127.0.0.1/private.png")
    with pytest.raises(ValueError) as error:
        asyncio.run(item.preload(None))

    assert str(error.value) == expected_message
    assert "127.0.0.1" not in str(error.value)
    assert "404" not in str(error.value)
