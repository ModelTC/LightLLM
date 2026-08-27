"""Validation and normalization for images sent to the Vision provider."""

from __future__ import annotations

import base64
import ipaddress
import mimetypes
import os
import socket
import stat
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote, urlsplit, urlunsplit


ALLOWED_IMAGE_MEDIA_TYPES = frozenset(
    {"image/gif", "image/jpeg", "image/png", "image/webp"}
)


def _is_loopback_host(host: str) -> bool:
    normalized = host.lower().rstrip(".")
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _normalized_image_content_type(value: str) -> str:
    media_type = value.split(";", 1)[0].strip().lower()
    return "image/jpeg" if media_type == "image/jpg" else media_type


def _sniff_supported_image_media_type(content: bytes) -> Optional[str]:
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if content.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if content.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return "image/webp"
    return None


def _pinned_remote_image_request(
    source: str,
    settings: Any,
) -> tuple[tuple[str, ...], str, Optional[str]]:
    """Validate every DNS answer and return IP-pinned request targets."""

    parsed = urlsplit(source)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Image URL must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password:
        raise ValueError("Image URL must not contain embedded credentials")
    try:
        host = parsed.hostname.encode("idna").decode("ascii").lower().rstrip(".")
    except UnicodeError as exc:
        raise ValueError("Image URL contains an invalid hostname") from exc
    if host not in settings.remote_image_hosts:
        raise ValueError(
            "Image URL host is not in the configured visual remote-image allowlist"
        )
    if parsed.scheme == "http" and not settings.allow_http_image_urls:
        raise ValueError("Plain HTTP image URLs are disabled; use HTTPS or a data URL")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        addresses = {
            item[4][0]
            for item in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        }
    except socket.gaierror as exc:
        raise ValueError("Remote image host could not be resolved") from exc
    if not addresses:
        raise ValueError("Remote image host resolved to no addresses")

    parsed_addresses = []
    for raw_address in addresses:
        address = ipaddress.ip_address(raw_address)
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
            address = address.ipv4_mapped
        parsed_addresses.append(address)
    if not settings.allow_private_remote_image_urls and any(
        not address.is_global or address.is_multicast for address in parsed_addresses
    ):
        raise ValueError(
            "Remote image host resolved to a private or non-global address"
        )

    parsed_addresses = sorted(
        set(parsed_addresses), key=lambda item: (item.version, item.packed)
    )
    pinned_urls = []
    for pinned_address in parsed_addresses:
        pinned_host = (
            f"[{pinned_address.compressed}]"
            if pinned_address.version == 6
            else pinned_address.compressed
        )
        pinned_urls.append(
            urlunsplit(
                (
                    parsed.scheme,
                    f"{pinned_host}:{port}",
                    parsed.path or "/",
                    parsed.query,
                    "",
                )
            )
        )
    explicit_port = parsed.port is not None
    default_port = (parsed.scheme == "https" and port == 443) or (
        parsed.scheme == "http" and port == 80
    )
    host_header = host if default_port and not explicit_port else f"{host}:{port}"
    return tuple(pinned_urls), host_header, host if parsed.scheme == "https" else None


def normalize_visual_remote_url(
    raw_url: str,
    allow_insecure_http: bool = False,
) -> str:
    url = raw_url.strip().rstrip("/")
    if not url:
        raise ValueError("visual_remote_url must not be empty")
    if url.endswith("/chat/completions"):
        normalized = url
    elif url.endswith("/v1"):
        normalized = f"{url}/chat/completions"
    else:
        normalized = f"{url}/v1/chat/completions"
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("visual_remote_url must be an absolute http(s) URL")
    if parsed.username or parsed.password:
        raise ValueError("visual_remote_url must not contain embedded credentials")
    if parsed.query or parsed.fragment:
        raise ValueError(
            "visual_remote_url must not contain query parameters or fragments"
        )
    if (
        parsed.scheme == "http"
        and not allow_insecure_http
        and not _is_loopback_host(parsed.hostname)
    ):
        raise ValueError(
            "visual_remote_url must use HTTPS unless it targets localhost/loopback; "
            "use --visual_allow_insecure_remote_url only for a trusted legacy network"
        )
    return normalized


def _validate_data_image(source: str, max_image_bytes: int) -> str:
    try:
        header, payload = source.split(",", 1)
    except ValueError as exc:
        raise ValueError("Malformed image data URL") from exc
    media_type = header[5:].split(";", 1)[0].lower()
    if media_type not in ALLOWED_IMAGE_MEDIA_TYPES or ";base64" not in header.lower():
        raise ValueError(
            "Image data URLs must use a supported raster image media type with "
            "base64 encoding"
        )
    estimated_size = (len(payload) * 3) // 4
    if estimated_size > max_image_bytes + 2:
        raise ValueError(f"Image exceeds the {max_image_bytes}-byte limit")
    try:
        decoded = base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise ValueError("Image data URL contains invalid base64") from exc
    if len(decoded) > max_image_bytes:
        raise ValueError(f"Image exceeds the {max_image_bytes}-byte limit")
    sniffed_media_type = _sniff_supported_image_media_type(decoded)
    if sniffed_media_type is None:
        raise ValueError("Image data URL is not a supported PNG/JPEG/GIF/WebP image")
    if sniffed_media_type != media_type:
        raise ValueError("Image data URL media type does not match its magic bytes")
    return f"data:{media_type};base64,{base64.b64encode(decoded).decode('ascii')}"


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def image_url_for_visual_provider(source: str, settings: Any) -> str:
    """Return a validated provider image URL, inlining allowed local files."""

    if source.startswith("data:"):
        return _validate_data_image(source, settings.max_image_bytes)
    if source.startswith(("http://", "https://")):
        if not settings.allow_remote_image_urls:
            raise ValueError(
                "Remote image URLs are disabled; send a data URL or explicitly "
                "enable --visual_allow_remote_image_urls"
            )
        parsed = urlsplit(source)
        if not parsed.hostname or parsed.username or parsed.password:
            raise ValueError(
                "Image URL must be an absolute URL without embedded credentials"
            )
        if parsed.hostname.lower().rstrip(".") not in settings.remote_image_hosts:
            raise ValueError(
                "Image URL host is not in the configured visual remote-image allowlist"
            )
        if parsed.scheme == "http" and not settings.allow_http_image_urls:
            raise ValueError(
                "Plain HTTP image URLs are disabled; use HTTPS or a data URL"
            )
        if len(source) > 8192:
            raise ValueError("Image URL exceeds the 8192-character limit")
        return source
    if source.startswith("file://"):
        if not settings.allow_local_files:
            raise ValueError("Local file images are disabled")
        parsed = urlsplit(source)
        if parsed.netloc not in {"", "localhost"}:
            raise ValueError("file:// image URLs must not contain a remote host")
        path = Path(unquote(parsed.path)).expanduser().resolve(strict=True)
        if not any(_path_is_within(path, root) for root in settings.local_file_roots):
            raise ValueError("Image file is outside the configured local file roots")
        if not path.is_file():
            raise ValueError(f"Image file does not exist: {path}")
        mime_type = _normalized_image_content_type(
            mimetypes.guess_type(path.name)[0] or ""
        )
        if mime_type not in ALLOWED_IMAGE_MEDIA_TYPES:
            raise ValueError(
                "Local visual inputs must use a supported raster image media type"
            )
        open_flags = (
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            file_descriptor = os.open(path, open_flags)
        except OSError as exc:
            raise ValueError("Image file could not be opened safely") from exc
        with os.fdopen(file_descriptor, "rb") as image_file:
            file_stat = os.fstat(image_file.fileno())
            if not stat.S_ISREG(file_stat.st_mode):
                raise ValueError("Local visual input must be a regular file")
            if file_stat.st_size > settings.max_image_bytes:
                raise ValueError(
                    f"Image exceeds the {settings.max_image_bytes}-byte limit"
                )
            image_bytes = image_file.read(settings.max_image_bytes + 1)
        if len(image_bytes) > settings.max_image_bytes:
            raise ValueError(f"Image exceeds the {settings.max_image_bytes}-byte limit")
        sniffed_media_type = _sniff_supported_image_media_type(image_bytes)
        if sniffed_media_type is None or sniffed_media_type != mime_type:
            raise ValueError("Local image type does not match its magic bytes")
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"
    raise ValueError(
        "Unrecognized image input. Use an image data URL or an explicitly enabled "
        "image source."
    )


# Compatibility name used by the PR implementation and its tests.
_remote_image_url = image_url_for_visual_provider
