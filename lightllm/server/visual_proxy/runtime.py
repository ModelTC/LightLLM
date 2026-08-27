"""Configuration and bounded HTTP/SSE transport for the visual chat proxy."""

from __future__ import annotations

import asyncio
import base64
import json
import random
from contextlib import asynccontextmanager, suppress
from typing import Any, AsyncIterator, Optional
from urllib.parse import urljoin

import httpx
from fastapi import Request

from lightllm.utils.error_utils import ClientDisconnected
from lightllm.utils.log_utils import init_logger

from .config import (
    VisualProxyCapacityError,
    VisualProxySettings,
    VisualProxyTimeoutError,
    VisualProxyUpstreamError,
    VisualProxyUpstreamRejectedError,
)
from .images import (
    _normalized_image_content_type,
    _pinned_remote_image_request,
    _sniff_supported_image_media_type,
)


logger = init_logger(__name__)
REMOTE_IMAGE_MAX_REDIRECTS = 4


async def _request_is_disconnected(request: Optional[Request]) -> bool:
    if request is None:
        return False
    try:
        return await request.is_disconnected()
    except RuntimeError as exc:
        # In-process tests and adapters may not attach an ASGI receive
        # channel. Disconnect polling is unavailable in that case.
        if "Receive channel has not been made available" in str(exc):
            return False
        raise


class VisualProxyRuntime:
    """Per-HTTP-worker resources for bounded, pooled visual upstream calls."""

    def __init__(
        self, settings: VisualProxySettings, client: Optional[httpx.AsyncClient] = None
    ):
        self.settings = settings
        if settings.trace_dump_dir is not None:
            logger.warning(
                "Visual trajectory dump enabled: dir=%s; full prompts and image payloads will be stored",
                settings.trace_dump_dir,
            )
        headers = dict(settings.remote_headers)
        if settings.remote_api_key and not any(
            name.lower() == "authorization" for name in headers
        ):
            headers["Authorization"] = f"Bearer {settings.remote_api_key}"
        timeout = httpx.Timeout(
            connect=settings.remote_connect_timeout,
            read=settings.remote_timeout,
            write=min(settings.remote_timeout, 30.0),
            pool=settings.remote_queue_timeout,
        )
        self.client = client or httpx.AsyncClient(
            headers=headers,
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=settings.remote_max_concurrency,
                max_keepalive_connections=settings.remote_max_concurrency,
            ),
            trust_env=False,
        )
        # Keep image downloads isolated so provider credentials can never be
        # forwarded to an image host.
        self.image_client = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=settings.remote_max_concurrency,
                max_keepalive_connections=settings.remote_max_concurrency,
            ),
            follow_redirects=False,
            trust_env=False,
        )
        self._owns_client = client is None
        self._semaphore = asyncio.Semaphore(settings.remote_max_concurrency)
        self._request_semaphore = asyncio.Semaphore(settings.max_inflight_requests)
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    def ensure_remote_available(self) -> None:
        """Fail before entering DSV4 while the isolated Vision circuit is open."""

        if asyncio.get_running_loop().time() < self._circuit_open_until:
            raise VisualProxyUpstreamError("Visual upstream circuit breaker is open")

    def record_remote_success(self) -> None:
        """Close the circuit only after the complete provider contract is valid."""

        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    def record_remote_failure(self, trace_id: str) -> None:
        """Count one logical provider invocation failure, including invalid HTTP 200 bodies."""

        self._consecutive_failures += 1
        if self._consecutive_failures >= self.settings.circuit_failure_threshold:
            self._circuit_open_until = (
                asyncio.get_running_loop().time()
                + self.settings.circuit_recovery_seconds
            )
            logger.error(
                "[visual-chat-proxy][circuit_open] trace_id=%s failures=%d recovery_seconds=%.1f",
                trace_id,
                self._consecutive_failures,
                self.settings.circuit_recovery_seconds,
            )

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()
        await self.image_client.aclose()

    async def _acquire_with_disconnect(
        self,
        semaphore: asyncio.Semaphore,
        request: Optional[Request],
        saturated_message: str,
    ) -> None:
        """Acquire queued capacity without retaining a slot after client cancellation."""

        deadline = (
            asyncio.get_running_loop().time() + self.settings.remote_queue_timeout
        )
        acquired = False
        try:
            while True:
                if await _request_is_disconnected(request):
                    raise ClientDisconnected(
                        reason="client disconnected while waiting for visual capacity"
                    )
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise VisualProxyCapacityError(saturated_message)
                try:
                    await asyncio.wait_for(
                        semaphore.acquire(), timeout=min(0.25, remaining)
                    )
                    acquired = True
                    break
                except asyncio.TimeoutError:
                    continue
            if await _request_is_disconnected(request):
                raise ClientDisconnected(
                    reason="client disconnected while waiting for visual capacity"
                )
        except BaseException:
            if acquired:
                semaphore.release()
            raise

    @asynccontextmanager
    async def request_slot(self, request: Optional[Request]):
        await self._acquire_with_disconnect(
            self._request_semaphore,
            request,
            "Visual proxy request limit is saturated",
        )
        try:
            yield
        finally:
            self._request_semaphore.release()

    async def freeze_remote_image(
        self,
        source: str,
        request: Optional[Request],
        trace_id: str,
    ) -> str:
        """Download an allowed URL with DNS validation and bounded redirects."""

        deadline = asyncio.get_running_loop().time() + self.settings.remote_timeout
        current_url = source
        for redirect_index in range(REMOTE_IMAGE_MAX_REDIRECTS + 1):
            pinned_urls, host_header, sni_hostname = await asyncio.to_thread(
                _pinned_remote_image_request,
                current_url,
                self.settings,
            )
            response: Optional[httpx.Response] = None
            try:
                for candidate_index, pinned_url in enumerate(pinned_urls):
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        raise VisualProxyTimeoutError("Remote image download timed out")
                    extensions = (
                        {"sni_hostname": sni_hostname} if sni_hostname else None
                    )
                    upstream_request = self.image_client.build_request(
                        "GET",
                        pinned_url,
                        headers={
                            "Host": host_header,
                            "Accept": "image/png,image/jpeg,image/gif,image/webp",
                        },
                        extensions=extensions,
                    )
                    try:
                        response = await asyncio.wait_for(
                            self.image_client.send(upstream_request, stream=True),
                            timeout=remaining,
                        )
                        break
                    except (httpx.ConnectError, httpx.ConnectTimeout):
                        if candidate_index + 1 == len(pinned_urls):
                            raise
                        logger.warning(
                            "Remote image connection failed for one validated "
                            "address; trying the next"
                        )
                if response is None:
                    raise VisualProxyUpstreamError(
                        "Remote image host has no reachable validated address"
                    )
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("Location")
                    if not location:
                        raise ValueError("Remote image redirect has no Location header")
                    if redirect_index >= REMOTE_IMAGE_MAX_REDIRECTS:
                        raise ValueError("Remote image URL exceeded the redirect limit")
                    current_url = urljoin(current_url, location)
                    continue
                if response.status_code < 200 or response.status_code >= 300:
                    raise VisualProxyUpstreamRejectedError(
                        f"Remote image download failed with HTTP {response.status_code}"
                    )

                declared_length = response.headers.get("Content-Length")
                if declared_length is not None:
                    try:
                        if int(declared_length) > self.settings.max_image_bytes:
                            raise ValueError(
                                "Remote image exceeds the "
                                f"{self.settings.max_image_bytes}-byte limit"
                            )
                    except ValueError as exc:
                        if "exceeds" in str(exc):
                            raise

                body = bytearray()
                iterator = response.aiter_bytes().__aiter__()
                while True:
                    if await _request_is_disconnected(request):
                        raise ClientDisconnected(
                            reason="client disconnected during remote image download"
                        )
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        raise VisualProxyTimeoutError("Remote image download timed out")
                    try:
                        chunk = await asyncio.wait_for(
                            iterator.__anext__(), timeout=remaining
                        )
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError as exc:
                        raise VisualProxyTimeoutError(
                            "Remote image download timed out"
                        ) from exc
                    body.extend(chunk)
                    if len(body) > self.settings.max_image_bytes:
                        raise ValueError(
                            "Remote image exceeds the "
                            f"{self.settings.max_image_bytes}-byte limit"
                        )

                media_type = _sniff_supported_image_media_type(bytes(body))
                if media_type is None:
                    raise ValueError(
                        "Remote image body is not a supported PNG/JPEG/GIF/WebP image"
                    )
                declared_type = _normalized_image_content_type(
                    response.headers.get("Content-Type", "")
                )
                if declared_type != media_type:
                    raise ValueError(
                        "Remote image Content-Type does not match its magic bytes"
                    )
                logger.info(
                    "[visual-chat-proxy][remote_image_frozen] "
                    "trace_id=%s bytes=%d media_type=%s",
                    trace_id,
                    len(body),
                    media_type,
                )
                encoded = base64.b64encode(body).decode("ascii")
                return f"data:{media_type};base64,{encoded}"
            except asyncio.TimeoutError as exc:
                raise VisualProxyTimeoutError(
                    "Remote image download timed out"
                ) from exc
            except httpx.TimeoutException as exc:
                raise VisualProxyTimeoutError(
                    "Remote image download timed out"
                ) from exc
            except httpx.TransportError as exc:
                raise VisualProxyUpstreamError("Remote image download failed") from exc
            finally:
                if response is not None:
                    await response.aclose()

        raise ValueError("Remote image URL exceeded the redirect limit")

    async def _wait_for_disconnect(self, request: Request) -> None:
        while True:
            if await _request_is_disconnected(request):
                return
            await asyncio.sleep(0.25)

    async def _post_once(
        self,
        url: str,
        payload: dict[str, Any],
        request: Optional[Request],
        timeout: float,
        idempotency_key: str,
    ) -> httpx.Response:
        post_task = asyncio.create_task(
            self._bounded_post(url, payload, idempotency_key)
        )
        disconnect_task = None
        if request is not None:
            disconnect_task = asyncio.create_task(self._wait_for_disconnect(request))
        wait_set = {post_task}
        if disconnect_task is not None:
            wait_set.add(disconnect_task)
        try:
            done, _ = await asyncio.wait(
                wait_set, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )
            if not done:
                post_task.cancel()
                raise VisualProxyTimeoutError("Visual upstream request timed out")
            if disconnect_task is not None and disconnect_task in done:
                post_task.cancel()
                raise ClientDisconnected(
                    reason="client disconnected during visual upstream request"
                )
            return await post_task
        finally:
            if disconnect_task is not None:
                disconnect_task.cancel()
                with suppress(asyncio.CancelledError):
                    await disconnect_task
            if not post_task.done():
                post_task.cancel()
                with suppress(asyncio.CancelledError):
                    await post_task

    async def _bounded_post(
        self,
        url: str,
        payload: dict[str, Any],
        idempotency_key: str,
    ) -> httpx.Response:
        # Production uses a bounded streaming read. Minimal test doubles may
        # expose only ``post``.
        if not hasattr(self.client, "build_request") or not hasattr(
            self.client, "send"
        ):
            return await self.client.post(url, json=payload)
        request = self.client.build_request(
            "POST",
            url,
            json=payload,
            headers={"Idempotency-Key": f"lightllm-{idempotency_key}"},
        )
        response = await self.client.send(request, stream=True)
        body = bytearray()
        try:
            async for chunk in response.aiter_bytes():
                body.extend(chunk)
                if len(body) > self.settings.max_upstream_body_bytes:
                    raise VisualProxyUpstreamError(
                        "Visual upstream response body exceeds the configured size limit"
                    )
        finally:
            await response.aclose()
        return httpx.Response(
            status_code=response.status_code,
            headers=response.headers,
            content=bytes(body),
            request=request,
        )

    async def _open_stream_once(
        self,
        url: str,
        payload: dict[str, Any],
        request: Optional[Request],
        timeout: float,
        idempotency_key: str,
    ) -> httpx.Response:
        async def open_response() -> httpx.Response:
            if not hasattr(self.client, "build_request") or not hasattr(
                self.client, "send"
            ):
                return await self.client.post(url, json=payload)
            upstream_request = self.client.build_request(
                "POST",
                url,
                json=payload,
                headers={"Idempotency-Key": f"lightllm-{idempotency_key}"},
            )
            return await self.client.send(upstream_request, stream=True)

        open_task = asyncio.create_task(open_response())
        disconnect_task = None
        if request is not None:
            disconnect_task = asyncio.create_task(self._wait_for_disconnect(request))
        wait_set = {open_task}
        if disconnect_task is not None:
            wait_set.add(disconnect_task)
        try:
            done, _ = await asyncio.wait(
                wait_set, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )
            if not done:
                open_task.cancel()
                raise VisualProxyTimeoutError("Visual upstream request timed out")
            if disconnect_task is not None and disconnect_task in done:
                open_task.cancel()
                raise ClientDisconnected(
                    reason="client disconnected during visual upstream request"
                )
            return await open_task
        finally:
            if disconnect_task is not None:
                disconnect_task.cancel()
                with suppress(asyncio.CancelledError):
                    await disconnect_task
            if not open_task.done():
                open_task.cancel()
                with suppress(asyncio.CancelledError):
                    await open_task

    async def post_json(
        self,
        payload: dict[str, Any],
        request: Optional[Request],
        trace_id: str,
    ) -> dict[str, Any]:
        """POST one bounded non-streaming OpenAI request with retries."""

        self.ensure_remote_available()
        await self._acquire_with_disconnect(
            self._semaphore,
            request,
            "Visual upstream concurrency limit is saturated",
        )
        deadline = asyncio.get_running_loop().time() + self.settings.remote_timeout
        try:
            for attempt in range(self.settings.remote_max_retries + 1):
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise VisualProxyTimeoutError("Visual upstream request timed out")
                try:
                    response = await self._post_once(
                        self.settings.remote_url,
                        payload,
                        request,
                        remaining,
                        trace_id,
                    )
                except ClientDisconnected:
                    raise
                except VisualProxyTimeoutError:
                    if attempt >= self.settings.remote_max_retries:
                        raise
                    response = None
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    if attempt >= self.settings.remote_max_retries:
                        if isinstance(exc, httpx.TimeoutException):
                            raise VisualProxyTimeoutError(
                                "Visual upstream request timed out"
                            ) from exc
                        raise VisualProxyUpstreamError(
                            "Visual upstream connection failed"
                        ) from exc
                    response = None

                retryable_status = response is not None and (
                    response.status_code in {408, 429} or response.status_code >= 500
                )
                if response is not None and not retryable_status:
                    if response.status_code < 200 or response.status_code >= 300:
                        logger.warning(
                            "[visual-chat-proxy][visual_model_error] "
                            "trace_id=%s status=%d",
                            trace_id,
                            response.status_code,
                        )
                        raise VisualProxyUpstreamRejectedError(
                            "Visual upstream rejected the request with HTTP "
                            f"{response.status_code}"
                        )
                    try:
                        data = response.json()
                    except ValueError as exc:
                        raise VisualProxyUpstreamError(
                            "Visual upstream returned invalid JSON"
                        ) from exc
                    if not isinstance(data, dict):
                        raise VisualProxyUpstreamError(
                            "Visual upstream returned a non-object response"
                        )
                    return data

                if attempt >= self.settings.remote_max_retries:
                    status = (
                        response.status_code
                        if response is not None
                        else "transport_error"
                    )
                    raise VisualProxyUpstreamError(
                        "Visual upstream remained unavailable after retries "
                        f"(status={status})"
                    )
                retry_after = 0.0
                if response is not None:
                    with suppress(ValueError, TypeError):
                        retry_after = float(response.headers.get("Retry-After", "0"))
                delay = min(2.0, max(retry_after, 0.25 * (2**attempt)))
                delay *= 0.8 + random.random() * 0.4
                logger.warning(
                    "[visual-chat-proxy][visual_model_retry] "
                    "trace_id=%s attempt=%d delay=%.2f",
                    trace_id,
                    attempt + 1,
                    delay,
                )
                await asyncio.sleep(
                    min(
                        delay,
                        max(0.0, deadline - asyncio.get_running_loop().time()),
                    )
                )
        finally:
            self._semaphore.release()

    async def stream_json(
        self,
        payload: dict[str, Any],
        request: Optional[Request],
        trace_id: str,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream OpenAI-compatible SSE JSON while retaining bounded retries and cancellation."""

        self.ensure_remote_available()
        await self._acquire_with_disconnect(
            self._semaphore,
            request,
            "Visual upstream concurrency limit is saturated",
        )

        deadline = asyncio.get_running_loop().time() + self.settings.remote_timeout
        response: Optional[httpx.Response] = None
        try:
            for attempt in range(self.settings.remote_max_retries + 1):
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise VisualProxyTimeoutError("Visual upstream request timed out")
                try:
                    response = await self._open_stream_once(
                        self.settings.remote_url,
                        payload,
                        request,
                        remaining,
                        trace_id,
                    )
                except ClientDisconnected:
                    raise
                except VisualProxyTimeoutError:
                    if attempt >= self.settings.remote_max_retries:
                        raise
                    response = None
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    if attempt >= self.settings.remote_max_retries:
                        if isinstance(exc, httpx.TimeoutException):
                            raise VisualProxyTimeoutError(
                                "Visual upstream request timed out"
                            ) from exc
                        raise VisualProxyUpstreamError(
                            "Visual upstream connection failed"
                        ) from exc
                    response = None

                retryable_status = response is not None and (
                    response.status_code in {408, 429} or response.status_code >= 500
                )
                if response is not None and not retryable_status:
                    if response.status_code < 200 or response.status_code >= 300:
                        raise VisualProxyUpstreamRejectedError(
                            f"Visual upstream rejected the request with HTTP {response.status_code}"
                        )
                    break
                if response is not None:
                    await response.aclose()
                    response = None
                if attempt >= self.settings.remote_max_retries:
                    raise VisualProxyUpstreamError(
                        "Visual upstream remained unavailable after retries"
                    )
                delay = 0.25 * (2**attempt) * (0.8 + random.random() * 0.4)
                logger.warning(
                    "[visual-chat-proxy][visual_model_retry] trace_id=%s attempt=%d delay=%.2f",
                    trace_id,
                    attempt + 1,
                    delay,
                )
                await asyncio.sleep(
                    min(delay, max(0.0, deadline - asyncio.get_running_loop().time()))
                )

            if response is None:
                raise VisualProxyUpstreamError(
                    "Visual upstream did not return a streaming response"
                )

            async def bounded_body() -> AsyncIterator[bytes]:
                total_bytes = 0
                iterator = response.aiter_bytes().__aiter__()
                while True:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        raise VisualProxyTimeoutError(
                            "Visual upstream request timed out"
                        )
                    if await _request_is_disconnected(request):
                        raise ClientDisconnected(
                            reason="client disconnected during visual upstream request"
                        )
                    try:
                        chunk = await asyncio.wait_for(
                            iterator.__anext__(), timeout=remaining
                        )
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError as exc:
                        raise VisualProxyTimeoutError(
                            "Visual upstream request timed out"
                        ) from exc
                    except httpx.TimeoutException as exc:
                        raise VisualProxyTimeoutError(
                            "Visual upstream request timed out"
                        ) from exc
                    except httpx.TransportError as exc:
                        raise VisualProxyUpstreamError(
                            "Visual upstream connection failed"
                        ) from exc
                    total_bytes += len(chunk)
                    if total_bytes > self.settings.max_upstream_body_bytes:
                        raise VisualProxyUpstreamError(
                            "Visual upstream response body exceeds the configured size limit"
                        )
                    yield chunk

            async for item in _iter_openai_sse_payloads(bounded_body()):
                yield item
        finally:
            if response is not None:
                await response.aclose()
            self._semaphore.release()


async def _iter_openai_sse_payloads(
    body_iterator: Any,
) -> AsyncIterator[dict[str, Any]]:
    """Decode complete JSON payloads from an OpenAI SSE body iterator."""

    buffer = ""
    async for raw_chunk in body_iterator:
        if not raw_chunk:
            continue
        if isinstance(raw_chunk, (bytes, bytearray, memoryview)):
            raw_chunk = bytes(raw_chunk).decode("utf-8", errors="replace")
        buffer += str(raw_chunk).replace("\r\n", "\n")
        while "\n\n" in buffer:
            event, buffer = buffer.split("\n\n", 1)
            data_lines = [
                line[5:].lstrip()
                for line in event.split("\n")
                if line.startswith("data:")
            ]
            if not data_lines:
                continue
            data = "\n".join(data_lines)
            if data == "[DONE]":
                continue
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                logger.debug("Skipping malformed main-model SSE payload: %r", data)
                continue
            if isinstance(payload, dict):
                yield payload
    if buffer.strip():
        for line in buffer.splitlines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload
