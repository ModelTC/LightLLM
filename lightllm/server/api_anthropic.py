"""Anthropic Messages API compatibility layer.

Translates incoming /v1/messages requests into LightLLM's internal chat
completions pipeline by delegating the hard parts (content-block parsing,
tool schema normalisation, stop-reason mapping) to LiteLLM's adapter.

The streaming path intercepts the OpenAI-format SSE stream from
chat_completions_impl and re-emits it as the Anthropic event sequence
(message_start, content_block_*, message_delta, message_stop).
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import os
import select
import shutil
import subprocess
import tempfile
import time
import uuid
import ujson as json
from collections import OrderedDict, deque
from contextlib import aclosing
from dataclasses import dataclass, field
from http import HTTPStatus
from threading import Lock
from typing import Any, Dict, Tuple

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.error_utils import GenerationError
from lightllm.utils.log_utils import init_logger

from .api_errors import is_rate_limit_error

logger = init_logger(__name__)

_cached_adapter: Any = None
_PDF_MAX_BYTES = 20 * 1024 * 1024
_PDF_MAX_DOCUMENTS = 20
_PDF_MAX_RENDER_PAGES = 20
_PDF_MAX_TEXT_BYTES = 10 * 1024 * 1024
_PDF_MAX_RENDER_DIMENSION = 2048
_PDF_MAX_RENDER_PAGE_BYTES = 10 * 1024 * 1024
_PDF_MAX_RENDERED_BYTES = 20 * 1024 * 1024
_PDF_MAX_CONVERTED_BYTES = 28 * 1024 * 1024
_PDF_SUBPROCESS_TIMEOUT_SECONDS = 30
_PDF_PARSING_ENV = "LIGHTLLM_ANTHROPIC_ENABLE_PDF_PARSING"
_PDF_REQUIRED_TOOLS = ("pdftotext", "pdftoppm", "pdfinfo")
_PDF_CACHE_MAX_BYTES = 2 * 1024 * 1024 * 1024
_PDF_CACHE: OrderedDict[tuple[str, bytes, int], tuple[int, Any]] = OrderedDict()
_PDF_CACHE_BYTES = 0
_PDF_CACHE_LOCK = Lock()
_PDF_CACHE_MISS = object()


def get_anthropic_messages_adapter() -> Any:
    """Return a cached instance of LiteLLM's Anthropic<->OpenAI adapter.

    The returned object exposes ``translate_anthropic_to_openai`` and
    ``translate_openai_response_to_anthropic`` methods.
    """
    global _cached_adapter
    if _cached_adapter is not None:
        return _cached_adapter

    try:
        from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
            LiteLLMAnthropicMessagesAdapter,
        )
    except ImportError as exc:
        raise RuntimeError(
            "The Anthropic Messages API (/v1/messages) requires the 'litellm' package. "
            "Install it with: pip install 'lightllm[anthropic]' "
            "(or directly: pip install 'litellm>=1.52.0,<1.85'). "
            f"Original error: {exc}"
        ) from exc

    _cached_adapter = LiteLLMAnthropicMessagesAdapter()
    return _cached_adapter


# ---------------------------------------------------------------------------
# Request translation
# ---------------------------------------------------------------------------


def _preserve_replayed_thinking(openai_dict: Dict[str, Any]) -> None:
    """Move LiteLLM's thinking blocks to a field understood by LightLLM."""
    for message in openai_dict.get("messages") or []:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        thinking_blocks = message.pop("thinking_blocks", None)
        if not isinstance(thinking_blocks, list):
            continue
        reasoning_text = "".join(
            block.get("thinking", "")
            for block in thinking_blocks
            if isinstance(block, dict) and block.get("type") == "thinking" and isinstance(block.get("thinking"), str)
        )
        if reasoning_text and not message.get("reasoning_content"):
            message["reasoning_content"] = reasoning_text


def _anthropic_to_chat_request(anthropic_body: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Translate an Anthropic Messages request body into a dict suitable
    for constructing a LightLLM ``ChatCompletionRequest``.

    Returns ``(chat_request_dict, tool_name_mapping)``. The mapping must
    be passed back to ``_chat_response_to_anthropic`` so that tool names
    truncated by LiteLLM's 64-character limit can be restored.
    """
    adapter = get_anthropic_messages_adapter()

    # LiteLLM's Anthropic request schema does not accept the native
    # ``thinking`` parameter.  Translate it to the LightLLM controls before
    # handing the request to LiteLLM, then remove the Anthropic-only field.
    # This also makes the native Anthropic spelling behave the same as the
    # existing ``extra_body.chat_template_kwargs`` escape hatch.
    anthropic_body = dict(anthropic_body)
    system = anthropic_body.get("system")
    billing_prefix = "x-anthropic-billing-header:"
    if isinstance(system, str) and system.startswith(billing_prefix):
        anthropic_body["system"] = system.partition("\n")[2]
    elif isinstance(system, list):
        system_blocks = []
        for block in system:
            if (
                isinstance(block, dict)
                and block.get("type") == "text"
                and (block.get("text") or "").startswith(billing_prefix)
            ):
                remaining_text = block["text"].partition("\n")[2]
                if remaining_text:
                    system_blocks.append({**block, "text": remaining_text})
            else:
                system_blocks.append(block)
        anthropic_body["system"] = system_blocks
    thinking = anthropic_body.pop("thinking", None)
    if isinstance(thinking, dict) and thinking.get("type") in {"enabled", "disabled"}:
        extra_body = dict(anthropic_body.get("extra_body") or {})
        chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
        chat_template_kwargs["enable_thinking"] = thinking["type"] == "enabled"
        extra_body["chat_template_kwargs"] = chat_template_kwargs
        # Anthropic has a separate thinking content block, so make sure the
        # downstream response keeps reasoning separate from normal text.
        extra_body["separate_reasoning"] = True
        anthropic_body["extra_body"] = extra_body

    _replace_anthropic_pdf_documents(anthropic_body)
    tool_result_image_urls = _tool_result_image_urls(anthropic_body)
    openai_request, tool_name_mapping = adapter.translate_anthropic_to_openai(anthropic_body)

    if hasattr(openai_request, "model_dump"):
        openai_dict = openai_request.model_dump(exclude_none=True)
    else:
        openai_dict = dict(openai_request)

    _preserve_replayed_thinking(openai_dict)

    if "max_tokens" not in openai_dict and "max_completion_tokens" not in openai_dict:
        if "max_tokens" in anthropic_body:
            openai_dict["max_tokens"] = anthropic_body["max_tokens"]

    _restore_tool_result_image_urls(openai_dict, tool_result_image_urls)

    # Forward LightLLM-specific fields nested under ``extra_body`` (OpenAI SDK
    # convention) so clients hitting /v1/messages can reach ChatCompletionRequest
    # options Anthropic's own schema does not expose — notably chat_template_kwargs
    # for models with optional thinking modes (Qwen3, DeepSeek). Fields already
    # produced by the Anthropic->OpenAI translation take precedence; unknown keys
    # are silently dropped by Pydantic (extra='ignore').
    extra_body = anthropic_body.get("extra_body")
    if isinstance(extra_body, dict):
        for k, v in extra_body.items():
            openai_dict.setdefault(k, v)

    if anthropic_body.get("stop_sequences") is not None:
        openai_dict["stop"] = anthropic_body["stop_sequences"]
    tool_choice = anthropic_body.get("tool_choice")
    if isinstance(tool_choice, dict) and tool_choice.get("disable_parallel_tool_use") is True:
        openai_dict["parallel_tool_calls"] = False

    output_config = anthropic_body.get("output_config")
    if isinstance(output_config, dict):
        effort = output_config.get("effort")
        if effort in ("low", "medium", "high", "xhigh", "max"):
            openai_dict.setdefault("reasoning_effort", effort)

    _UNKNOWN_FIELDS = {"extra_body", "metadata", "anthropic_version", "cache_control"}
    dropped = [k for k in anthropic_body if k in _UNKNOWN_FIELDS]
    if dropped:
        logger.debug("Dropping Anthropic-only fields not forwarded to chat pipeline: %s", dropped)
    for key in dropped:
        openai_dict.pop(key, None)

    return openai_dict, tool_name_mapping


def _restore_tool_result_image_urls(openai_dict: Dict[str, Any], tool_result_image_urls: Dict[str, set[str]]) -> None:
    """LiteLLM flattens Anthropic tool_result image blocks into data URL text.

    LightLLM's multimodal path only sees OpenAI image_url content parts, so
    restore image-only tool results to that shape before constructing the
    ChatCompletionRequest.
    """
    for msg in openai_dict.get("messages") or []:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        if content not in tool_result_image_urls.get(msg.get("tool_call_id"), set()):
            continue
        msg["content"] = [{"type": "image_url", "image_url": {"url": content}}]


def _pdf_data_url_to_anthropic_parts(data_url: str, deadline: float | None = None) -> list[Dict[str, Any]]:
    _ensure_pdf_parsing_supported()
    pdf_bytes = _decode_pdf_data_url(data_url)
    if deadline is None:
        deadline = time.monotonic() + _PDF_SUBPROCESS_TIMEOUT_SECONDS
    page_count = _ensure_pdf_page_limit(pdf_bytes, deadline)
    return _pdf_bytes_to_anthropic_parts(pdf_bytes, page_count, deadline)


def _pdf_bytes_to_anthropic_parts(pdf_bytes: bytes, page_count: int, deadline: float) -> list[Dict[str, Any]]:
    if _is_vision_enabled():
        pages = _render_pdf_pages_to_png_b64(pdf_bytes, page_count, deadline)
        if pages:
            return [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": page}}
                for page in pages
            ]
    return [_pdf_bytes_to_text_part(pdf_bytes, deadline)]


def check_pdf_parsing_supported_at_startup() -> None:
    if _is_pdf_parsing_enabled():
        _ensure_pdf_tools_installed()


def _ensure_pdf_parsing_supported() -> None:
    if not _is_pdf_parsing_enabled():
        raise RuntimeError(
            f"PDF document parsing is disabled. Set {_PDF_PARSING_ENV}=1 to enable it; "
            "requires Poppler tools: pdftotext, pdftoppm, pdfinfo."
        )
    _ensure_pdf_tools_installed()


def _is_pdf_parsing_enabled() -> bool:
    return os.getenv(_PDF_PARSING_ENV, "1").upper() in {"1", "TRUE", "ON"}


def _ensure_pdf_tools_installed() -> None:
    missing = [tool for tool in _PDF_REQUIRED_TOOLS if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(
            "PDF document parsing requires Poppler tools. "
            f"Missing: {', '.join(missing)}. Required: {', '.join(_PDF_REQUIRED_TOOLS)}."
        )


def _ensure_pdf_page_limit(pdf_bytes: bytes, deadline: float | None = None) -> int:
    page_count = _pdf_page_count(pdf_bytes, deadline)
    if page_count is None:
        raise ValueError("Unable to determine PDF page count")
    if page_count > _PDF_MAX_RENDER_PAGES:
        raise ValueError(
            f"PDF document has {page_count} pages and exceeds configured page limit of {_PDF_MAX_RENDER_PAGES}"
        )
    return page_count


def _pdf_page_count(pdf_bytes: bytes, deadline: float | None = None) -> int | None:
    pdfinfo = shutil.which("pdfinfo")
    if not pdfinfo:
        return None
    timeout = _remaining_pdf_time(deadline)
    if timeout <= 0:
        return None
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        try:
            proc = subprocess.run(
                [pdfinfo, tmp_path],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
        except Exception:
            return None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    if proc.returncode != 0:
        return None
    for line in proc.stdout.decode("utf-8", "replace").splitlines():
        key, _, value = line.partition(":")
        if key == "Pages":
            try:
                return int(value.strip())
            except ValueError:
                return None
    return None


def _pdf_bytes_to_text_part(pdf_bytes: bytes, deadline: float | None = None) -> Dict[str, str]:
    text = _extract_pdf_text(pdf_bytes, deadline).strip()
    if not text:
        text = (
            "[PDF document attached, but no extractable text was found. "
            "This backend does not OCR scanned PDFs in the Anthropic adapter. "
            "Ask the client to read specific pages or use an OCR/vision-enabled backend.]"
        )
    return {"type": "text", "text": f"[PDF extracted text]\n{text}"}


def _is_vision_enabled() -> bool:
    try:
        args = get_env_start_args()
    except Exception:
        return False
    return bool(getattr(args, "enable_multimodal", False) and not getattr(args, "disable_vision", True))


def _decode_pdf_data_url(data_url: str) -> bytes:
    try:
        _, encoded = data_url.split("base64,", 1)
    except ValueError as exc:
        raise ValueError("Invalid base64 PDF document block") from exc
    if len(encoded) > ((_PDF_MAX_BYTES + 2) // 3 * 4 + 4):
        raise ValueError("PDF document block exceeds configured size limit")
    try:
        pdf_bytes = base64.b64decode(encoded, validate=True)
    except binascii.Error as exc:
        raise ValueError("Invalid base64 PDF document block") from exc
    if len(pdf_bytes) > _PDF_MAX_BYTES:
        raise ValueError("PDF document block exceeds configured size limit")
    return pdf_bytes


def _remaining_pdf_time(deadline: float | None) -> float:
    if deadline is None:
        return _PDF_SUBPROCESS_TIMEOUT_SECONDS
    return max(0.0, deadline - time.monotonic())


def _kill_pdf_process(proc: subprocess.Popen) -> None:
    try:
        proc.kill()
    except OSError:
        pass
    try:
        proc.wait()
    except OSError:
        pass


def _extract_pdf_text(pdf_bytes: bytes, deadline: float | None = None) -> str:
    key = _pdf_cache_key("text", pdf_bytes)
    cached = _pdf_cache_get(key)
    if cached is not _PDF_CACHE_MISS:
        return cached

    pdftotext = shutil.which("pdftotext")
    if not pdftotext:
        text = ""
    else:
        try:
            text = _extract_pdf_text_with_pdftotext(pdftotext, pdf_bytes, deadline)
        except Exception:
            text = ""
        text = text if text.strip() else ""
    if text:
        _pdf_cache_set(key, text)
    return text


def _extract_pdf_text_with_pdftotext(pdftotext: str, pdf_bytes: bytes, deadline: float | None = None) -> str:
    if deadline is None:
        deadline = time.monotonic() + _PDF_SUBPROCESS_TIMEOUT_SECONDS
    if _remaining_pdf_time(deadline) <= 0:
        return ""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        proc = subprocess.Popen(
            [pdftotext, "-layout", tmp_path, "-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        try:
            output_parts = []
            output_size = 0
            while True:
                timeout = _remaining_pdf_time(deadline)
                if timeout <= 0:
                    _kill_pdf_process(proc)
                    return ""
                readable, _, _ = select.select([proc.stdout], [], [], timeout)
                if not readable:
                    _kill_pdf_process(proc)
                    return ""
                chunk = os.read(
                    proc.stdout.fileno(),
                    min(64 * 1024, _PDF_MAX_TEXT_BYTES + 1 - output_size),
                )
                if not chunk:
                    break
                output_parts.append(chunk)
                output_size += len(chunk)
                if output_size > _PDF_MAX_TEXT_BYTES:
                    _kill_pdf_process(proc)
                    return ""
            try:
                proc.wait(timeout=_remaining_pdf_time(deadline))
            except subprocess.TimeoutExpired:
                _kill_pdf_process(proc)
                return ""
        except Exception:
            _kill_pdf_process(proc)
            return ""
        finally:
            proc.stdout.close()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    if proc.returncode != 0:
        return ""
    output = b"".join(output_parts)
    return output.decode("utf-8", "replace")


def _render_pdf_pages_to_png_b64(
    pdf_bytes: bytes,
    page_count: int = _PDF_MAX_RENDER_PAGES,
    deadline: float | None = None,
) -> tuple[str, ...]:
    key = _pdf_cache_key("vision", pdf_bytes)
    cached = _pdf_cache_get(key)
    if cached is not _PDF_CACHE_MISS:
        return cached

    pdftoppm = shutil.which("pdftoppm")
    if not pdftoppm:
        return ()
    if deadline is None:
        deadline = time.monotonic() + _PDF_SUBPROCESS_TIMEOUT_SECONDS
    timeout = _remaining_pdf_time(deadline)
    if timeout <= 0:
        return ()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        with tempfile.TemporaryDirectory() as out_dir:
            prefix = os.path.join(out_dir, "page")
            pages = []
            rendered_bytes = 0
            try:
                proc = subprocess.run(
                    [
                        pdftoppm,
                        "-png",
                        "-scale-to",
                        str(_PDF_MAX_RENDER_DIMENSION),
                        "-f",
                        "1",
                        "-l",
                        str(page_count),
                        tmp_path,
                        prefix,
                    ],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=timeout,
                )
            except Exception:
                return ()
            if proc.returncode != 0:
                return ()
            for page_path in _rendered_pdf_page_paths(out_dir):
                try:
                    page_size = os.path.getsize(page_path)
                except OSError:
                    break
                if page_size > _PDF_MAX_RENDER_PAGE_BYTES or rendered_bytes + page_size > _PDF_MAX_RENDERED_BYTES:
                    return ()
                with open(page_path, "rb") as f:
                    page = f.read()
                if len(page) != page_size:
                    return ()
                rendered_bytes += page_size
                pages.append(base64.b64encode(page).decode("ascii"))
            pages = tuple(pages)
            if pages:
                _pdf_cache_set(key, pages)
            return pages
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _rendered_pdf_page_paths(out_dir: str) -> list[str]:
    numbered_paths = []
    for name in os.listdir(out_dir):
        if not name.startswith("page-") or not name.endswith(".png"):
            continue
        try:
            page_number = int(name[len("page-") : -len(".png")])
        except ValueError:
            continue
        numbered_paths.append((page_number, os.path.join(out_dir, name)))
    return [path for _, path in sorted(numbered_paths)]


def _pdf_cache_key(kind: str, pdf_bytes: bytes) -> tuple[str, bytes, int]:
    return (kind, hashlib.sha256(pdf_bytes).digest(), len(pdf_bytes))


def _pdf_cache_get(key: tuple[str, bytes, int]) -> Any:
    with _PDF_CACHE_LOCK:
        item = _PDF_CACHE.get(key)
        if item is None:
            return _PDF_CACHE_MISS
        _PDF_CACHE.move_to_end(key)
        return item[1]


def _pdf_cache_set(key: tuple[str, bytes, int], value: Any) -> None:
    global _PDF_CACHE_BYTES
    size = _pdf_cache_size(value)
    if size == 0 or size > _PDF_CACHE_MAX_BYTES:
        return
    with _PDF_CACHE_LOCK:
        old = _PDF_CACHE.pop(key, None)
        if old is not None:
            _PDF_CACHE_BYTES -= old[0]
        while _PDF_CACHE_BYTES + size > _PDF_CACHE_MAX_BYTES and _PDF_CACHE:
            _, (old_size, _) = _PDF_CACHE.popitem(last=False)
            _PDF_CACHE_BYTES -= old_size
        _PDF_CACHE[key] = (size, value)
        _PDF_CACHE_BYTES += size


def _pdf_cache_size(value: Any) -> int:
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, tuple):
        return sum(len(item.encode("utf-8")) for item in value if isinstance(item, str))
    return 0


def _clear_pdf_cache() -> None:
    global _PDF_CACHE_BYTES
    with _PDF_CACHE_LOCK:
        _PDF_CACHE.clear()
        _PDF_CACHE_BYTES = 0


_extract_pdf_text.cache_clear = _clear_pdf_cache
_render_pdf_pages_to_png_b64.cache_clear = _clear_pdf_cache


def _replace_anthropic_pdf_documents(anthropic_body: Dict[str, Any]) -> None:
    pdf_blocks = list(_iter_anthropic_pdf_documents(anthropic_body))
    if not pdf_blocks:
        return
    if any(block["source"].get("type") == "url" for block in pdf_blocks):
        raise ValueError("URL-backed PDF document blocks are not supported; send the PDF as base64 instead")
    _ensure_pdf_parsing_supported()
    if len(pdf_blocks) > _PDF_MAX_DOCUMENTS:
        raise ValueError(
            f"Request contains {len(pdf_blocks)} PDF documents and exceeds configured document limit "
            f"of {_PDF_MAX_DOCUMENTS}"
        )

    pdf_bytes_list = []
    total_bytes = 0
    for block in pdf_blocks:
        pdf_bytes = _decode_pdf_data_url(f"data:application/pdf;base64,{block['source']['data']}")
        total_bytes += len(pdf_bytes)
        if total_bytes > _PDF_MAX_BYTES:
            raise ValueError("PDF documents exceed configured request size limit")
        pdf_bytes_list.append(pdf_bytes)

    deadline = time.monotonic() + _PDF_SUBPROCESS_TIMEOUT_SECONDS
    page_counts = []
    total_pages = 0
    for pdf_bytes in pdf_bytes_list:
        page_count = _ensure_pdf_page_limit(pdf_bytes, deadline)
        total_pages += page_count
        if total_pages > _PDF_MAX_RENDER_PAGES:
            raise ValueError(
                f"PDF documents have {total_pages} pages and exceed configured request page limit "
                f"of {_PDF_MAX_RENDER_PAGES}"
            )
        page_counts.append(page_count)

    replacements = {}
    converted_bytes = 0
    for block, pdf_bytes, page_count in zip(pdf_blocks, pdf_bytes_list, page_counts):
        replacement = _pdf_bytes_to_anthropic_parts(pdf_bytes, page_count, deadline)
        converted_bytes += _pdf_converted_size(replacement)
        if converted_bytes > _PDF_MAX_CONVERTED_BYTES:
            raise ValueError("PDF conversions exceed configured request output size limit")
        replacements[id(block)] = replacement
    for message in anthropic_body.get("messages") or []:
        if isinstance(message, dict):
            _replace_pdf_documents_in_content(message.get("content"), replacements)


def _pdf_converted_size(parts: list[Dict[str, Any]]) -> int:
    size = 0
    for part in parts:
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            size += len(part["text"].encode("utf-8"))
            continue
        source = part.get("source")
        if (
            part.get("type") == "image"
            and isinstance(source, dict)
            and source.get("type") == "base64"
            and isinstance(source.get("data"), str)
        ):
            size += len(source["data"])
    return size


def _replace_pdf_documents_in_content(value: Any, replacements: Dict[int, list[Dict[str, Any]]]) -> None:
    if isinstance(value, list):
        new_items = []
        changed = False
        for item in value:
            replacement = replacements.get(id(item))
            if replacement is not None:
                new_items.extend(replacement)
                changed = True
            else:
                if isinstance(item, dict) and item.get("type") == "tool_result":
                    _replace_pdf_documents_in_content(item.get("content"), replacements)
                new_items.append(item)
        if changed:
            value[:] = new_items


def _iter_anthropic_pdf_documents(
    anthropic_body: Dict[str, Any],
):
    for message in anthropic_body.get("messages") or []:
        if isinstance(message, dict):
            yield from _iter_pdf_documents_in_content(message.get("content"))


def _iter_pdf_documents_in_content(value: Any):
    if not isinstance(value, list):
        return
    for item in value:
        if _is_anthropic_pdf_document(item):
            yield item
        elif isinstance(item, dict) and item.get("type") == "tool_result":
            yield from _iter_pdf_documents_in_content(item.get("content"))


def _tool_result_image_urls(anthropic_body: Dict[str, Any]) -> Dict[str, set[str]]:
    image_urls: Dict[str, set[str]] = {}
    for message in anthropic_body.get("messages") or []:
        if not isinstance(message, dict) or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            tool_use_id = block.get("tool_use_id")
            content = block.get("content")
            if not isinstance(tool_use_id, str) or not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "image":
                    continue
                source = item.get("source")
                if not isinstance(source, dict):
                    continue
                if source.get("type") == "url" and isinstance(source.get("url"), str) and source["url"]:
                    image_urls.setdefault(tool_use_id, set()).add(source["url"])
                elif (
                    source.get("type") == "base64"
                    and isinstance(source.get("media_type"), str)
                    and source["media_type"].startswith("image/")
                    and isinstance(source.get("data"), str)
                ):
                    image_urls.setdefault(tool_use_id, set()).add(
                        f"data:{source['media_type']};base64,{source['data']}"
                    )
    return image_urls


def _is_anthropic_pdf_document(value: Any) -> bool:
    if not isinstance(value, dict) or value.get("type") != "document":
        return False
    source = value.get("source")
    return isinstance(source, dict) and (
        (
            source.get("type") == "base64"
            and source.get("media_type") == "application/pdf"
            and isinstance(source.get("data"), str)
        )
        or (source.get("type") == "url" and isinstance(source.get("url"), str) and bool(source["url"]))
    )


# ---------------------------------------------------------------------------
# Response translation
# ---------------------------------------------------------------------------


_FINISH_REASON_TO_STOP_REASON = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    None: "end_turn",
}
_SYNTHETIC_THINKING_SIGNATURE = "lightllm"


def _extract_reasoning_text(openai_dict: Dict[str, Any]) -> str:
    """Return reasoning emitted by either of LightLLM's response fields."""
    choice = (openai_dict.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    for key in ("reasoning_content", "reasoning"):
        value = message.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _prepend_thinking_block(result: Dict[str, Any], reasoning_text: str) -> None:
    """Expose OpenAI reasoning as the Anthropic extended-thinking block."""
    if not reasoning_text:
        return
    content = result.setdefault("content", [])
    if any(isinstance(block, dict) and block.get("type") == "thinking" for block in content):
        return
    content.insert(
        0,
        {
            "type": "thinking",
            "thinking": reasoning_text,
            "signature": _SYNTHETIC_THINKING_SIGNATURE,
        },
    )


def _chat_response_to_anthropic(
    chat_response: Any,
    tool_name_mapping: Dict[str, str],
    requested_model: str,
    *,
    stop_sequence: str | None = None,
) -> Dict[str, Any]:
    """Wrap a LightLLM ``ChatCompletionResponse`` into an Anthropic
    Messages response dict.

    LiteLLM's ``translate_openai_response_to_anthropic`` requires a
    ``litellm.ModelResponse`` object (discovered via Task 3's characterisation
    test). We construct one from the LightLLM response's dict form.
    """
    adapter = get_anthropic_messages_adapter()
    if hasattr(chat_response, "model_dump"):
        openai_dict = chat_response.model_dump(exclude_none=True)
    else:
        openai_dict = dict(chat_response)
    if stop_sequence is not None:
        openai_dict["choices"][0]["stop_reason"] = stop_sequence
    reasoning_text = _extract_reasoning_text(openai_dict)

    # LiteLLM currently consumes ``reasoning_content`` but not LightLLM's
    # legacy ``reasoning`` field.  Normalise both names before translation;
    # the post-translation guard below covers adapters that drop the field.
    if reasoning_text:
        message = ((openai_dict.get("choices") or [{}])[0]).get("message") or {}
        message.setdefault("reasoning_content", reasoning_text)

    try:
        # Lazy import so this module stays importable when litellm is absent.
        from litellm import ModelResponse  # type: ignore

        model_response = ModelResponse(**openai_dict)
        anthropic_obj = adapter.translate_openai_response_to_anthropic(model_response, tool_name_mapping)
    except Exception as exc:
        logger.warning("LiteLLM response translation failed (%s); using fallback", exc)
        return _fallback_openai_to_anthropic(openai_dict, requested_model)

    if hasattr(anthropic_obj, "model_dump"):
        result = anthropic_obj.model_dump(exclude_none=True)
    else:
        result = dict(anthropic_obj)

    result = _normalize_anthropic_response(result, requested_model)
    _prepend_thinking_block(result, reasoning_text)
    if result.get("stop_reason") == "end_turn":
        choice = openai_dict["choices"][0]
        if isinstance(choice.get("stop_reason"), str):
            result["stop_reason"] = "stop_sequence"
            result["stop_sequence"] = choice["stop_reason"]
    return result


def _normalize_anthropic_response(result: Dict[str, Any], requested_model: str) -> Dict[str, Any]:
    """Cosmetic clean-ups applied to every non-streaming Anthropic response:

    - echo the client-supplied model name (LiteLLM sometimes emits the
      upstream model id instead);
    - force the Anthropic ``msg_`` id prefix (LiteLLM passes LightLLM's
      raw numeric request id through, which confuses strict clients);
    - set default ``type`` / ``role`` / ``stop_sequence`` when missing;
    - drop empty text blocks (LiteLLM sometimes produces a leading
      ``{"type":"text","text":""}`` before a tool_use block);
    - strip the LiteLLM-specific ``provider_specific_fields`` leak from
      every content block.
    """
    result["model"] = requested_model

    if not str(result.get("id", "")).startswith("msg_"):
        result["id"] = f"msg_{uuid.uuid4().hex[:24]}"
    result.setdefault("type", "message")
    result.setdefault("role", "assistant")
    result.setdefault("stop_sequence", None)

    cleaned_content = []
    for block in result.get("content") or []:
        if not isinstance(block, dict):
            cleaned_content.append(block)
            continue
        if block.get("type") == "text" and not block.get("text"):
            continue
        if block.get("type") == "thinking":
            signature = block.get("signature")
            if not isinstance(signature, str) or not signature:
                block["signature"] = _SYNTHETIC_THINKING_SIGNATURE
        block.pop("provider_specific_fields", None)
        cleaned_content.append(block)
    result["content"] = cleaned_content

    return result


def _fallback_openai_to_anthropic(openai_dict: Dict[str, Any], requested_model: str) -> Dict[str, Any]:
    """Minimal hand-built OpenAI->Anthropic translation for text-only responses.

    Used only when LiteLLM's adapter raises on the response path. Does
    not support tool_use; errors out loudly if tool calls are present
    since silently dropping them would corrupt the response.
    """
    choice = (openai_dict.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    if message.get("tool_calls"):
        raise RuntimeError("Fallback translator cannot handle tool_calls; LiteLLM adapter path is required.")
    text = message.get("content") or ""
    reasoning_text = _extract_reasoning_text(openai_dict)
    content = []
    if reasoning_text:
        content.append(
            {
                "type": "thinking",
                "thinking": reasoning_text,
                "signature": _SYNTHETIC_THINKING_SIGNATURE,
            }
        )
    if text:
        content.append({"type": "text", "text": text})
    usage = openai_dict.get("usage") or {}
    finish_reason = choice.get("finish_reason")
    stop_sequence = choice.get("stop_reason") if finish_reason == "stop" else None
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": requested_model,
        "content": content,
        "stop_reason": (
            "stop_sequence"
            if stop_sequence is not None
            else _FINISH_REASON_TO_STOP_REASON.get(finish_reason, "end_turn")
        ),
        "stop_sequence": stop_sequence,
        "usage": _anthropic_usage(usage),
    }


# ---------------------------------------------------------------------------
# Streaming bridge
# ---------------------------------------------------------------------------


def _anthropic_usage(usage: dict) -> dict:
    cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    return {
        "input_tokens": max(0, int(usage.get("prompt_tokens") or 0) - cached),
        "output_tokens": int(usage.get("completion_tokens") or 0),
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": cached,
    }


def _sse_event(event_type: str, data_obj: Dict[str, Any]) -> bytes:
    """Encode an Anthropic-style SSE event."""
    return f"event: {event_type}\ndata: {json.dumps(data_obj)}\n\n".encode("utf-8")


def _content_block_close_events(current_open: tuple[str, int]) -> list[bytes]:
    """Return the protocol events required to close an Anthropic block."""
    block_type, block_index = current_open
    events = []
    if block_type == "thinking":
        events.append(
            _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {
                        "type": "signature_delta",
                        "signature": _SYNTHETIC_THINKING_SIGNATURE,
                    },
                },
            )
        )
    events.append(
        _sse_event(
            "content_block_stop",
            {"type": "content_block_stop", "index": block_index},
        )
    )
    return events


def _anthropic_error_type(error: Dict[str, Any]) -> str:
    error_type = error.get("type")
    if is_rate_limit_error(error):
        return "rate_limit_error"
    if error_type == "invalid_request_error":
        return error_type
    return "api_error"


@dataclass
class _AnthropicStreamBlock:
    kind: str
    parts: list[str] = field(default_factory=list)
    sent: int = 0
    index: int | None = None
    tool_id: str | None = None
    name: str | None = None
    closed: bool = False

    def has_complete_input(self) -> bool:
        last_part = next((part for part in reversed(self.parts) if part.strip()), "")
        if not last_part.rstrip().endswith("}"):
            return False
        try:
            return isinstance(json.loads("".join(self.parts)), dict)
        except ValueError:
            return False


async def _openai_sse_to_anthropic_events(
    openai_body_iterator,
    requested_model: str,
    message_id: str,
    tool_name_mapping: dict[str, str] | None = None,
    parallel_tool_calls: bool = True,
):
    """Emit each Anthropic content block once, in first-appearance order."""
    blocks: deque[_AnthropicStreamBlock] = deque()
    tools: dict[int, _AnthropicStreamBlock] = {}
    next_index = 0
    message_started = False
    usage = _anthropic_usage({})
    finish_reason = stop_sequence = None
    error = None
    tool_name_mapping = tool_name_mapping or {}

    def emit_blocks(final=False):
        nonlocal next_index
        while blocks:
            block = blocks[0]
            if block.kind == "tool_use" and block.name is None:
                return
            if block.index is None:
                block.index = next_index
                next_index += 1
                if block.kind == "tool_use":
                    content = {
                        "type": "tool_use",
                        "id": block.tool_id or f"toolu_{uuid.uuid4().hex[:24]}",
                        "name": tool_name_mapping.get(block.name, block.name),
                        "input": {},
                    }
                elif block.kind == "thinking":
                    content = {"type": "thinking", "thinking": "", "signature": ""}
                else:
                    content = {"type": "text", "text": ""}
                yield _sse_event(
                    "content_block_start",
                    {"type": "content_block_start", "index": block.index, "content_block": content},
                )

            for part in block.parts[block.sent :]:
                if block.kind == "tool_use":
                    delta = {"type": "input_json_delta", "partial_json": part}
                else:
                    delta = {"type": f"{block.kind}_delta", block.kind: part}
                yield _sse_event(
                    "content_block_delta", {"type": "content_block_delta", "index": block.index, "delta": delta}
                )
            if block.kind == "tool_use":
                block.sent = len(block.parts)
            else:
                block.parts.clear()

            if not final:
                if len(blocks) == 1:
                    return
                # An OpenAI stream can interleave indices. Keep delivering the
                # active tool while later blocks wait for its JSON to finish.
                if block.kind == "tool_use" and not block.has_complete_input():
                    return
            block.closed = True
            block.parts.clear()
            blocks.popleft()
            yield from _content_block_close_events((block.kind, block.index))

    upstream = aiter(openai_body_iterator)
    while True:
        try:
            raw_chunk = await anext(upstream)
        except StopAsyncIteration:
            break
        except RuntimeError as exc:
            logger.exception("Anthropic streaming generation failed")
            if not message_started:
                raise GenerationError("Generation failed") from exc
            error = {"type": "api_error", "message": "Generation failed"}
            break
        if not raw_chunk:
            continue
        if isinstance(raw_chunk, (bytes, bytearray)):
            raw_chunk = raw_chunk.decode("utf-8")
        # The chat endpoint can batch several complete SSE frames together.
        for line in raw_chunk.split("\n"):
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                continue
            try:
                chunk = json.loads(payload)
            except ValueError:
                error = {"type": "api_error", "message": "Invalid upstream stream payload"}
                break
            if "error" in chunk and "choices" not in chunk:
                error = chunk["error"]
                break

            if chunk.get("usage"):
                usage = _anthropic_usage(chunk["usage"])
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
                stop_sequence = choice.get("stop_reason")
                if finish_reason == "abort" and isinstance(stop_sequence, str):
                    finish_reason = "stop"
                if finish_reason in ("error", "abort"):
                    message = "Generation failed" if finish_reason == "error" else "Generation was aborted"
                    error = {"type": "api_error", "message": message}
                    break

            if not message_started:
                message_started = True
                yield _sse_event(
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {
                            "id": message_id,
                            "type": "message",
                            "role": "assistant",
                            "model": requested_model,
                            "content": [],
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": dict(usage, output_tokens=0),
                        },
                    },
                )

            delta = choice.get("delta") or {}
            for kind, part in (
                ("thinking", delta.get("reasoning_content") or delta.get("reasoning")),
                ("text", delta.get("content")),
            ):
                if part:
                    if not blocks or blocks[-1].kind != kind:
                        blocks.append(_AnthropicStreamBlock(kind))
                    blocks[-1].parts.append(part)

            for call in delta.get("tool_calls") or []:
                tc_index = call.get("index", 0)
                if tc_index not in tools:
                    if not parallel_tool_calls and tools:
                        continue
                    tools[tc_index] = block = _AnthropicStreamBlock("tool_use")
                    blocks.append(block)
                block = tools[tc_index]
                function = call.get("function") or {}
                arguments = function.get("arguments") or ""
                if block.closed:
                    if arguments.strip():
                        error = {"type": "api_error", "message": "Arguments arrived after the tool call completed"}
                        break
                    continue
                block.tool_id = block.tool_id or call.get("id")
                block.name = block.name or function.get("name")
                if arguments:
                    block.parts.append(arguments)
            if error:
                break
            for event in emit_blocks():
                yield event
        if error:
            break

    if error is None and finish_reason not in ("stop", "length", "tool_calls"):
        error = {"type": "api_error", "message": "Generation ended without a finish status"}
    matched_stop = stop_sequence if isinstance(stop_sequence, str) else None
    incomplete_tool = any(
        not block.name or (bool(block.parts) and not block.has_complete_input())
        for block in blocks
        if block.kind == "tool_use"
    )
    if error is None and finish_reason == "tool_calls" and incomplete_tool and matched_stop is None:
        error = {"type": "api_error", "message": "Generation ended with incomplete tool arguments"}
    if error is not None:
        if blocks and blocks[0].index is not None:
            for event in _content_block_close_events((blocks[0].kind, blocks[0].index)):
                yield event
        yield _sse_event(
            "error",
            {
                "type": "error",
                "error": {
                    "type": _anthropic_error_type(error),
                    "message": error.get("message", "Generation failed"),
                },
            },
        )
        if message_started:
            yield _sse_event("message_stop", {"type": "message_stop"})
        return

    for event in emit_blocks(final=True):
        yield event
    if finish_reason == "length":
        stop_reason, matched_stop = "max_tokens", None
    elif finish_reason == "tool_calls" and not incomplete_tool:
        stop_reason, matched_stop = "tool_use", None
    else:
        stop_reason = "stop_sequence" if matched_stop is not None else "end_turn"
    yield _sse_event(
        "message_delta",
        {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": matched_stop}, "usage": usage},
    )
    yield _sse_event("message_stop", {"type": "message_stop"})


# ---------------------------------------------------------------------------
# Error response helper
# ---------------------------------------------------------------------------


# HTTP status → Anthropic error type. Derived from
# https://docs.anthropic.com/en/api/errors ; values outside this map fall
# back to "api_error".
_STATUS_TO_ERROR_TYPE = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    429: "rate_limit_error",
    500: "api_error",
    529: "overloaded_error",
}


def _anthropic_error_response(status: HTTPStatus, message: str) -> JSONResponse:
    """Return an Anthropic-shaped error envelope.

    Anthropic clients (including Claude Code) parse the {"type":"error",
    "error":{"type":..., "message":...}} shape; the OpenAI-style envelope
    from create_error_response hides the real message from them.
    """
    err_type = _STATUS_TO_ERROR_TYPE.get(int(status), "api_error")
    return JSONResponse(
        {"type": "error", "error": {"type": err_type, "message": message}},
        status_code=int(status),
    )


def _rewrap_openai_error_as_anthropic(resp: JSONResponse) -> JSONResponse:
    """Convert an OpenAI-format JSONResponse produced by create_error_response
    into Anthropic's error envelope. Best-effort: if we can't decode the body
    we leave the response alone so the caller still sees something."""
    try:
        body = json.loads(bytes(resp.body).decode("utf-8"))
        inner = (body or {}).get("error") or {}
        message = inner.get("message") or "request failed"
    except Exception:
        return resp
    return _anthropic_error_response(HTTPStatus(resp.status_code), message)


# ---------------------------------------------------------------------------
# HTTP entry point
# ---------------------------------------------------------------------------


async def anthropic_messages_impl(raw_request: Request) -> Response:
    # Lazy imports to avoid pulling in heavy server deps at module import time.
    from .api_models import ChatCompletionRequest, ChatCompletionResponse
    from .api_openai import chat_completions_impl

    try:
        raw_body = await raw_request.json()
    except Exception as exc:
        return _anthropic_error_response(HTTPStatus.BAD_REQUEST, f"Invalid JSON body: {exc}")

    if not isinstance(raw_body, dict):
        return _anthropic_error_response(HTTPStatus.BAD_REQUEST, "Request body must be a JSON object")

    requested_model = raw_body.get("model", "default")
    is_stream = bool(raw_body.get("stream"))

    try:
        chat_dict, tool_name_mapping = await asyncio.to_thread(_anthropic_to_chat_request, raw_body)
    except Exception as exc:
        logger.exception("Failed to translate Anthropic request")
        return _anthropic_error_response(HTTPStatus.BAD_REQUEST, f"Request translation failed: {exc}")

    # Force the downstream path to stream if the client asked for stream.
    chat_dict["stream"] = is_stream

    try:
        chat_request = ChatCompletionRequest(**chat_dict)
    except Exception as exc:
        logger.exception("Failed to build ChatCompletionRequest")
        return _anthropic_error_response(HTTPStatus.BAD_REQUEST, f"Invalid request after translation: {exc}")

    try:
        downstream = await chat_completions_impl(
            chat_request, raw_request, preserve_abort=True, include_initial_usage=is_stream
        )
    except RuntimeError as exc:
        logger.exception("Anthropic generation failed")
        raise GenerationError("Generation failed") from exc

    if is_stream:
        from .api_stream_obj import CustomStreamingResponse

        if not isinstance(downstream, CustomStreamingResponse):
            # chat_completions_impl returned an OpenAI-format error — rewrap it.
            if isinstance(downstream, JSONResponse):
                return _rewrap_openai_error_as_anthropic(downstream)
            return downstream

        message_id = f"msg_{uuid.uuid4().hex[:24]}"

        async def anthropic_stream():
            async with aclosing(downstream.body_iterator) as upstream:
                async for event in _openai_sse_to_anthropic_events(
                    upstream,
                    requested_model,
                    message_id,
                    tool_name_mapping,
                    parallel_tool_calls=chat_request.parallel_tool_calls is not False,
                ):
                    yield event

        return CustomStreamingResponse(anthropic_stream(), media_type="text/event-stream")

    if not isinstance(downstream, ChatCompletionResponse):
        if isinstance(downstream, JSONResponse):
            return _rewrap_openai_error_as_anthropic(downstream)
        return downstream

    choice = downstream.choices[0]
    if choice.finish_reason == "abort" and choice.stop_reason is not None:
        choice.finish_reason = "stop"
    if choice.finish_reason not in ("stop", "length", "tool_calls"):
        message = "Generation was aborted" if choice.finish_reason == "abort" else "Generation failed"
        return _anthropic_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, message)
    if chat_request.parallel_tool_calls is False and choice.message.tool_calls:
        choice.message.tool_calls = choice.message.tool_calls[:1]

    try:
        anthropic_dict = _chat_response_to_anthropic(
            downstream, tool_name_mapping, requested_model, stop_sequence=choice.stop_reason
        )
    except Exception as exc:
        logger.error("Failed to translate response to Anthropic format: %s", exc)
        return _anthropic_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
    return JSONResponse(anthropic_dict)


async def anthropic_count_tokens_impl(raw_request: Request) -> Response:
    """Count the fully rendered input for Anthropic's Messages API."""
    from .api_http import g_objs
    from .api_models import ChatCompletionRequest
    from .api_openai import _build_multimodal_params, _select_chat_template_tools
    from .build_prompt import build_prompt
    from .core.objs.sampling_params import SamplingParams

    raw_body = await raw_request.json()
    if not isinstance(raw_body, dict):
        raise ValueError("Request body must be a JSON object")
    if "messages" not in raw_body:
        raise ValueError("Missing field: messages")
    # LiteLLM validates the body as a Messages creation request, where
    # max_tokens is required. Anthropic's count_tokens request omits it;
    # the value has no effect on prompt rendering, so provide a sentinel.
    raw_body.setdefault("max_tokens", 1)
    try:
        chat_dict, _ = await asyncio.to_thread(_anthropic_to_chat_request, raw_body)
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Invalid Anthropic request: {exc}") from exc
    chat_request = ChatCompletionRequest(**chat_dict)

    prompt = await build_prompt(chat_request, _select_chat_template_tools(chat_request))
    multimodal_params = _build_multimodal_params(chat_request)
    await multimodal_params.verify_and_preload(raw_request)

    sampling_params = SamplingParams()
    sampling_params.init(tokenizer=g_objs.httpserver_manager.tokenizer, add_special_tokens=False)
    input_tokens = g_objs.httpserver_manager.tokens(
        prompt,
        multimodal_params,
        sampling_params,
        {"add_special_tokens": False},
    )

    return JSONResponse(
        {
            "input_tokens": input_tokens,
            "context_management": {"original_input_tokens": input_tokens},
        }
    )
