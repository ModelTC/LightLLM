"""LiteLLM integration shim for the Anthropic Messages API endpoint.

LiteLLM's Anthropic<->OpenAI translation code lives under an
``experimental_pass_through`` import path. Centralising all LiteLLM imports
here means a LiteLLM upgrade that relocates those symbols requires editing
exactly one file. Callers should use the getters below; they must not
import LiteLLM symbols directly from elsewhere in the server package.
"""
from __future__ import annotations

from typing import Any

from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

# Known-good LiteLLM versions. Bump explicitly after retesting.
_MIN_LITELLM_VERSION = "1.52.0"
_MAX_TESTED_LITELLM_VERSION = "1.84.0"

_cached_adapter: Any = None
_import_checked: bool = False


def _raise_missing() -> None:
    raise RuntimeError(
        "--enable_anthropic_api requires the 'litellm' package. Install it with:\n"
        f"    pip install 'litellm>={_MIN_LITELLM_VERSION}'"
    )


def _get_litellm_version() -> str:
    """Return the installed litellm version string, or 'unknown' if not found.

    litellm >= 1.x does not expose ``__version__`` as a module attribute;
    use importlib.metadata as the primary source.
    """
    try:
        import importlib.metadata
        return importlib.metadata.version("litellm")
    except Exception:
        pass
    # Fallback: some older builds do expose it.
    try:
        import litellm
        return getattr(litellm, "__version__", "unknown")
    except Exception:
        return "unknown"


def _check_import_once() -> None:
    global _import_checked
    if _import_checked:
        return
    try:
        import litellm  # noqa: F401
    except ImportError:
        _raise_missing()
    else:
        version = _get_litellm_version()
        logger.info(
            "LiteLLM detected (version=%s) for Anthropic API compatibility layer. "
            "Tested range: %s..%s",
            version,
            _MIN_LITELLM_VERSION,
            _MAX_TESTED_LITELLM_VERSION,
        )
    _import_checked = True


def get_anthropic_messages_adapter() -> Any:
    """Return a cached instance of LiteLLM's Anthropic<->OpenAI adapter.

    The returned object exposes ``translate_anthropic_to_openai`` and
    ``translate_openai_response_to_anthropic`` methods.
    """
    global _cached_adapter
    if _cached_adapter is not None:
        return _cached_adapter

    _check_import_once()
    try:
        from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
            LiteLLMAnthropicMessagesAdapter,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import LiteLLMAnthropicMessagesAdapter from LiteLLM. "
            "The experimental_pass_through module may have been relocated in a newer release. "
            f"Tested with LiteLLM {_MIN_LITELLM_VERSION}..{_MAX_TESTED_LITELLM_VERSION}. "
            f"To pin to a known-good version: pip install 'litellm<={_MAX_TESTED_LITELLM_VERSION}'. "
            f"Original error: {exc}"
        ) from exc

    _cached_adapter = LiteLLMAnthropicMessagesAdapter()
    return _cached_adapter


def ensure_available() -> None:
    """Eagerly verify LiteLLM is importable. Called once at server startup
    so that misconfiguration fails loudly instead of on the first request."""
    _check_import_once()
    get_anthropic_messages_adapter()
