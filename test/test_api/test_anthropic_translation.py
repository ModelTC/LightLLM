"""Unit tests for the Anthropic API translation layer.

These tests import the translation helpers directly and do not require
a running LightLLM server. They do require 'litellm' to be installed —
tests are skipped if it is not available.
"""
import pytest

litellm = pytest.importorskip("litellm")


def test_shim_imports_adapter():
    from lightllm.server._litellm_shim import get_anthropic_messages_adapter

    adapter = get_anthropic_messages_adapter()
    assert hasattr(adapter, "translate_anthropic_to_openai")
    assert hasattr(adapter, "translate_openai_response_to_anthropic")


def test_shim_raises_clear_error_when_litellm_missing(monkeypatch):
    import sys

    from lightllm.server import _litellm_shim

    monkeypatch.setitem(sys.modules, "litellm", None)
    _litellm_shim._cached_adapter = None
    _litellm_shim._import_checked = False  # reset module-level cache

    with pytest.raises(RuntimeError, match="--enable_anthropic_api requires"):
        _litellm_shim.get_anthropic_messages_adapter()
