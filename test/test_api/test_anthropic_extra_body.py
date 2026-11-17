"""Unit test for Anthropic -> OpenAI request translation with extra_body.

Verifies that ``extra_body.chat_template_kwargs`` (and other backend-specific
fields nested under ``extra_body`` per OpenAI SDK convention) survive the
/v1/messages request translation, so clients can opt out of model-default
thinking modes on engines that expose the toggle through
ChatCompletionRequest.chat_template_kwargs.

No server required — calls the pure translation helper directly.
"""

import asyncio
import base64
from contextlib import aclosing
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import ujson as json

pytest.importorskip("litellm")

import lightllm.server.api_anthropic as api_anthropic
from lightllm.server.api_anthropic import (
    _anthropic_to_chat_request,
    _fallback_openai_to_anthropic,
    _openai_sse_to_anthropic_events,
)
from lightllm.server.api_models import ChatCompletionRequest


def _base_body():
    return {
        "model": "test-model",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "hi"}],
    }


def _pdf_document_block():
    return {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": base64.b64encode(b"%PDF-1.4\n").decode("ascii"),
        },
    }


def _url_pdf_document_block():
    return {
        "type": "document",
        "source": {
            "type": "url",
            "url": "https://example.com/document.pdf",
        },
    }


def _enable_pdf_parsing(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_ANTHROPIC_ENABLE_PDF_PARSING", "1")
    monkeypatch.setattr(api_anthropic, "_ensure_pdf_tools_installed", lambda: None)
    monkeypatch.setattr(api_anthropic, "_pdf_page_count", lambda _, deadline=None: 1)


def _user_pdf_body():
    return {
        "model": "test-model",
        "max_tokens": 32,
        "messages": [
            {
                "role": "user",
                "content": [
                    _pdf_document_block(),
                    {"type": "text", "text": "What is in the PDF?"},
                ],
            }
        ],
    }


def _tool_result_body(content):
    return {
        "model": "test-model",
        "max_tokens": 32,
        "tools": [
            {
                "name": "Read",
                "description": "Read a file",
                "input_schema": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                    "required": ["file_path"],
                },
            }
        ],
        "messages": [
            {"role": "user", "content": "read a file"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_read",
                        "name": "Read",
                        "input": {"file_path": "file"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_read",
                        "content": content,
                    }
                ],
            },
        ],
    }


def test_extra_body_chat_template_kwargs_forwarded():
    body = _base_body()
    body["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    chat_dict, _ = _anthropic_to_chat_request(body)

    assert chat_dict.get("chat_template_kwargs") == {"enable_thinking": False}
    assert "extra_body" not in chat_dict


def test_native_thinking_parameter_is_forwarded():
    body = _base_body()
    body["thinking"] = {"type": "enabled", "budget_tokens": 128}

    chat_dict, _ = _anthropic_to_chat_request(body)

    assert chat_dict["chat_template_kwargs"]["enable_thinking"] is True
    assert chat_dict["separate_reasoning"] is True
    assert "thinking" not in chat_dict


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_native_output_config_effort_is_forwarded(effort):
    body = _base_body()
    body["output_config"] = {"effort": effort}

    chat_dict, _ = _anthropic_to_chat_request(body)

    assert ChatCompletionRequest(**chat_dict).reasoning_effort == effort


def test_replayed_thinking_is_preserved_through_request_validation():
    body = _tool_result_body("file contents")
    body["messages"][1]["content"].insert(
        0,
        {
            "type": "thinking",
            "thinking": "I should read the requested file.",
            "signature": "client-signature",
        },
    )

    chat_dict, _ = _anthropic_to_chat_request(body)
    assistant_dict = chat_dict["messages"][1]
    assert "thinking_blocks" not in assistant_dict
    assert assistant_dict["reasoning_content"] == "I should read the requested file."

    request = ChatCompletionRequest(**chat_dict)
    validated_assistant = request.messages[1].model_dump(exclude_none=True)
    assert validated_assistant["reasoning_content"] == "I should read the requested file."
    assert validated_assistant["tool_calls"]


def test_extra_body_multiple_fields_forwarded():
    body = _base_body()
    body["extra_body"] = {
        "chat_template_kwargs": {"enable_thinking": False},
        "do_sample": False,
        "top_k": 5,
    }

    chat_dict, _ = _anthropic_to_chat_request(body)

    assert chat_dict.get("chat_template_kwargs") == {"enable_thinking": False}
    assert chat_dict.get("do_sample") is False
    assert chat_dict.get("top_k") == 5


def test_top_level_openai_field_beats_extra_body_duplicate():
    # If a field ends up in openai_dict via the Anthropic->OpenAI translation
    # AND the same key appears in extra_body, the translation path wins.
    body = _base_body()
    body["temperature"] = 0.1  # translated by litellm -> openai_dict["temperature"] = 0.1
    body["extra_body"] = {"temperature": 0.9}

    chat_dict, _ = _anthropic_to_chat_request(body)

    assert chat_dict.get("temperature") == 0.1


def test_missing_extra_body_is_noop():
    body = _base_body()
    chat_dict, _ = _anthropic_to_chat_request(body)
    assert "extra_body" not in chat_dict
    assert "chat_template_kwargs" not in chat_dict


def test_fallback_response_exposes_reasoning_as_thinking_block():
    response = _fallback_openai_to_anthropic(
        {
            "choices": [
                {
                    "message": {
                        "reasoning": "First inspect the input.",
                        "content": "The answer is 42.",
                    },
                    "finish_reason": "stop",
                }
            ]
        },
        "test-model",
    )

    assert response["content"] == [
        {
            "type": "thinking",
            "thinking": "First inspect the input.",
            "signature": api_anthropic._SYNTHETIC_THINKING_SIGNATURE,
        },
        {"type": "text", "text": "The answer is 42."},
    ]


def test_litellm_response_translation_preserves_reasoning_field():
    from types import SimpleNamespace

    response = SimpleNamespace(
        model_dump=lambda exclude_none=True: {
            "id": "chatcmpl-test",
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "reasoning": "First inspect the input.",
                        "content": "The answer is 42.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        }
    )

    result = api_anthropic._chat_response_to_anthropic(response, {}, "test-model")

    assert result["content"][0]["type"] == "thinking"
    assert result["content"][0]["thinking"] == "First inspect the input."
    assert result["content"][0]["signature"] == api_anthropic._SYNTHETIC_THINKING_SIGNATURE
    assert result["content"][1] == {"type": "text", "text": "The answer is 42."}


def test_non_dict_extra_body_is_ignored():
    body = _base_body()
    body["extra_body"] = "not-a-dict"
    chat_dict, _ = _anthropic_to_chat_request(body)
    assert "extra_body" not in chat_dict


def test_pdf_document_block_becomes_text_not_pdf_image_url(monkeypatch):
    monkeypatch.delenv("LIGHTLLM_ANTHROPIC_ENABLE_PDF_PARSING", raising=False)
    monkeypatch.setattr(api_anthropic, "_ensure_pdf_tools_installed", lambda: None)
    monkeypatch.setattr(api_anthropic, "_pdf_page_count", lambda _, deadline=None: 1)
    monkeypatch.setattr(api_anthropic, "_is_vision_enabled", lambda: False)
    monkeypatch.setattr(api_anthropic, "_extract_pdf_text", lambda _, deadline=None: "PDF_SENTINEL_DIRECT")
    body = _user_pdf_body()

    chat_dict, _ = _anthropic_to_chat_request(body)

    content = chat_dict["messages"][0]["content"]
    assert content[0]["type"] == "text"
    assert "PDF_SENTINEL_DIRECT" in content[0]["text"]
    assert "data:application/pdf" not in json.dumps(chat_dict)


def test_pdf_document_block_can_be_disabled(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_ANTHROPIC_ENABLE_PDF_PARSING", "0")
    with pytest.raises(RuntimeError, match="PDF document parsing is disabled"):
        _anthropic_to_chat_request(_user_pdf_body())


def test_pdf_document_block_becomes_images_when_vision_enabled(monkeypatch):
    _enable_pdf_parsing(monkeypatch)
    monkeypatch.setattr(api_anthropic, "_is_vision_enabled", lambda: True)
    monkeypatch.setattr(
        api_anthropic,
        "_render_pdf_pages_to_png_b64",
        lambda _, page_count=None, deadline=None: ["UE5HMQ==", "UE5HMg=="],
    )
    body = _user_pdf_body()

    chat_dict, _ = _anthropic_to_chat_request(body)

    content = chat_dict["messages"][0]["content"]
    assert [p["type"] for p in content[:2]] == ["image_url", "image_url"]
    assert [p["image_url"]["url"] for p in content[:2]] == [
        "data:image/png;base64,UE5HMQ==",
        "data:image/png;base64,UE5HMg==",
    ]
    assert "data:application/pdf" not in json.dumps(chat_dict)
    assert "PDF extracted text" not in json.dumps(chat_dict)


def test_pdf_document_block_with_invalid_base64_fails_cleanly(monkeypatch):
    _enable_pdf_parsing(monkeypatch)
    body = {
        "model": "test-model",
        "max_tokens": 32,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": "not-base64!",
                        },
                    }
                ],
            }
        ],
    }

    with pytest.raises(ValueError, match="Invalid base64 PDF document block"):
        _anthropic_to_chat_request(body)


def test_pdf_document_block_over_size_fails_cleanly(monkeypatch):
    _enable_pdf_parsing(monkeypatch)
    monkeypatch.setattr(api_anthropic, "_PDF_MAX_BYTES", 4)

    with pytest.raises(ValueError, match="PDF document block exceeds configured size limit"):
        _anthropic_to_chat_request(_user_pdf_body())


def test_pdf_request_rejects_too_many_documents_before_page_checks(monkeypatch):
    _enable_pdf_parsing(monkeypatch)
    monkeypatch.setattr(api_anthropic, "_PDF_MAX_DOCUMENTS", 1)
    monkeypatch.setattr(
        api_anthropic,
        "_pdf_page_count",
        lambda *_: pytest.fail("page count should not run before document-count validation"),
    )
    body = _user_pdf_body()
    body["messages"][0]["content"].insert(0, _pdf_document_block())

    with pytest.raises(ValueError, match="document limit"):
        _anthropic_to_chat_request(body)


def test_pdf_request_enforces_total_byte_limit(monkeypatch):
    _enable_pdf_parsing(monkeypatch)
    monkeypatch.setattr(api_anthropic, "_PDF_MAX_BYTES", 10)
    monkeypatch.setattr(
        api_anthropic,
        "_pdf_page_count",
        lambda *_: pytest.fail("page count should not run before byte validation"),
    )
    first = _pdf_document_block()
    second = _pdf_document_block()
    first["source"]["data"] = base64.b64encode(b"123456").decode("ascii")
    second["source"]["data"] = base64.b64encode(b"abcdef").decode("ascii")
    body = _user_pdf_body()
    body["messages"][0]["content"] = [first, second]

    with pytest.raises(ValueError, match="request size limit"):
        _anthropic_to_chat_request(body)


def test_pdf_request_validates_total_pages_before_conversion(monkeypatch):
    _enable_pdf_parsing(monkeypatch)
    monkeypatch.setattr(api_anthropic, "_pdf_page_count", lambda *args: 11)
    monkeypatch.setattr(
        api_anthropic,
        "_pdf_bytes_to_anthropic_parts",
        lambda *_: pytest.fail("conversion should not run before page validation"),
    )
    body = _user_pdf_body()
    body["messages"][0]["content"].insert(0, _pdf_document_block())

    with pytest.raises(ValueError, match="request page limit"):
        _anthropic_to_chat_request(body)


def test_pdf_request_uses_one_subprocess_deadline(monkeypatch):
    _enable_pdf_parsing(monkeypatch)
    deadlines = []

    def fake_page_count(_pdf_bytes, deadline):
        deadlines.append(deadline)
        return 1

    def fake_convert(_pdf_bytes, _page_count, deadline):
        deadlines.append(deadline)
        return [{"type": "text", "text": "PDF"}]

    monkeypatch.setattr(api_anthropic, "_pdf_page_count", fake_page_count)
    monkeypatch.setattr(api_anthropic, "_pdf_bytes_to_anthropic_parts", fake_convert)
    body = _user_pdf_body()
    body["messages"][0]["content"].insert(0, _pdf_document_block())

    _anthropic_to_chat_request(body)

    assert len(deadlines) == 4
    assert len(set(deadlines)) == 1


@pytest.mark.parametrize(
    "replacement",
    [
        [{"type": "text", "text": "1234"}],
        [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "YWJj"}}],
    ],
    ids=["text", "rendered-image"],
)
def test_pdf_request_enforces_total_converted_output_limit(monkeypatch, replacement):
    _enable_pdf_parsing(monkeypatch)
    monkeypatch.setattr(api_anthropic, "_PDF_MAX_CONVERTED_BYTES", 7)
    calls = {"count": 0}

    def fake_convert(_pdf_bytes, _page_count, _deadline):
        calls["count"] += 1
        return replacement

    monkeypatch.setattr(api_anthropic, "_pdf_bytes_to_anthropic_parts", fake_convert)
    body = _user_pdf_body()
    body["messages"][0]["content"] = [
        _pdf_document_block(),
        _pdf_document_block(),
        _pdf_document_block(),
    ]

    with pytest.raises(ValueError, match="request output size limit"):
        _anthropic_to_chat_request(body)

    assert calls["count"] == 2


def test_url_pdf_document_block_is_rejected_explicitly():
    body = _user_pdf_body()
    body["messages"][0]["content"][0] = _url_pdf_document_block()

    with pytest.raises(ValueError, match="URL-backed PDF document blocks are not supported"):
        _anthropic_to_chat_request(body)


def test_tool_result_url_pdf_document_block_is_rejected_explicitly():
    body = _tool_result_body([_url_pdf_document_block()])

    with pytest.raises(ValueError, match="URL-backed PDF document blocks are not supported"):
        _anthropic_to_chat_request(body)


def test_tool_result_pdf_document_block_becomes_text_not_pdf_image_url(monkeypatch):
    _enable_pdf_parsing(monkeypatch)
    monkeypatch.setattr(api_anthropic, "_is_vision_enabled", lambda: False)
    monkeypatch.setattr(api_anthropic, "_extract_pdf_text", lambda _, deadline=None: "PDF_SENTINEL_TOOL")
    body = _tool_result_body([_pdf_document_block()])

    chat_dict, _ = _anthropic_to_chat_request(body)

    assert chat_dict["messages"][2]["role"] == "tool"
    assert "PDF_SENTINEL_TOOL" in chat_dict["messages"][2]["content"]
    assert "data:application/pdf" not in json.dumps(chat_dict)


def test_tool_result_pdf_document_block_becomes_images_when_vision_enabled(monkeypatch):
    _enable_pdf_parsing(monkeypatch)
    monkeypatch.setattr(api_anthropic, "_is_vision_enabled", lambda: True)
    monkeypatch.setattr(
        api_anthropic,
        "_render_pdf_pages_to_png_b64",
        lambda _, page_count=None, deadline=None: ["UE5HMQ=="],
    )
    body = _tool_result_body([_pdf_document_block()])

    chat_dict, _ = _anthropic_to_chat_request(body)

    assert chat_dict["messages"][2]["role"] == "tool"
    assert chat_dict["messages"][2]["content"] == [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,UE5HMQ=="},
        }
    ]
    assert "data:application/pdf" not in json.dumps(chat_dict)


def test_pdf_vision_render_limits_pages(monkeypatch):
    captured = []

    api_anthropic._render_pdf_pages_to_png_b64.cache_clear()

    def fake_run(cmd, **kwargs):
        captured.append(cmd)
        with open(f"{cmd[-1]}-1.png", "wb") as f:
            f.write(b"png")
        return type("Proc", (), {"returncode": 0})()

    monkeypatch.setattr(api_anthropic.shutil, "which", lambda _: "/usr/bin/pdftoppm")
    monkeypatch.setattr(api_anthropic.subprocess, "run", fake_run)

    assert api_anthropic._render_pdf_pages_to_png_b64(b"%PDF-1.4\n") == ("cG5n",)
    assert len(captured) == 1
    for cmd in captured:
        assert "-l" in cmd
        assert cmd[cmd.index("-f") + 1] == "1"
        assert cmd[cmd.index("-l") + 1] == str(api_anthropic._PDF_MAX_RENDER_PAGES)
        assert "-scale-to" in cmd
        assert "-scale-to-x" not in cmd
        assert "-scale-to-y" not in cmd
        assert str(api_anthropic._PDF_MAX_RENDER_DIMENSION) in cmd


def test_pdf_vision_reads_zero_padded_page_names(monkeypatch):
    api_anthropic._render_pdf_pages_to_png_b64.cache_clear()

    def fake_run(cmd, **kwargs):
        with open(f"{cmd[-1]}-01.png", "wb") as f:
            f.write(b"one")
        with open(f"{cmd[-1]}-02.png", "wb") as f:
            f.write(b"two")
        return type("Proc", (), {"returncode": 0})()

    monkeypatch.setattr(api_anthropic.shutil, "which", lambda _: "/usr/bin/pdftoppm")
    monkeypatch.setattr(api_anthropic.subprocess, "run", fake_run)

    assert api_anthropic._render_pdf_pages_to_png_b64(b"%PDF-padded\n", page_count=2) == (
        "b25l",
        "dHdv",
    )


def test_pdf_vision_rejects_large_rendered_page(monkeypatch):
    api_anthropic._render_pdf_pages_to_png_b64.cache_clear()
    monkeypatch.setattr(api_anthropic, "_PDF_MAX_RENDER_PAGE_BYTES", 2)

    def fake_run(cmd, **kwargs):
        with open(f"{cmd[-1]}-1.png", "wb") as f:
            f.write(b"png")
        return type("Proc", (), {"returncode": 0})()

    monkeypatch.setattr(api_anthropic.shutil, "which", lambda _: "/usr/bin/pdftoppm")
    monkeypatch.setattr(api_anthropic.subprocess, "run", fake_run)

    assert api_anthropic._render_pdf_pages_to_png_b64(b"%PDF-1.4\n") == ()


def test_pdf_text_extraction_limits_output(monkeypatch):
    class FakeStdout:
        def fileno(self):
            return 42

        def close(self):
            pass

    class FakeProcess:
        stdout = FakeStdout()
        returncode = -9
        killed = False

        def kill(self):
            self.killed = True

        def wait(self, **kwargs):
            pass

    process = FakeProcess()
    monkeypatch.setattr(api_anthropic, "_PDF_MAX_TEXT_BYTES", 4)
    monkeypatch.setattr(api_anthropic.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(api_anthropic.select, "select", lambda *args: ([process.stdout], [], []))
    monkeypatch.setattr(api_anthropic.os, "read", lambda _fd, size: b"x" * size)

    assert api_anthropic._extract_pdf_text_with_pdftotext("/usr/bin/pdftotext", b"%PDF-1.4\n") == ""
    assert process.killed


def test_pdf_text_extraction_times_out_while_stdout_is_open(monkeypatch):
    class FakeStdout:
        def fileno(self):
            return 42

        def close(self):
            pass

    class FakeProcess:
        stdout = FakeStdout()
        returncode = -9
        killed = False

        def kill(self):
            self.killed = True

        def wait(self, **kwargs):
            pass

    process = FakeProcess()
    timeouts = []
    monkeypatch.setattr(api_anthropic.subprocess, "Popen", lambda *args, **kwargs: process)

    def fake_select(*args):
        timeouts.append(args[-1])
        return [], [], []

    monkeypatch.setattr(api_anthropic.select, "select", fake_select)

    assert api_anthropic._extract_pdf_text_with_pdftotext("/usr/bin/pdftotext", b"%PDF-1.4\n") == ""
    assert process.killed
    assert 0 < timeouts[0] <= api_anthropic._PDF_SUBPROCESS_TIMEOUT_SECONDS


def test_pdf_text_extraction_does_not_start_after_request_deadline(monkeypatch):
    monkeypatch.setattr(
        api_anthropic.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("expired request must not start pdftotext"),
    )

    assert api_anthropic._extract_pdf_text_with_pdftotext("/usr/bin/pdftotext", b"%PDF-1.4\n", deadline=0) == ""


def test_pdf_text_extraction_is_cached(monkeypatch):
    calls = {"count": 0}

    api_anthropic._extract_pdf_text.cache_clear()

    def fake_extract(_pdftotext, _pdf_bytes, _deadline=None):
        calls["count"] += 1
        return "PDF_SENTINEL"

    monkeypatch.setattr(api_anthropic.shutil, "which", lambda _: "/usr/bin/pdftotext")
    monkeypatch.setattr(api_anthropic, "_extract_pdf_text_with_pdftotext", fake_extract)

    assert api_anthropic._extract_pdf_text(b"%PDF-1.4\n") == "PDF_SENTINEL"
    assert api_anthropic._extract_pdf_text(b"%PDF-1.4\n") == "PDF_SENTINEL"
    assert calls["count"] == 1


def test_pdf_cache_evicts_by_memory_budget(monkeypatch):
    calls = {"count": 0}

    api_anthropic._extract_pdf_text.cache_clear()
    monkeypatch.setattr(api_anthropic, "_PDF_CACHE_MAX_BYTES", 8)

    def fake_extract(_pdftotext, pdf_bytes, _deadline=None):
        calls["count"] += 1
        return pdf_bytes.decode("ascii")

    monkeypatch.setattr(api_anthropic.shutil, "which", lambda _: "/usr/bin/pdftotext")
    monkeypatch.setattr(api_anthropic, "_extract_pdf_text_with_pdftotext", fake_extract)

    assert api_anthropic._extract_pdf_text(b"aaaa") == "aaaa"
    assert api_anthropic._extract_pdf_text(b"bbbbbbbb") == "bbbbbbbb"
    assert api_anthropic._extract_pdf_text(b"aaaa") == "aaaa"
    assert calls["count"] == 3


def test_pdf_cache_does_not_store_empty_results():
    api_anthropic._clear_pdf_cache()

    api_anthropic._pdf_cache_set(("text", b"a", 1), "")
    api_anthropic._pdf_cache_set(("vision", b"b", 1), ())

    assert not api_anthropic._PDF_CACHE
    assert api_anthropic._PDF_CACHE_BYTES == 0


def test_anthropic_messages_impl_runs_translation_in_thread(monkeypatch):
    called = {}

    async def fake_to_thread(fn, *args, **kwargs):
        called["to_thread"] = True
        return fn(*args, **kwargs)

    def fake_translate(_body):
        return {"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}, {}

    async def fake_chat_completions_impl(_request, _raw_request, **_kwargs):
        from fastapi.responses import Response

        return Response("ok")

    class FakeRequest:
        async def json(self):
            return {"model": "test-model", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]}

    import lightllm.server.api_openai as api_openai

    monkeypatch.setattr(api_anthropic.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(api_anthropic, "_anthropic_to_chat_request", fake_translate)
    monkeypatch.setattr(api_openai, "chat_completions_impl", fake_chat_completions_impl)

    response = asyncio.run(api_anthropic.anthropic_messages_impl(FakeRequest()))

    assert called["to_thread"]
    assert response.body == b"ok"


def test_tool_result_image_blocks_survive_as_image_url_parts():
    body = _tool_result_body(
        [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "YWJjZA==",
                },
            }
        ]
    )

    chat_dict, _ = _anthropic_to_chat_request(body)

    assert chat_dict["messages"][2]["role"] == "tool"
    assert chat_dict["messages"][2]["content"] == [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,YWJjZA=="},
        }
    ]


def test_tool_result_url_image_block_survives_as_image_url_part():
    body = _tool_result_body(
        [
            {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": "https://example.com/image.png",
                },
            }
        ]
    )

    chat_dict, _ = _anthropic_to_chat_request(body)

    assert chat_dict["messages"][2]["role"] == "tool"
    assert chat_dict["messages"][2]["content"] == [
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/image.png"},
        }
    ]


def test_tool_result_textual_image_data_url_stays_text():
    data_url = "data:image/png;base64,YWJjZA=="
    body = _tool_result_body(data_url)

    chat_dict, _ = _anthropic_to_chat_request(body)

    assert chat_dict["messages"][2]["role"] == "tool"
    assert chat_dict["messages"][2]["content"] == data_url


def test_pdf_shaped_tool_schema_example_is_not_rewritten(monkeypatch):
    body = _base_body()
    body["tools"] = [
        {
            "name": "inspect",
            "description": "inspect",
            "input_schema": {
                "type": "object",
                "examples": [_pdf_document_block()],
            },
        }
    ]
    original = json.loads(json.dumps(body))
    monkeypatch.setattr(
        api_anthropic,
        "_ensure_pdf_parsing_supported",
        lambda: pytest.fail("tool schemas are not message content"),
    )

    api_anthropic._replace_anthropic_pdf_documents(body)

    assert body == original


# Helpers for streaming test
def _chunk(delta, finish_reason=None, usage=None):
    obj = {"choices": [{"delta": delta, "finish_reason": finish_reason}]}
    if usage is not None:
        obj["usage"] = usage
    return f"data: {json.dumps(obj)}\n\n"


def test_interleaved_tool_calls_do_not_emit_against_closed_block():
    """Deltas for tool-call idx=1 arriving after idx=0 started must not
    stream into the (now-closed) idx=0 block."""

    async def chunks():
        yield _chunk(
            {
                "tool_calls": [
                    {"index": 0, "id": "call_a", "function": {"name": "fn_a", "arguments": '{"x":1'}},
                ]
            }
        )
        yield _chunk(
            {
                "tool_calls": [
                    {"index": 1, "id": "call_b", "function": {"name": "fn_b", "arguments": '{"y":2}'}},
                ]
            }
        )
        yield _chunk(
            {
                "tool_calls": [
                    {"index": 0, "function": {"arguments": "}"}},
                ]
            }
        )
        yield _chunk({}, finish_reason="tool_calls", usage={"prompt_tokens": 3, "completion_tokens": 4})

    async def run():
        out = []
        async for ev in _openai_sse_to_anthropic_events(chunks(), "m", "msg_x"):
            out.append(ev.decode("utf-8"))
        return out

    events = asyncio.run(run())
    inputs = {}
    currently_open = None
    for raw in events:
        lines = raw.strip().split("\n")
        etype = lines[0].split(": ", 1)[1]
        data = json.loads(lines[1].split(": ", 1)[1])
        if etype == "content_block_start":
            currently_open = data["index"]
            assert currently_open not in inputs
            inputs[currently_open] = ""
        elif etype == "content_block_stop":
            currently_open = None
        elif etype == "content_block_delta":
            assert (
                currently_open == data["index"]
            ), f"delta for index {data['index']} but open block is {currently_open}"
            inputs[currently_open] += data["delta"]["partial_json"]
        assert etype != "error"
    assert [json.loads(args) for args in inputs.values()] == [{"x": 1}, {"y": 2}]


@pytest.fixture
def messages_endpoint(monkeypatch):
    from lightllm.server import api_http, api_openai, api_stream_obj, build_prompt
    from lightllm.server.core.objs import FinishStatus
    from lightllm.server.core.objs.start_args_type import StartArgs
    from lightllm.server.httpserver.stop_seq_match import wrap_stop_match_stream

    args = StartArgs(run_mode="normal", reasoning_parser=None, tool_call_parser="deepseekv41")
    tokens = []
    manager = SimpleNamespace(
        args=args,
        tokenizer=SimpleNamespace(encode=lambda *args, **kwargs: []),
        output_grammar_compiler=None,
        stop_request=AsyncMock(),
        tokens=lambda prompt, *_args: len(prompt),
    )

    def generate(prompt, sampling_params, *_args, **_kwargs):
        async def raw_stream():
            for index, (text, status) in enumerate(tokens):
                if isinstance(status, Exception):
                    raise status
                yield 0, text, {"id": index, "prompt_tokens": 100, "prompt_cache_len": 80}, FinishStatus(status)

        return wrap_stop_match_stream(manager, raw_stream(), sampling_params, prompt)

    monkeypatch.setattr(api_http, "get_env_start_args", lambda: args)
    monkeypatch.setattr(api_openai, "get_env_start_args", lambda: args)
    monkeypatch.setattr(api_stream_obj, "get_env_start_args", lambda: args)
    manager.generate = generate
    metrics = SimpleNamespace(counter_inc=lambda *args: None)
    monkeypatch.setattr(
        api_http, "g_objs", SimpleNamespace(args=args, httpserver_manager=manager, metric_client=metrics)
    )
    monkeypatch.setattr(api_openai, "build_prompt", AsyncMock(return_value="prompt"))
    monkeypatch.setattr(build_prompt, "build_prompt", api_openai.build_prompt)

    def post(body=None, *, path="/v1/messages", **kwargs):
        async def run():
            transport = httpx.ASGITransport(app=api_http.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                if body is not None:
                    kwargs["json"] = body
                return await client.post(path, **kwargs)

        return asyncio.run(run())

    return post, tokens, FinishStatus


def _events(response):
    return [json.loads(line[6:]) for line in response.text.split("\n") if line.startswith("data: ")]


@pytest.mark.parametrize("stream", [False, True])
def test_messages_matches_stop_sequence_and_reports_cache_usage(messages_endpoint, stream):
    post, tokens, status = messages_endpoint
    tokens[:] = [("hello#", status.NO_FINISH), ("END ignored", status.NO_FINISH), ("late", status.FINISHED_LENGTH)]
    response = post({**_base_body(), "stop_sequences": ["#END"], "stream": stream})
    assert response.status_code == 200
    if stream:
        events = _events(response)
        start, end = events[0]["message"], events[-2]
        assert start["usage"]["input_tokens"] == 20
        assert start["usage"]["cache_read_input_tokens"] == 80
        assert start["usage"]["output_tokens"] == 0
        text = "".join(event["delta"]["text"] for event in events if event["type"] == "content_block_delta")
        stop, usage = end["delta"], end["usage"]
    else:
        message = response.json()
        stop, usage = message, message["usage"]
        text = message["content"][0]["text"]
    assert text == "hello"
    assert (stop["stop_reason"], stop["stop_sequence"]) == ("stop_sequence", "#END")
    assert (usage["input_tokens"], usage["cache_read_input_tokens"], usage["output_tokens"]) == (20, 80, 2)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "failure", ["first_error", "late_error", "abort", "early_eof", "pd_disconnect", "first_pd_error"]
)
def test_messages_surfaces_failed_generation(messages_endpoint, stream, failure):
    post, tokens, status = messages_endpoint
    first = failure in ("first_error", "first_pd_error")
    if not first:
        tokens.append(("partial text", status.NO_FINISH))
    if failure in ("pd_disconnect", "first_pd_error"):
        tokens.append(("", RuntimeError("PD node disconnected")))
    elif failure != "early_eof":
        tokens.append(("", status.FINISHED_ABORTED if failure == "abort" else status.FINISHED_ERROR))
    response = post({**_base_body(), "stream": stream})
    if stream and not first:
        events = _events(response)
        assert response.status_code == 200
        assert [event["type"] for event in events[-3:]] == ["content_block_stop", "error", "message_stop"]
        assert events[-2]["error"]["type"] == "api_error"
        assert all(event["type"] != "message_delta" for event in events)
    else:
        assert response.status_code == 500
        assert response.json()["type"] == "error"
        assert response.json()["error"]["type"] == "api_error"


@pytest.mark.parametrize("split", [False, True])
def test_messages_restores_long_tool_name_and_preserves_order_in_anthropic_sdk(messages_endpoint, split):
    anthropic = pytest.importorskip("anthropic")
    post, tokens, status = messages_endpoint
    name = "mcp__" + "server_" * 10 + "write"
    body = {**_base_body(), "tools": [{"name": name, "input_schema": {"type": "object"}}], "stream": True}
    chat, _ = _anthropic_to_chat_request(body)
    short_name = chat["tools"][0]["function"]["name"]
    assert short_name != name
    start = (
        f'Before. <｜DSML｜ calls><｜DSML｜ invoke name="{short_name}">'
        '<｜DSML｜ parameter name="content" string="true">abc'
    )
    tail = "</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls> Done."
    if split:
        tokens.append((start, status.NO_FINISH))
    tokens.append((tail if split else start + tail, status.FINISHED_STOP))
    response = post(body)
    transport = httpx.MockTransport(lambda _: httpx.Response(200, headers=response.headers, content=response.content))
    with anthropic.Anthropic(api_key="test", http_client=httpx.Client(transport=transport)) as client:
        with client.messages.stream(model=body["model"], max_tokens=body["max_tokens"], messages=body["messages"]) as s:
            message = s.get_final_message()
    assert [block.type for block in message.content] == ["text", "tool_use", "text"]
    assert message.content[0].text == "Before. "
    assert message.content[1].name == name
    assert message.content[1].input == {"content": "abc"}
    assert message.content[2].text == " Done."


@pytest.mark.parametrize("stream", [False, True])
def test_messages_enforces_disabled_parallel_tool_use(messages_endpoint, stream):
    post, tokens, status = messages_endpoint
    body = {
        **_base_body(),
        "tools": [{"name": "Read", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
        "stream": stream,
    }
    chat, _ = _anthropic_to_chat_request(body)
    assert chat["parallel_tool_calls"] is False
    invoke = '<｜DSML｜ invoke name="Read"></｜DSML｜ invoke>'
    tokens[:] = [(f"<｜DSML｜ calls>{invoke}{invoke}</｜DSML｜ calls>", status.FINISHED_STOP)]
    response = post(body)
    assert response.status_code == 200
    if stream:
        events = _events(response)
        content = [event["content_block"] for event in events if event["type"] == "content_block_start"]
        stop = events[-2]["delta"]
    else:
        stop = response.json()
        content = stop["content"]
    assert [(block["type"], block["name"]) for block in content] == [("tool_use", "Read")]
    assert stop["stop_reason"] == "tool_use"


def test_messages_ignores_variable_billing_header():
    stable = {"type": "text", "text": "Keep this instruction.", "cache_control": {"type": "ephemeral"}}
    expected, _ = _anthropic_to_chat_request({**_base_body(), "system": [stable]})
    for nonce in ("aaa", "bbb"):
        header = f"x-anthropic-billing-header: cc_version=2.1.{nonce}; cch={nonce};"
        system = [{"type": "text", "text": header}, stable]
        request, _ = _anthropic_to_chat_request({**_base_body(), "system": system})
        assert request["messages"] == expected["messages"]
        request, _ = _anthropic_to_chat_request({**_base_body(), "system": header + "\n" + stable["text"]})
        assert request["messages"][0]["content"] == stable["text"]


@pytest.mark.parametrize("payload", [{"content": "{"}, {"json": []}, {"json": {"model": "test"}}])
def test_messages_count_tokens_returns_invalid_request(messages_endpoint, payload):
    post, _, _ = messages_endpoint
    response = post(path="/v1/messages/count_tokens", **payload)
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_messages_count_tokens_accepts_request_without_max_tokens(messages_endpoint):
    post, _, _ = messages_endpoint
    response = post({"model": "test", "messages": _base_body()["messages"]}, path="/v1/messages/count_tokens")
    assert response.status_code == 200
    assert response.json() == {"input_tokens": 6, "context_management": {"original_input_tokens": 6}}


def test_closing_anthropic_response_closes_underlying_stream(monkeypatch):
    from lightllm.server import api_openai
    from lightllm.server.api_stream_obj import CustomStreamingResponse

    closed = []

    async def chunks():
        try:
            yield _chunk({"content": "partial"})
        finally:
            closed.append(True)

    inner = CustomStreamingResponse(chunks())
    monkeypatch.setattr(api_openai, "chat_completions_impl", AsyncMock(return_value=inner))

    async def run():
        request = SimpleNamespace(json=AsyncMock(return_value={**_base_body(), "stream": True}))
        response = await api_anthropic.anthropic_messages_impl(request)
        async with aclosing(response.body_iterator) as iterator:
            await anext(iterator)
        assert closed == [True]

    asyncio.run(run())


def test_streaming_reasoning_uses_anthropic_thinking_events():
    async def chunks():
        yield _chunk({"reasoning_content": "First "})
        yield _chunk({"reasoning_content": "think."})
        yield _chunk({"content": "42"}, finish_reason="stop")
        yield _chunk({}, usage={"prompt_tokens": 3, "completion_tokens": 4})

    async def run():
        return [event.decode("utf-8") async for event in _openai_sse_to_anthropic_events(chunks(), "m", "msg_x")]

    events = []
    for raw in asyncio.run(run()):
        lines = raw.strip().split("\n")
        events.append((lines[0].split(": ", 1)[1], json.loads(lines[1].split(": ", 1)[1])))

    assert [event_type for event_type, _ in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[1][1]["content_block"] == {"type": "thinking", "thinking": "", "signature": ""}
    assert events[2][1]["delta"] == {"type": "thinking_delta", "thinking": "First "}
    assert events[3][1]["delta"] == {"type": "thinking_delta", "thinking": "think."}
    assert events[4][1]["delta"] == {
        "type": "signature_delta",
        "signature": api_anthropic._SYNTHETIC_THINKING_SIGNATURE,
    }
    assert events[7][1]["delta"] == {"type": "text_delta", "text": "42"}


def test_chat_response_translation_failure_returns_valid_json():
    """If response translation raises, the error path must return a clean
    Anthropic-shaped JSONResponse — not a JSONResponse wrapped in another
    JSONResponse."""
    from fastapi.responses import JSONResponse

    from lightllm.server import api_anthropic

    # Exercise the helper directly; the bug in anthropic_messages_impl was
    # wrapping this return value in another JSONResponse.
    resp = api_anthropic._anthropic_error_response(api_anthropic.HTTPStatus.INTERNAL_SERVER_ERROR, "synthetic")
    assert isinstance(resp, JSONResponse)
    body = bytes(resp.body).decode("utf-8")
    assert '"type":"error"' in body
    assert '"message":"synthetic"' in body
    assert resp.status_code == 500


def test_unknown_fields_emit_debug_log(caplog):
    """Silently-dropped Anthropic fields should at least emit a debug log so
    users can trace 'my metadata isn't propagating' without adding prints."""
    import logging

    from lightllm.server.api_anthropic import _anthropic_to_chat_request

    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8,
        "metadata": {"user_id": "abc"},
        "anthropic_version": "2023-06-01",
    }
    # Set logger to DEBUG so caplog can capture it
    logger = logging.getLogger("lightllm.server.api_anthropic")
    logger.setLevel(logging.DEBUG)

    # Manually add caplog's handler to the logger to intercept logs
    # (works even with propagate=False)
    caplog_handler = logging.Handler()
    caplog_handler.emit = lambda record: caplog.records.append(record)
    logger.addHandler(caplog_handler)

    try:
        try:
            _anthropic_to_chat_request(body)
        except RuntimeError:
            import pytest

            pytest.skip("litellm not available; cannot exercise drop path")
        joined = "\n".join(rec.getMessage() for rec in caplog.records)
        assert "metadata" in joined or "anthropic_version" in joined
    finally:
        logger.removeHandler(caplog_handler)
