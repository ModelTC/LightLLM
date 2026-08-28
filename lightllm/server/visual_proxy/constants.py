"""Shared constants and compiled policies for the visual proxy package."""

from __future__ import annotations

import re
from typing import Any, Final


VISION_READER_NAME: Final = "vision_reader"
MAX_AGENT_STEPS: Final = 12
MAX_REASONING_CONTEXT_BYTES: Final = 256 * 1024
ANTHROPIC_SEQUENTIAL_TOOL_PROMPT: Final = (
    "Call builtin vision_reader by itself and wait for its structured result before issuing any external "
    "tool call. Never mix vision_reader with another tool in the same assistant turn."
)

# Server-owned tool contract.
BUILTIN_VISION_READER_TOOL: Final[dict[str, Any]] = {
    "type": "function",
    "function": {
        "name": VISION_READER_NAME,
        "description": (
            "BUILTIN tag-based vision reader for inspecting an image already materialized as an exact "
        "<image_n/> tag. Before answering anything that depends on visual content—including an image, "
        "screenshot, chart, table, UI, OCR, object count, color, position, layout, or PDF appearance—"
        "first identify the image targeted by the latest user request. "
        "If that target is represented by an exact <image_n/> tag in the latest request or in a matching "
        "tool response generated for it, MUST call vision_reader with that exact tag before answering. "
        "vision_reader cannot load a workspace filename, path, URL, file URI, or base64 value. "
        "If the latest target is only such an unloaded resource and has no matching current-request tag, "
        "DO NOT call vision_reader on any historical tag and DO NOT answer from historical visual results; "
        "first call an available external tool, such as vision_analyze, to load that exact resource. "
        "After its matching tool response exposes a new <image_n/> tag, MUST call vision_reader with that "
        "exact new tag before answering. "
        "Use a historical tag only when the latest request explicitly targets that historical image. "
        "Never invent or renumber image tags."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": "An exact conversation image tag such as <image_1/>.",
                },
                "task": {
                    "type": "string",
                    "description": "The visual inspection task or question.",
                },
            },
            "required": ["image", "task"],
        },
    },
}

# Private trace parsing and public-output safety.
IMAGE_TAG_PATTERN = re.compile(r"<\s*image[_-](\d+)\s*/?\s*>", re.IGNORECASE)
IMAGE_ALIAS_PATTERN = re.compile(r"(?:image[_\s-]*|picture\s*)(\d+)", re.IGNORECASE)
TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
FUNCTION_PATTERN = re.compile(
    r"<function=([A-Za-z0-9_.:-]+)>\s*(.*?)\s*</function>", re.DOTALL
)
PARAMETER_PATTERN = re.compile(
    r"<parameter=([A-Za-z0-9_.:-]+)>\s*(.*?)\s*</parameter>", re.DOTALL
)
RAW_XML_TOOL_BLOCK_PATTERN = re.compile(
    r"<tool_call\b[^>]*>.*?</tool_call\s*>|<tool_response\b[^>]*>.*?</tool_response\s*>",
    re.IGNORECASE | re.DOTALL,
)
RAW_DSML_TOOL_BLOCK_PATTERN = re.compile(
    r"<｜DSML｜tool_calls\b[^>]*>.*?</｜DSML｜tool_calls\s*>",
    re.IGNORECASE | re.DOTALL,
)
RAW_FUNCTION_CALL_LINE_PATTERN = re.compile(
    r"(?:</?(?:tool_call|tool_response)\b|<function=|</function>|<parameter=|</parameter>|"
    r"<｜DSML｜|\"?tool_calls\"?\s*:|[\"']type[\"']\s*:\s*[\"'](?:tool_use|tool_result)[\"']|"
    r"[\"']tool_use_id[\"']\s*:|[\"']function[\"']\s*:|^\s*function\s*:|"
    r"(?:recipient|to)\s*=\s*functions\.)",
    re.IGNORECASE,
)
PROVIDER_CONTROL_TAG_PATTERN = re.compile(
    r"</?pcwpd_[A-Za-z0-9_.:-]+\s*>", re.IGNORECASE
)
NATURAL_VISION_TRACE_PATTERN = re.compile(
    r"^我(?:先|接着|随后)查看了图片\s+(<image_\d+/?>)，让内建读图能力完成这个任务：(.+?)(?:。)?$"
)
NATURAL_OBSERVATION_PATTERN = re.compile(
    r"^我(?:先|接着|随后)仔细看了图片\s+(<image_\d+/?>)，想确认\s*"
)
NATURAL_OBSERVATION_RESULT_MARKER: Final = "；从画面中得到的信息是 "
XML_BUILTIN_TRACE_PAIR_PATTERN = re.compile(
    r"(<tool_call>\s*<function=vision_reader>\s*.*?</function>\s*</tool_call>)\s*"
    r"<tool_response>\s*(.*?)\s*</tool_response>",
    re.DOTALL,
)
PRIVATE_MULTIMODAL_MARKERS: Final = (
    "vision_reader",
    "让内建读图能力完成这个任务",
    "读图结果：",
    "builtin vision",
    "built-in vision",
    "builtin tool",
    "built-in tool",
    "multimodal harness",
    "multimodal_harness",
    "内建读图",
    "内置读图",
    "内建工具",
    "内置工具",
    "读图工具",
)
PRIVATE_MULTIMODAL_MECHANISM_PATTERN = re.compile(
    r"(?:\bvision[\s_-]*reader\b|"
    r"\b(?:internal|private|built[\s-]?in)\s+(?:(?:image|vision)\s+)?(?:reader|tool)\b|"
    r"(?:内部|内建|内置)(?:(?:读图|视觉|图片)(?:工具|读取器|阅读器|模型)?|工具)|"
    r"图片(?:读取器|阅读器|工具))",
    re.IGNORECASE,
)
PRIVATE_IMAGE_LABEL_PATTERN = re.compile(
    r"(?:</?image_\d+\s*/?>|\bimage_\d+\b)", re.IGNORECASE
)
PRIVATE_EXECUTION_STATE_PATTERN = re.compile(
    r"(?:\bthere\s+(?:are|is)\s+no\s+tools?\s+with\s+results?\b|"
    r"\bno\s+tool\s+results?\s+(?:are\s+)?available\b|"
    r"\btool\s+results?\s+(?:(?:is|are)\s+)?(?:unavailable|missing|not\s+available)\b|"
    r"(?:没有|无)(?:任何|可用的?)?(?:工具结果|工具返回)(?:可用)?)",
    re.IGNORECASE,
)

# Evidence acceptance.
INVALID_BUILTIN_VISION_RESULT_MARKERS: Final = (
    "Builtin vision_reader cannot read",
    "Builtin vision_reader requires both arguments",
    "Builtin vision_reader rejected the call",
)
TRUNCATED_FINISH_REASONS: Final = frozenset(
    {"length", "max_tokens", "max_output_tokens"}
)

BUILTIN_TRACE_FORMATS: Final = frozenset({"xml", "natural"})
THINKING_POLICIES: Final = frozenset({"request", "force_on", "force_off"})

# Vision provider request limits.
VISION_MAX_OUTPUT_TOKENS: Final = 8192
VISION_TASK_MAX_CHARS: Final = 16384
