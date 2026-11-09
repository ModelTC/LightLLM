import json

import pytest

from lightllm.server.api_models import Tool
from lightllm.server.function_call_parser import FunctionCallParser


TOOLS = [
    Tool.model_validate(
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                },
            },
        }
    ),
    Tool.model_validate(
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}, "days": {"type": "integer"}},
                },
            },
        }
    ),
]

DSML = "｜DSML｜"


def dsml_text(wrapper):
    return (
        f"<{DSML}{wrapper}>\n"
        f'<{DSML}invoke name="write_file">\n'
        f'<{DSML}parameter name="path" string="true">src/app.py</{DSML}parameter>\n'
        f'<{DSML}parameter name="content" string="true">print("hi")\n</{DSML}parameter>\n'
        f"</{DSML}invoke>\n"
        f'<{DSML}invoke name="get_weather">\n'
        f'<{DSML}parameter name="location" string="true">Hangzhou</{DSML}parameter>\n'
        f'<{DSML}parameter name="days" string="false">3</{DSML}parameter>\n'
        f"</{DSML}invoke>\n"
        f"</{DSML}{wrapper}>"
    )


def collect_stream(parser_name, text, chunk_size):
    parser = FunctionCallParser(TOOLS, parser_name)
    calls = {}
    for start in range(0, len(text), chunk_size):
        _, deltas = parser.parse_stream_chunk(text[start : start + chunk_size])
        for delta in deltas:
            if delta.name is not None:
                calls[delta.tool_index] = {"name": delta.name, "arguments": ""}
            calls[delta.tool_index]["arguments"] += delta.parameters
    return calls


@pytest.mark.parametrize("parser_name,wrapper", [("deepseekv32", "function_calls"), ("deepseekv4", "tool_calls")])
@pytest.mark.parametrize("chunk_size", [1, 5, 16])
def test_streamed_arguments_concatenate_to_valid_json(parser_name, wrapper, chunk_size):
    calls = collect_stream(parser_name, dsml_text(wrapper), chunk_size)
    assert [call["name"] for call in calls.values()] == ["write_file", "get_weather"]
    assert json.loads(calls[0]["arguments"]) == {"path": "src/app.py", "content": 'print("hi")\n'}
    assert json.loads(calls[1]["arguments"]) == {"location": "Hangzhou", "days": 3}
