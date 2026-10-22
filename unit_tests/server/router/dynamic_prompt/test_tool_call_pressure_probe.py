import copy
import importlib.util
from pathlib import Path


path = Path(__file__).resolve().parents[4] / "test/benchmark/service/tool_call_cache/pressure_probe.py"
spec = importlib.util.spec_from_file_location("tool_call_pressure_probe", path)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_continuation_keeps_tools_and_actual_results_with_stable_opaque_ids():
    body = {"messages": [{"role": "user", "content": "reference and instruction"}], "tools": [{"type": "function"}]}
    original = copy.deepcopy(body)
    response = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "server_generated_opaque_id",
                            "type": "function",
                            "function": {"name": "echo", "arguments": '{"text":"actual generated value"}'},
                        }
                    ],
                }
            }
        ]
    }
    result = probe.continuation_body(body, response)
    assert body == original
    assert response["choices"][0]["message"]["tool_calls"][0]["id"] == "server_generated_opaque_id"
    assert result["tools"] == body["tools"]
    assert result["messages"][:1] == body["messages"]
    assert result["messages"][1]["tool_calls"][0]["id"] == "pressure_call_0"
    assert result["messages"][2] == {
        "role": "tool",
        "tool_call_id": "pressure_call_0",
        "content": "actual generated value",
    }
