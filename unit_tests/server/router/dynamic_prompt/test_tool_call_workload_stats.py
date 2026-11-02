import copy
import importlib.util
import json
from pathlib import Path

import pytest


path = Path(__file__).resolve().parents[4] / "test/benchmark/service/tool_call_cache/workload_stats.py"
spec = importlib.util.spec_from_file_location("tool_call_workload_stats", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def phase(tmp_path):
    def save(name, data):
        (tmp_path / name).write_text(json.dumps(data))

    save("progress.json", {"status": "complete"})
    save("protocol.json", {"cases": [{"name": "trace"}], "repeats": 1})
    save("summary.json", [{"case": "trace", "repeat": 0}])
    payload = {
        "configuration": {"url": "DO_NOT_PUBLISH_PRIVATE_URL", "context_scale": 0.5, "max_prompt_tokens": 40},
        "output_target_capped_requests": 0,
        "rows": [
            {
                "finish_reason": "length",
                "client_prompt_tokens": 12,
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
                "compaction": True,
                "wait_capped": True,
                "source_input_tokens": 100,
                "source_output_tokens": 50,
                "source_wait_seconds": 1000,
                "wait_seconds": 5,
                "source_request": "DO_NOT_PUBLISH_SOURCE_REQUEST",
                "output_signature": {"content": "DO_NOT_PUBLISH_CONTENT"},
            }
        ],
    }
    for policy in ["candidate", "control"]:
        save(f"trace-r0-{policy}.json", payload)
    return payload


def test_public_workload_export_keeps_caps_and_lengths_without_private_fields(tmp_path):
    phase(tmp_path)
    result = module.summarize([tmp_path])
    assert "DO_NOT_PUBLISH" not in json.dumps(result)
    row = result["trials"][0]
    assert row["compacted_requests"] == row["wait_capped_requests"] == 1
    assert row["prompt_target_capped_requests"] == 1
    assert row["completion_tokens"]["p95"] == 3
    assert row["finish_reason_counts"] == {"length": 1}


def test_workload_export_rejects_unequal_source_characteristics(tmp_path):
    payload = copy.deepcopy(phase(tmp_path))
    payload["rows"][0]["source_input_tokens"] = 200
    (tmp_path / "trace-r0-control.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="paired workload characteristics differ"):
        module.summarize([tmp_path])


def test_workload_export_rejects_quarantined_data(tmp_path):
    phase(tmp_path)
    (tmp_path / "invalidity.json").write_text("{}")
    with pytest.raises(ValueError, match="invalidated phase"):
        module.summarize([tmp_path])
