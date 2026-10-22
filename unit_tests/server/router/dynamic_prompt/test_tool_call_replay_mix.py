import copy
import importlib.util
from pathlib import Path

import pytest


path = Path(__file__).resolve().parents[4] / "test/benchmark/service/tool_call_cache/replay.py"
spec = importlib.util.spec_from_file_location("tool_call_replay_mix", path)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


@pytest.mark.parametrize("sessions", [4, 8, 16, 24])
@pytest.mark.parametrize("mix,divisor", [("agent25", 4), ("agent50", 2)])
def test_agent_mixtures_have_exact_stable_session_counts(sessions, mix, divisor):
    selected = [(f"s{i}", [{"turn": 0}], 0) for i in range(sessions)]
    kinds = replay.stream_kinds(selected, mix)
    assert len(kinds) == sessions
    assert list(kinds.values()).count("agent") == sessions // divisor
    assert list(kinds.values()).count("one-shot") == sessions - sessions // divisor
    assert kinds == replay.stream_kinds(copy.deepcopy(selected), mix)


@pytest.mark.parametrize("mix", ["trace", "ordinary", "all-tools", "mixed25"])
def test_existing_modes_keep_their_original_stream_semantics(mix):
    selected = [("a", [], 0), ("b", [], 0)]
    assert replay.stream_kinds(selected, mix) == {"a": "trace", "b": "trace"}


def test_independent_requests_do_not_reuse_a_long_previous_conversation():
    system = {"role": "system", "content": "Session s. Follow instructions."}
    original = copy.deepcopy(system)
    messages = [system, {"role": "user", "content": "OLD_PRIVATE_REFERENCE" * 1000}]
    replay.reset_one_shot(messages, system, 0)
    first = copy.deepcopy(messages)
    messages.extend([{"role": "assistant", "content": "OK"}, {"role": "user", "content": "MORE_OLD_DATA"}])
    replay.reset_one_shot(messages, system, 1)
    assert len(messages) == 1
    assert messages != first
    assert "OLD" not in str(messages)
    assert system == original


def test_cohort_latency_statistics_keep_empty_groups_explicit():
    assert replay.latency_stats([]) == {"mean": None, "p50": None, "p95": None}
    result = replay.latency_stats([1, 2, 3, 4])
    assert result["mean"] == 2.5
    assert result["p95"] >= result["p50"]


def test_experimental_labels_are_opaque_to_the_model_and_scope_is_stable():
    namespace = "PRIVATE_POLICY_NEW_VS_OFF"
    name = "PRIVATE_DATASET_SESSION"
    first = replay.session_system(namespace, name)
    assert namespace not in first["content"] and name not in first["content"]
    assert first == replay.session_system(namespace, name)
    assert first != replay.session_system(namespace, name + "_next")
    assert first != replay.session_system(namespace + "_next", name)
    assert "latest user instruction" in first["content"]
    assert "actual tool call" in first["content"]


def test_compaction_changes_identity_before_long_reference_and_clears_old_history():
    system = {"role": "system", "content": "Session s. Follow latest instructions."}
    original = copy.deepcopy(system)
    messages = [system, {"role": "user", "content": "OLD_HISTORY"}]
    replay.reset_compacted_context(messages, system, 1)
    first = copy.deepcopy(messages)
    messages.append({"role": "user", "content": "SYNTHETIC_LONG_REFERENCE" * 1000})
    replay.reset_compacted_context(messages, system, 2)
    assert len(messages) == 1 and messages != first
    assert messages[0]["role"] == "system"
    assert "Context revision 2" in messages[0]["content"]
    assert "OLD_HISTORY" not in str(messages) and "LONG_REFERENCE" not in str(messages)
    assert system == original


def test_legacy_shared_compaction_identity_is_explicit_and_reproducible():
    system = {"role": "system", "content": "Session s."}
    messages = []
    replay.reset_compacted_context(messages, system, 1, identity="shared")
    first = copy.deepcopy(messages)
    replay.reset_compacted_context(messages, system, 2, identity="shared")
    assert first == messages == [system]
    with pytest.raises(ValueError, match="compaction identity"):
        replay.reset_compacted_context(messages, system, 3, identity="unknown")
