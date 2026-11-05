import pytest
import pickle
from unittest.mock import Mock
from lightllm.server.core.objs.sampling_params import (
    StopStrMatchMode,
    StopSequence,
    StopSequenceGroups,
    RegularConstraint,
    ExponentialDecayLengthPenalty,
    NodeUUId,
    SamplingParams,
    GuidedGrammar,
    GuidedJsonSchema,
    STOP_SEQUENCE_MAX_LENGTH,
    REGULAR_CONSTRAINT_MAX_LENGTH,
    JSON_SCHEMA_MAX_LENGTH,
    GRAMMAR_CONSTRAINT_MAX_LENGTH,
    MAX_BEST_OF,
)
from lightllm.server.core.objs.py_sampling_params import SamplingParams as PySamplingParams

grammar_str = r"""root ::= (expr "=" term)+
expr ::= term ([-+*/] term)*
term ::= num | "(" expr ")"
num ::= [0-9]+"""

schema_str = r"""{
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "Title": {"type": "string"},
            "Date": {"type": "string"},
            "Time": {"type": "string"}
        },
        "required": ["Title", "Time", "Date"]
    }
}"""


@pytest.mark.parametrize(
    "sequence, expected",
    [
        ([1, 2, 3], [1, 2, 3]),
        ([1] * (STOP_SEQUENCE_MAX_LENGTH), [1] * STOP_SEQUENCE_MAX_LENGTH),
    ],
)
def test_stop_sequence_initialization(sequence, expected):
    seq = StopSequence()
    seq.initialize(sequence)
    assert seq.size == len(expected)
    assert seq.to_list() == expected


def test_stop_sequence_initialization_too_many():
    seq = StopSequence()
    with pytest.raises(AssertionError):
        seq.initialize([1] * (STOP_SEQUENCE_MAX_LENGTH + 1))


@pytest.mark.parametrize(
    "stop_sequences, expected_ids, expected_strings, expected_original",
    [
        (["stop1", "stop2"], [], ["stop1", "stop2"], ["stop1", "stop2"]),
        ("END", [], ["END"], ["END"]),
        ([[1, 2], [3, 4]], [[1, 2], [3, 4]], [], [[1, 2], [3, 4]]),
        (["END", [1, 2], "结束"], [[1, 2]], ["END", "结束"], ["END", [1, 2], "结束"]),
        (["", [], "END", [1]], [[1]], ["END"], ["END", [1]]),
        ([], [], [], []),
        (None, [], [], []),
    ],
)
def test_stop_sequence_groups_initialization(stop_sequences, expected_ids, expected_strings, expected_original):
    groups = StopSequenceGroups()
    groups.initialize(stop_sequences)
    assert groups.size == len(expected_original)
    assert groups.to_token_ids_list() == expected_ids
    assert groups.to_stop_str_list() == expected_strings
    assert groups.has_stop_strs() is bool(expected_strings)
    assert groups.to_origin_list() == expected_original


@pytest.mark.parametrize("mode", [StopStrMatchMode.ALL_TEXT, StopStrMatchMode.CONTENT_ONLY])
def test_string_stops_are_not_encoded_and_round_trip_with_token_ids(mode):
    tokenizer = Mock()
    tokenizer.encode.side_effect = AssertionError("String stops must not be tokenized")
    stops = ["END", [10, 11], "结束"]
    params = SamplingParams()
    params.init(tokenizer, stop_sequences=stops, stop_str_match_mode=mode)
    serialized = params.to_dict()
    assert serialized["stop_sequences"] == stops
    restored = SamplingParams()
    restored.init(tokenizer, **serialized)
    for candidate in (params, restored, SamplingParams.from_buffer_copy(params), pickle.loads(pickle.dumps(params))):
        assert candidate.stop_sequences.to_token_ids_list() == [[10, 11]]
        assert candidate.stop_sequences.to_stop_str_list() == ["END", "结束"]
        assert candidate.stop_str_match_mode.get_mode() == mode
        assert candidate.to_origin_dict()["stop_sequences"] == stops
    py_params = PySamplingParams(stop_sequences=serialized["stop_sequences"], stop_str_match_mode=mode)
    py_params.verify()
    assert py_params.to_dict()["stop_sequences"] == stops
    tokenizer.encode.assert_not_called()


def test_regular_constraint_initialization():
    constraint = RegularConstraint()
    constraint.initialize("[a-zA-Z]+")
    assert constraint.length == len("[a-zA-Z]+")
    assert constraint.to_str() == "[a-zA-Z]+"

    with pytest.raises(AssertionError):
        constraint.initialize("a" * (REGULAR_CONSTRAINT_MAX_LENGTH + 1))


def test_guided_grammar_initialization():
    grammar = GuidedGrammar()
    grammar.initialize(grammar_str, None)
    assert grammar.to_str() == grammar_str

    with pytest.raises(AssertionError):
        grammar.initialize("a" * (GRAMMAR_CONSTRAINT_MAX_LENGTH + 1), None)


def test_guided_json_schema_initialization():
    schema = GuidedJsonSchema()
    schema.initialize(schema_str, None)
    assert schema.to_str() == schema_str

    with pytest.raises(AssertionError):
        schema.initialize("a" * (JSON_SCHEMA_MAX_LENGTH + 1), None)


def test_exponential_decay_length_penalty_initialization():
    penalty = ExponentialDecayLengthPenalty()
    penalty.initialize((5, 1.5))
    assert penalty.to_tuple() == (5, 1.5)

    with pytest.raises(AssertionError):
        penalty.initialize((5, 0.5))


def test_node_uuid_initialization():
    node_id = 12345678901234567890
    node_uuid = NodeUUId()
    node_uuid.initialize(node_id)

    assert node_uuid.node_id_high == (node_id >> 64) & 0xFFFFFFFFFFFFFFFF
    assert node_uuid.node_id_low == node_id & 0xFFFFFFFFFFFFFFFF
    assert node_uuid.get() == node_id


def test_sampling_params_initialization():
    params = SamplingParams()
    pd_master_node_id = 12345678901234567890
    pd_kv_trans_params = b"pd-kv-transport-params"
    data = {
        "best_of": 2,
        "n": 2,
        "do_sample": True,
        "presence_penalty": 0.5,
        "frequency_penalty": 0.5,
        "repetition_penalty": 1.0,
        "temperature": 1.0,
        "top_p": 0.9,
        "top_k": 50,
        "ignore_eos": False,
        "max_new_tokens": 16,
        "min_new_tokens": 1,
        "input_penalty": True,
        "group_request_id": 1,
        "suggested_dp_index": -1,
        "skip_special_tokens": True,
        "add_special_tokens": True,
        "add_spaces_between_special_tokens": True,
        "print_eos_token": False,
        "regular_constraint": "",
        "stop_sequences": [[2, 1], [3, 4]],
        "exponential_decay_length_penalty": (1, 1.0),
        "pd_master_node_id": pd_master_node_id,
        "pd_kv_trans_params": pd_kv_trans_params,
    }
    params.init(None, **data)

    assert params.best_of == 2
    assert params.n == 2
    assert params.do_sample is True
    assert params.presence_penalty == 0.5
    assert params.temperature == 1.0
    assert params.stop_sequences.size == 2
    assert params.pd_master_node_id.get() == pd_master_node_id
    assert params.pd_kv_trans_params.get() == pd_kv_trans_params


@pytest.mark.parametrize(
    "kwargs,expected",
    [({}, False), ({"include_stop_str_in_output": False}, False), ({"include_stop_str_in_output": True}, True)],
)
def test_include_stop_str_in_output_round_trip(kwargs, expected):
    params = SamplingParams()
    params.init(None, **kwargs)
    assert params.include_stop_str_in_output is expected
    assert params.to_dict()["include_stop_str_in_output"] is expected
    assert SamplingParams.from_buffer_copy(params).include_stop_str_in_output is expected

    restored = SamplingParams()
    restored.init(None, **params.to_dict())
    assert restored.include_stop_str_in_output is expected

    py_params = PySamplingParams(**kwargs)
    assert py_params.include_stop_str_in_output is expected
    assert py_params.to_dict()["include_stop_str_in_output"] is expected


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({}, StopStrMatchMode.ALL_TEXT),
        ({"stop_str_match_mode": StopStrMatchMode.ALL_TEXT}, StopStrMatchMode.ALL_TEXT),
        ({"stop_str_match_mode": StopStrMatchMode.CONTENT_ONLY}, StopStrMatchMode.CONTENT_ONLY),
    ],
)
def test_stop_matching_policy_round_trip(kwargs, expected):
    params = SamplingParams()
    params.init(None, **kwargs)
    serialized = params.to_dict()
    assert serialized["stop_str_match_mode"] == expected
    assert type(serialized["stop_str_match_mode"]) is int
    assert "enable_stop_match_in_backend" not in serialized
    assert "_skip_stop_in_reasoning" not in serialized
    restored = SamplingParams()
    restored.init(None, **serialized)
    py_params = PySamplingParams(**kwargs)
    for candidate in (params, SamplingParams.from_buffer_copy(params), restored, py_params):
        assert candidate.stop_str_match_mode.get_mode() == expected
        assert candidate.stop_str_match_mode.is_all_text() is (expected == StopStrMatchMode.ALL_TEXT)
        assert candidate.stop_str_match_mode.is_content_only() is (expected == StopStrMatchMode.CONTENT_ONLY)
        assert candidate.to_dict()["stop_str_match_mode"] == expected
    py_restored = PySamplingParams(**py_params.to_dict())
    assert py_restored.stop_str_match_mode.get_mode() == expected


@pytest.mark.parametrize("mode", [-1, 0, 3, 2 ** 32, "CONTENT_ONLY", None, 0.0])
def test_invalid_stop_str_match_mode(mode):
    params = SamplingParams()
    with pytest.raises((ValueError, TypeError)):
        params.init(None, stop_str_match_mode=mode)
    with pytest.raises((ValueError, TypeError)):
        PySamplingParams(stop_str_match_mode=mode)


def test_stop_str_match_mode_updates_shared_buffer():
    params = SamplingParams()
    params.init(None, include_stop_str_in_output=True, _initial_reasoning_state=1)
    storage = bytearray(bytes(params))
    writer = SamplingParams.from_buffer(storage)
    reader = SamplingParams.from_buffer(storage)
    writer.stop_str_match_mode.set_mode(StopStrMatchMode.CONTENT_ONLY)
    assert reader.stop_str_match_mode.is_content_only()
    assert reader.to_dict()["stop_str_match_mode"] == StopStrMatchMode.CONTENT_ONLY
    assert reader.include_stop_str_in_output is True
    assert reader._initial_reasoning_state == 1
    assert params.stop_str_match_mode.is_all_text()
    with pytest.raises(ValueError):
        writer.stop_str_match_mode.set_mode(3)
    assert reader.stop_str_match_mode.is_content_only()


def test_infer_high_priority_is_internal_and_ignores_external_value():
    params = SamplingParams()
    params.init(None, infer_high_priority=-3)
    assert params.infer_high_priority == 0


def _make_params(**overrides):
    """Build a SamplingParams whose fields are valid by default, applying overrides.

    ``do_sample=True`` is used so that the sampling-related fields (temperature, top_p,
    top_k) are kept as provided; with greedy decoding ``init`` overrides them to defaults.
    """
    data = {
        "best_of": 1,
        "n": 1,
        "do_sample": True,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_new_tokens": 16,
        "min_new_tokens": 1,
    }
    data.update(overrides)
    params = SamplingParams()
    params.init(None, **data)
    return params


def test_verify_accepts_valid_defaults():
    # A minimally specified, valid configuration must pass verification.
    _make_params().verify()


def test_verify_accepts_n_equal_best_of_greater_than_one():
    params = _make_params(best_of=2, n=2)
    params.verify()
    assert params.n == params.best_of == 2


def test_verify_rejects_n_not_equal_best_of():
    # The engine currently only supports n == best_of; a mismatch must be rejected.
    with pytest.raises(ValueError):
        _make_params(best_of=2, n=1).verify()


@pytest.mark.parametrize("best_of", [0, -1, MAX_BEST_OF + 1])
def test_verify_rejects_best_of_out_of_range(best_of):
    with pytest.raises(ValueError):
        _make_params(best_of=best_of, n=best_of).verify()


@pytest.mark.parametrize(
    "field, value",
    [
        ("presence_penalty", -0.1),
        ("frequency_penalty", -0.1),
        ("repetition_penalty", 0.5),
        ("temperature", -1.0),
        ("top_p", 0.0),
        ("top_p", 1.5),
        ("top_k", 0),
        ("top_k", -2),
        ("max_new_tokens", 0),
        ("min_new_tokens", 0),
    ],
)
def test_verify_rejects_invalid_sampling_fields(field, value):
    with pytest.raises(ValueError):
        _make_params(**{field: value}).verify()


def test_verify_rejects_min_new_tokens_greater_than_max():
    with pytest.raises(ValueError):
        _make_params(min_new_tokens=8, max_new_tokens=4).verify()


@pytest.mark.parametrize("top_k", [-1, 1, 50])
def test_verify_accepts_valid_top_k(top_k):
    _make_params(top_k=top_k).verify()


# Mock tokenizer for testing
class MockTokenizer:
    def encode(self, text, add_special_tokens=False):
        # 这里模拟返回 token ids
        return [1, 2] if text == "stop1" else [3, 4] if text == "stop2" else []


# 运行测试
if __name__ == "__main__":
    pytest.main()
