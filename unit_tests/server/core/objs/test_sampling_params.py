import pytest
from lightllm.server.core.objs.sampling_params import (
    StopSequence,
    StopSequenceGroups,
    RegularConstraint,
    AllowedTokenIds,
    ExponentialDecayLengthPenalty,
    SamplingParams,
    GuidedGrammar,
    GuidedJsonSchema,
    STOP_SEQUENCE_MAX_LENGTH,
    REGULAR_CONSTRAINT_MAX_LENGTH,
    ALLOWED_TOKEN_IDS_MAX_LENGTH,
    JSON_SCHEMA_MAX_LENGTH,
    GRAMMAR_CONSTRAINT_MAX_LENGTH,
    MAX_BEST_OF,
)

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
    "stop_sequences, expected",
    [
        (["stop1", "stop2"], [[1, 2], [3, 4]]),  # 根据 MockTokenizer 返回的 ID
        ([], []),  # 空输入
    ],
)
def test_stop_sequence_groups_initialization(stop_sequences, expected):
    tokenizer = MockTokenizer()
    groups = StopSequenceGroups()
    groups.initialize(stop_sequences, tokenizer)
    assert groups.size == len(expected)
    assert groups.to_list() == expected


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


def test_allowed_token_ids_initialization():
    allowed_ids = AllowedTokenIds()
    allowed_ids.initialize([1, 2, 3])
    assert allowed_ids.size == 3
    assert allowed_ids.to_list() == [1, 2, 3]

    with pytest.raises(AssertionError):
        allowed_ids.initialize([1] * (ALLOWED_TOKEN_IDS_MAX_LENGTH + 1))


def test_exponential_decay_length_penalty_initialization():
    penalty = ExponentialDecayLengthPenalty()
    penalty.initialize((5, 1.5))
    assert penalty.to_tuple() == (5, 1.5)

    with pytest.raises(AssertionError):
        penalty.initialize((5, 0.5))


def test_sampling_params_initialization():
    params = SamplingParams()
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
        "allowed_token_ids": [1, 2, 3],
        "stop_sequences": [[2, 1], [3, 4]],
        "exponential_decay_length_penalty": (1, 1.0),
    }
    params.init(None, **data)

    assert params.best_of == 2
    assert params.n == 2
    assert params.do_sample is True
    assert params.presence_penalty == 0.5
    assert params.temperature == 1.0
    assert params.stop_sequences.size == 2


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


def test_verify_rejects_regular_constraint_with_allowed_token_ids():
    # regular_constraint and allowed_token_ids are mutually exclusive.
    with pytest.raises(ValueError):
        _make_params(regular_constraint="[a-z]+", allowed_token_ids=[1, 2, 3]).verify()


# Mock tokenizer for testing
class MockTokenizer:
    def encode(self, text, add_special_tokens=False):
        # 这里模拟返回 token ids
        return [1, 2] if text == "stop1" else [3, 4] if text == "stop2" else []


# 运行测试
if __name__ == "__main__":
    pytest.main()
