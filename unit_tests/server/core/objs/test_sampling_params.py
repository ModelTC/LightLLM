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
    NodeUUId,
    STOP_SEQUENCE_MAX_LENGTH,
    REGULAR_CONSTRAINT_MAX_LENGTH,
    ALLOWED_TOKEN_IDS_MAX_LENGTH,
    JSON_SCHEMA_MAX_LENGTH,
    GRAMMAR_CONSTRAINT_MAX_LENGTH,
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


def test_node_uuid_roundtrip():
    node_id = 12345678901234567890
    uuid = NodeUUId()
    uuid.initialize(node_id)
    assert uuid.node_id_high == (node_id >> 64) & 0xFFFFFFFFFFFFFFFF
    assert uuid.node_id_low == node_id & 0xFFFFFFFFFFFFFFFF
    assert uuid.get() == node_id


def test_allowed_token_ids_rejects_non_int():
    allowed_ids = AllowedTokenIds()
    with pytest.raises(AssertionError):
        allowed_ids.initialize([1, "2", 3])


@pytest.mark.parametrize(
    "stop_sequences, expected_token_ids, expected_strings",
    [
        # 全部有效，无过滤，token id 组与字符串一一对应。
        (["stop1", "stop2"], [[1, 2], [3, 4]], ["stop1", "stop2"]),
        # 前置的空字符串会被过滤掉，后面的条目不能因此错位。
        (["", "stop1"], [[1, 2]], ["stop1"]),
        # 前置的空 id 列表同理。
        ([[], "stop2"], [[3, 4]], ["stop2"]),
        # 被过滤掉的字符串条目不能把自己的字符串挂到后一个条目的 token id 上。
        (["unknown", "stop2"], [[3, 4]], ["stop2"]),
        # 纯 id 条目不携带字符串。
        ([[7, 8], "stop1"], [[7, 8], [1, 2]], ["stop1"]),
    ],
)
def test_stop_sequence_groups_keeps_ids_and_strings_aligned(stop_sequences, expected_token_ids, expected_strings):
    groups = StopSequenceGroups()
    groups.initialize(stop_sequences, MockTokenizer())
    assert groups.to_list() == expected_token_ids
    assert sorted(groups.to_strings()) == sorted(expected_strings)


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
        "move_kv_to_decode_node": None,
    }
    params.init(None, **data)

    assert params.best_of == 2
    assert params.n == 2
    assert params.do_sample is True
    assert params.presence_penalty == 0.5
    assert params.temperature == 1.0
    assert params.stop_sequences.size == 2


# Mock tokenizer for testing
class MockTokenizer:
    def encode(self, text, add_special_tokens=False):
        # 这里模拟返回 token ids
        return [1, 2] if text == "stop1" else [3, 4] if text == "stop2" else []


# 运行测试
if __name__ == "__main__":
    pytest.main()
