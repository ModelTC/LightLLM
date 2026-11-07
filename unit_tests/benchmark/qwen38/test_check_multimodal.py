import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def checker(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "test/benchmark/qwen38"))
    return importlib.import_module("check_multimodal")


def fixture():
    case = dict(prompt_tokens=90, image_tokens=64, max_new_tokens=64, ignore_eos=False, expected_words=["red"])
    body = dict(
        prompt_tokens=90,
        count_output_tokens=2,
        finish_reason="stop",
        tokens=[dict(id=7), dict(id=248046)],
        input_usage=dict(input_text_tokens=26, input_image_tokens=64, input_audio_tokens=0),
    )
    tokenizer = SimpleNamespace(decode=lambda ids, **_: " ".join({7: "Red", 8: "blue"}[value] for value in ids))
    return case, body, tokenizer


def test_image_count_eos_and_semantic_validation(checker):
    case, body, tokenizer = fixture()
    result = checker.validate_image_result(body, case, [248044, 248046], tokenizer)
    assert result["token_ids"] == [7, 248046] and result["answer"] == "Red"


@pytest.mark.parametrize(
    "field,value", [("input_text_tokens", 27), ("input_image_tokens", 63), ("input_audio_tokens", 1)]
)
def test_incorrect_modality_counts_cannot_pass(checker, field, value):
    case, body, tokenizer = fixture()
    body["input_usage"][field] = value
    with pytest.raises(ValueError, match="accounting"):
        checker.validate_image_result(body, case, [248046], tokenizer)


def test_wrong_color_and_swapped_image_order_cannot_pass(checker):
    case, body, tokenizer = fixture()
    body["tokens"][0]["id"] = 8
    with pytest.raises(ValueError, match="color/order"):
        checker.validate_image_result(body, case, [248046], tokenizer)
    case["expected_words"] = ["red", "blue"]
    body["tokens"] = [dict(id=8), dict(id=7), dict(id=248046)]
    body["count_output_tokens"] = 3
    with pytest.raises(ValueError, match="color/order"):
        checker.validate_image_result(body, case, [248046], tokenizer)


def test_pressure_request_has_explicit_fixed_length_contract(checker):
    case, body, tokenizer = fixture()
    case.update(expected_words=[], max_new_tokens=1, ignore_eos=True)
    body.update(tokens=[dict(id=7)], count_output_tokens=1, finish_reason="length")
    assert checker.validate_image_result(body, case, [248046], tokenizer)["token_ids"] == [7]


def cache_events():
    first = dict(id=1, token_id=100000000, token_num=64, md5="digest_0")
    last = dict(id=2, token_id=100000384, token_num=64, md5="digest_0")
    return [
        dict(event="alloc", records=[first]),
        dict(event="embedded", records=[first]),
        dict(event="evicted", record=first),
        dict(event="alloc", records=[last]),
        dict(event="embedded", records=[last]),
    ]


def test_reallocation_requires_new_virtual_ids_eviction_and_embedding_completion(checker):
    proof = checker.verify_reallocation(cache_events(), "digest")
    assert proof["original_evicted"] and proof["both_embedded"]
    assert proof["original"]["token_id"] != proof["reallocated"]["token_id"]


@pytest.mark.parametrize("missing", ["alloc", "evicted", "embedded"])
def test_incomplete_cache_evidence_cannot_pass(checker, missing):
    events = [item for item in cache_events() if item["event"] != missing]
    with pytest.raises(ValueError):
        checker.verify_reallocation(events, "digest")


def test_cache_reallocation_must_change_virtual_range(checker):
    events = cache_events()
    events[3]["records"][0]["token_id"] = events[0]["records"][0]["token_id"]
    with pytest.raises(ValueError, match="new virtual IDs"):
        checker.verify_reallocation(events, "digest")


def test_fixture_hashes_distinguish_cache_pressure_images(checker, tmp_path):
    payload, images = checker.image_fixtures(tmp_path, 5)
    assert len(payload) == len(images) == 8
    assert len({image["sha256"] for image in images.values()}) == 8
    assert images["rect_blue"]["width"] == 384 and images["rect_blue"]["height"] == 128
