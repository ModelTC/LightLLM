from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from lightllm.models.deepseek_v41.engram import EngramHasher


def test_shared_metadata_cache_preserves_values_and_invalidates_inputs(tmp_path, monkeypatch):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"A": 0, "a": 1, "b": 2, "[PAD]": 3}))
    )
    config = dict(
        engram_compressed_vocab_size=3,
        engram_pad_token_id=3,
        engram_max_ngram_size=4,
        engram_layer_ids=[1, 14],
        engram_vocab_size=101,
        engram_n_heads=2,
    )
    build = EngramHasher._build_cpu_metadata
    expected = build(config, tokenizer)
    calls = []

    def counted_build(config, tokenizer):
        calls.append(None)
        return build(config, tokenizer)

    monkeypatch.setattr(EngramHasher, "_build_cpu_metadata", staticmethod(counted_build))
    monkeypatch.setenv("LIGHTLLM_ENGRAM_CACHE_DIR", str(tmp_path))
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda _: EngramHasher._load_cpu_metadata(config, tokenizer), range(4)))
    assert len(calls) == 1
    for result in results:
        for actual, wanted in zip(result, expected):
            np.testing.assert_array_equal(actual, wanted)
    changed_config = dict(config, engram_vocab_size=103)
    changed = EngramHasher._load_cpu_metadata(changed_config, tokenizer)
    assert len(calls) == 2
    assert not np.array_equal(changed[1], expected[1])
    tokenizer.add_tokens(["B"])
    EngramHasher._load_cpu_metadata(config, tokenizer)
    assert len(calls) == 3
    assert len(list(tmp_path.glob("*.npz"))) == 3
