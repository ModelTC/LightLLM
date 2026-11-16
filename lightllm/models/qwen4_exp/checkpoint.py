import math
from functools import lru_cache


def normalize_text_weight_names(weights: dict):
    prefix = "model.language_model."
    for name in tuple(weights):
        if name.startswith(prefix):
            target = "model." + name[len(prefix) :]
            if target in weights and weights[target] is not weights[name]:
                raise ValueError(f"Conflicting text checkpoint names: {name} and {target}")
            weights[target] = weights.pop(name)


def ngram_eos_token_id(config: dict) -> int:
    eos = config["eos_token_id"]
    if isinstance(eos, list):
        if not eos:
            raise ValueError("PLE requires an EOS token ID")
        eos = eos[0]
    return eos


@lru_cache(maxsize=None)
def ngram_table_shape(base: int, heads: int, ple_index: int, embedding_dim: int, padding: int):
    if base < 1 or heads < 1 or ple_index < 0 or embedding_dim < 1 or embedding_dim % heads or padding < 1:
        raise ValueError("Invalid n-gram table dimensions")
    sizes = []
    candidate = base - 1
    for index in range((ple_index + 1) * heads):
        while True:
            candidate += 1
            if candidate >= 2 and all(candidate % divisor for divisor in range(2, math.isqrt(candidate) + 1)):
                break
        if index >= ple_index * heads:
            sizes.append(candidate)
    total = sum(sizes)
    return ((total + padding - 1) // padding) * padding, embedding_dim // heads
