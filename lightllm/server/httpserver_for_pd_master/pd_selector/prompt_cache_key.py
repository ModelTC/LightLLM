"""Compact prefix hints that cover every character of each contiguous block."""

from hashlib import blake2b


def prompt_cache_key(prompt: str, block_chars: int) -> list[str]:
    if block_chars < 1:
        raise ValueError("block_chars must be positive")
    # These are historical location hints, never proof of KV residency. A
    # growing final partial block loses at most one block of estimated affinity.
    return [
        blake2b(prompt[start : start + block_chars].encode("utf-8"), digest_size=8).hexdigest()
        for start in range(0, len(prompt), block_chars)
    ]
