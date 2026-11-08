from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class Reservation:
    base_tokens: int
    remaining_tokens: int
    prefix_pages: Tuple[bytes, ...] = ()
    cache_domain: Optional[bytes] = None


@dataclass(frozen=True)
class Peak:
    legacy_tokens: int
    shared_tokens: int
    max_prefix_credit: int


def estimate_peak(entries: Sequence[Reservation], page_tokens: int) -> Peak:
    if type(page_tokens) is not int or page_tokens <= 0:
        raise ValueError("Page token count must be a positive integer")
    for entry in entries:
        if (
            type(entry.base_tokens) is not int
            or entry.base_tokens < 0
            or type(entry.remaining_tokens) is not int
            or entry.remaining_tokens < 0
        ):
            raise ValueError("Reservation counts must be nonnegative integers")
        if not isinstance(entry.prefix_pages, tuple) or any(type(page) is not bytes for page in entry.prefix_pages):
            raise ValueError("Prefix pages must have exact immutable identities")
        if entry.cache_domain is not None and type(entry.cache_domain) is not bytes:
            raise ValueError("An explicit shared domain must have immutable identity")
        if len(entry.prefix_pages) * page_tokens > entry.base_tokens:
            raise ValueError("Known cached pages exceed the resident base")
    seen = set()
    base = credit = legacy = shared = 0
    for count, entry in enumerate(sorted(entries, key=lambda entry: -entry.remaining_tokens), 1):
        base += entry.base_tokens
        if entry.cache_domain is not None:
            prefix = (entry.cache_domain,)
            for page in entry.prefix_pages:
                prefix += (page,)
                if prefix in seen:
                    credit += page_tokens
                else:
                    seen.add(prefix)
        projected = base + count * entry.remaining_tokens
        legacy = max(legacy, projected)
        shared = max(shared, projected - credit)
    return Peak(legacy, shared, credit)
