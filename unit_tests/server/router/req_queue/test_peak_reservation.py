import random

import pytest

from lightllm.server.router.req_queue.peak_reservation import Reservation, estimate_peak


def simulated_peak(entries, page_tokens):
    totals = []
    for progress in range(max((entry.remaining_tokens for entry in entries), default=0) + 1):
        pages, private = set(), 0
        for entry in entries:
            if entry.remaining_tokens < progress:
                continue
            private += entry.base_tokens + progress
            if entry.cache_domain is not None:
                private -= len(entry.prefix_pages) * page_tokens
                for depth in range(1, len(entry.prefix_pages) + 1):
                    pages.add((entry.cache_domain, entry.prefix_pages[:depth]))
        totals.append(private + len(pages) * page_tokens)
    return max(totals)


def test_shared_prefix_reservation_changes_admission_without_increasing_capacity():
    entries = [Reservation(1800, 350, (b"same exact page",), b"one cache") for _ in range(8)]
    peak = estimate_peak(entries, 512)
    assert peak.legacy_tokens == 17200
    assert peak.shared_tokens == simulated_peak(entries, 512) == 13616
    assert peak.shared_tokens < 16384 < peak.legacy_tokens


def test_prefix_ancestry_and_cache_domain_prevent_false_sharing():
    entries = [
        Reservation(20, 3, (b"a", b"same"), b"cache"),
        Reservation(20, 3, (b"b", b"same"), b"cache"),
        Reservation(20, 3, (b"a", b"same"), b"other cache"),
    ]
    peak = estimate_peak(entries, 4)
    assert peak.legacy_tokens == peak.shared_tokens == simulated_peak(entries, 4)


def test_unknown_domains_keep_legacy_reservation():
    entries = [Reservation(20, remaining, (b"same",)) for remaining in (1, 3, 7)]
    peak = estimate_peak(entries, 4)
    assert peak.legacy_tokens == peak.shared_tokens == simulated_peak(entries, 4)


def test_credits_follow_each_live_subset_when_requests_complete():
    entries = [
        Reservation(25, 2, (b"a", b"b", b"c"), b"cache"),
        Reservation(25, 8, (b"a",), b"cache"),
        Reservation(25, 5, (b"a", b"b"), b"cache"),
    ]
    assert estimate_peak(entries, 4).shared_tokens == simulated_peak(entries, 4)


def test_reservation_matches_independent_simulated_lifetimes():
    rng = random.Random(381020)
    for _ in range(1500):
        page_tokens = rng.choice((1, 2, 4, 512))
        entries = []
        for _ in range(rng.randrange(1, 9)):
            pages = tuple(rng.choice((b"a", b"b", b"c")) for _ in range(rng.randrange(5)))
            entries.append(
                Reservation(
                    len(pages) * page_tokens + rng.randrange(8),
                    rng.randrange(13),
                    pages,
                    rng.choice((None, b"x", b"y")),
                )
            )
        peak = estimate_peak(entries, page_tokens)
        assert peak.shared_tokens == simulated_peak(entries, page_tokens)
        assert 0 <= peak.shared_tokens <= peak.legacy_tokens


@pytest.mark.parametrize(
    "entry,page_tokens",
    [
        (Reservation(-1, 1), 1),
        (Reservation(1, -1), 1),
        (Reservation(True, 1), 1),
        (Reservation(1, 1), 0),
        (Reservation(1, 1), True),
        (Reservation(1, 1, (b"a", b"b"), b"cache"), 1),
        (Reservation(1, 1, ("text is not an exact byte identity",), b"cache"), 1),
        (Reservation(1, 1, (b"a",), "mutable or ambiguous domain"), 1),
    ],
)
def test_rejects_invalid_accounting_metadata(entry, page_tokens):
    with pytest.raises(ValueError):
        estimate_peak([entry], page_tokens)


def test_empty_batch_has_no_reservation():
    assert estimate_peak([], 512).shared_tokens == 0
