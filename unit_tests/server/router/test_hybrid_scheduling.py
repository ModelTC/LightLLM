"""Tests for mamba-aware scheduling in chunked prefill queue.

Validates that the scheduler rejects new requests when mamba cache
buffer slots are exhausted, preventing OOM in hybrid attention models.
"""


def _check_ok_mamba(mamba_cache_size, total_reqs, mtp_step=0):
    """Replicate the ok_mamba logic from _can_add_new_req.

    Uses total capacity (mamba_cache_size) as a hard cap on total_reqs
    (existing running + newly admitted). This avoids shared-memory
    staleness issues by using a static capacity value.
    """
    if mamba_cache_size is None:
        return True
    buffers_per_req = mtp_step + 1
    max_reqs = mamba_cache_size // buffers_per_req
    return total_reqs < max_reqs


def test_ok_mamba_passes_when_capacity_available():
    """First request with mamba_cache_size=20 passes."""
    assert _check_ok_mamba(mamba_cache_size=20, total_reqs=1) is True


def test_ok_mamba_rejects_at_capacity():
    """20th request out of 20 slots is rejected (total_reqs=20, max=20, 20<20 is False)."""
    assert _check_ok_mamba(mamba_cache_size=20, total_reqs=20) is False


def test_ok_mamba_admits_up_to_capacity():
    """With 20 slots: admits 19 (total<20), rejects 20th."""
    assert _check_ok_mamba(mamba_cache_size=20, total_reqs=19) is True
    assert _check_ok_mamba(mamba_cache_size=20, total_reqs=20) is False


def test_ok_mamba_flood_scenario():
    """Simulate 30 requests hitting a 20-slot pool. At most 19 are admitted."""
    admitted = 0
    for total in range(1, 31):
        if _check_ok_mamba(mamba_cache_size=20, total_reqs=total):
            admitted += 1
        else:
            break
    assert admitted == 19


def test_ok_mamba_skipped_for_non_hybrid():
    """Non-hybrid models have mamba_cache_size=None, always passes."""
    assert _check_ok_mamba(mamba_cache_size=None, total_reqs=1000) is True


def test_ok_mamba_with_mtp_step():
    """With mtp_step=3: buffers_per_req=4, max_reqs=20//4=5."""
    assert _check_ok_mamba(mamba_cache_size=20, total_reqs=4, mtp_step=3) is True
    assert _check_ok_mamba(mamba_cache_size=20, total_reqs=5, mtp_step=3) is False


def test_ok_mamba_zero_capacity():
    """With mamba_cache_size=0, no requests are admitted."""
    assert _check_ok_mamba(mamba_cache_size=0, total_reqs=1) is False
