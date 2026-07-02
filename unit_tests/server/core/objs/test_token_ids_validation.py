import pytest
from lightllm.server.core.objs.sampling_params import (
    StopSequence,
    AllowedTokenIds,
    InvalidTokenIds,
    store_int_token_ids,
    STOP_SEQUENCE_MAX_LENGTH,
    ALLOWED_TOKEN_IDS_MAX_LENGTH,
    INVALID_TOKEN_IDS_MAX_LENGTH,
)


def test_allowed_token_ids_accepts_valid_ints():
    allowed_ids = AllowedTokenIds()
    allowed_ids.initialize([1, 2, 3])
    assert allowed_ids.size == 3
    assert allowed_ids.to_list() == [1, 2, 3]


@pytest.mark.parametrize("bad_ids", [[1, 2, "3"], [1, 2.5], [None, 1]])
def test_allowed_token_ids_rejects_non_int(bad_ids):
    # A non-int entry must fail with the explicit "all must be int" guard,
    # not slip past validation into an opaque ctypes TypeError.
    allowed_ids = AllowedTokenIds()
    with pytest.raises(AssertionError):
        allowed_ids.initialize(bad_ids)


def test_allowed_token_ids_rejects_too_many():
    allowed_ids = AllowedTokenIds()
    with pytest.raises(AssertionError):
        allowed_ids.initialize([1] * (ALLOWED_TOKEN_IDS_MAX_LENGTH + 1))


def test_invalid_token_ids_accepts_valid_ints():
    invalid_ids = InvalidTokenIds()
    invalid_ids.initialize([4, 5, 6])
    assert invalid_ids.size == 3
    assert invalid_ids.to_list() == [4, 5, 6]


@pytest.mark.parametrize("bad_ids", [[4, "5"], [4, 5.0]])
def test_invalid_token_ids_rejects_non_int(bad_ids):
    invalid_ids = InvalidTokenIds()
    with pytest.raises(AssertionError):
        invalid_ids.initialize(bad_ids)


def test_invalid_token_ids_rejects_too_many():
    invalid_ids = InvalidTokenIds()
    with pytest.raises(AssertionError):
        invalid_ids.initialize([1] * (INVALID_TOKEN_IDS_MAX_LENGTH + 1))


def test_stop_sequence_rejects_non_int():
    seq = StopSequence()
    with pytest.raises(AssertionError):
        seq.initialize([1, "2"])


def test_store_int_token_ids_returns_size_and_writes_buffer():
    import ctypes

    buf = (ctypes.c_int * 8)()
    size = store_int_token_ids(buf, [7, 8, 9], 8, "test ids")
    assert size == 3
    assert list(buf[:size]) == [7, 8, 9]


def test_store_int_token_ids_rejects_overflow():
    import ctypes

    buf = (ctypes.c_int * 2)()
    with pytest.raises(AssertionError):
        store_int_token_ids(buf, [1, 2, 3], 2, "test ids")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
