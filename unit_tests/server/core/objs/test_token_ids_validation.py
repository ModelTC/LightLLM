import pytest
from lightllm.server.core.objs.sampling_params import (
    StopSequence,
    AllowedTokenIds,
    InvalidTokenIds,
    SamplingParams,
    _check_and_store_int_token_ids,
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


@pytest.mark.parametrize("bad_ids", [[-3], [100, -1], [2 ** 31], [2 ** 40]])
def test_allowed_token_ids_rejects_out_of_range_ids(bad_ids):
    # Out-of-range ids would wrap silently in the c_int buffer (2**31 -> -2147483648,
    # 2**40 -> 0), so they must be rejected at intake instead of being stored garbage.
    allowed_ids = AllowedTokenIds()
    with pytest.raises(AssertionError, match="allowed token ids"):
        allowed_ids.initialize(bad_ids)


def test_allowed_token_ids_accepts_int32_boundary_ids():
    allowed_ids = AllowedTokenIds()
    allowed_ids.initialize([0, 2 ** 31 - 1])
    assert allowed_ids.size == 2
    assert allowed_ids.to_list() == [0, 2 ** 31 - 1]


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


def test_invalid_token_ids_rejects_negative():
    # A negative id survives the GPU-side vocab-size filter and is then used as a
    # pointer offset into the logits tensor by the apply_invalid_token triton kernel.
    invalid_ids = InvalidTokenIds()
    with pytest.raises(AssertionError, match="invalid token ids"):
        invalid_ids.initialize([-5])


@pytest.mark.parametrize("bad_ids", [[2 ** 31], [4, 2 ** 40]])
def test_invalid_token_ids_rejects_int32_overflow(bad_ids):
    # An id >= 2**31 would wrap to a different value in the c_int buffer
    # (2**31 -> -2147483648), silently banning the wrong token.
    invalid_ids = InvalidTokenIds()
    with pytest.raises(AssertionError, match="invalid token ids"):
        invalid_ids.initialize(bad_ids)


def test_invalid_token_ids_accepts_int32_boundary_ids():
    invalid_ids = InvalidTokenIds()
    invalid_ids.initialize([0, 2 ** 31 - 1])
    assert invalid_ids.size == 2
    assert invalid_ids.to_list() == [0, 2 ** 31 - 1]


def test_stop_sequence_rejects_non_int():
    seq = StopSequence()
    with pytest.raises(AssertionError):
        seq.initialize([1, "2"])


@pytest.mark.parametrize("bad_ids", [[-7], [2 ** 31 + 5]])
def test_stop_sequence_rejects_out_of_range_ids(bad_ids):
    seq = StopSequence()
    with pytest.raises(AssertionError, match="stop token ids"):
        seq.initialize(bad_ids)


def test_stop_sequence_accepts_int32_boundary_ids():
    seq = StopSequence()
    seq.initialize([0, 2 ** 31 - 1])
    assert seq.to_list() == [0, 2 ** 31 - 1]


def test_check_and_store_int_token_ids_returns_size_and_writes_buffer():
    import ctypes

    buf = (ctypes.c_int * 8)()
    size = _check_and_store_int_token_ids(buf, [7, 8, 9], 8, "test ids")
    assert size == 3
    assert list(buf[:size]) == [7, 8, 9]


def test_check_and_store_int_token_ids_rejects_overflow():
    import ctypes

    buf = (ctypes.c_int * 2)()
    with pytest.raises(AssertionError):
        _check_and_store_int_token_ids(buf, [1, 2, 3], 2, "test ids")


@pytest.mark.parametrize("bad_key", ["-3", str(2 ** 31)])
def test_logit_bias_out_of_range_keys_rejected(bad_key):
    # logit_bias keys are int()-converted and stored as invalid_token_ids by
    # SamplingParams.init, so out-of-range keys must fail the same range check
    # on the real production path, not only when initialize() is called directly.
    params = SamplingParams()
    with pytest.raises(AssertionError, match="invalid token ids"):
        params.init(None, logit_bias={bad_key: 0.5})


def test_logit_bias_boundary_keys_round_trip():
    params = SamplingParams()
    params.init(None, logit_bias={"0": 1.0, str(2 ** 31 - 1): 0.5})
    assert params.invalid_token_ids.to_list() == [0, 2 ** 31 - 1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
