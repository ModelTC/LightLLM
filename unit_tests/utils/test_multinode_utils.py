import os
import pickle
import pytest

from lightllm.utils.multinode_utils import _decode_child_ip, _MAX_CHILD_IP_BYTES


class _RceProbe:
    """Pickle payload that would run code if the receiver called pickle.loads()."""

    MARKER = "/tmp/lightllm_multinode_rce_probe"

    def __reduce__(self):
        return (os.system, (f"touch {self.MARKER}",))


@pytest.mark.parametrize("ip", ["127.0.0.1", "10.0.0.7", "192.168.1.255", "::1", "fe80::1"])
def test_decode_child_ip_accepts_valid_addresses(ip):
    assert _decode_child_ip(ip.encode("utf-8")) == ip


def test_decode_child_ip_strips_surrounding_whitespace():
    assert _decode_child_ip(b"  10.0.0.7\n") == "10.0.0.7"


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"not-an-ip",
        b"10.0.0.256",
        b"127.0.0.1; rm -rf /",
        b"\xff\xfe\xfd",  # invalid utf-8
    ],
)
def test_decode_child_ip_rejects_malformed_payloads(raw):
    with pytest.raises((ValueError, UnicodeDecodeError)):
        _decode_child_ip(raw)


def test_decode_child_ip_rejects_oversized_payload():
    with pytest.raises(ValueError, match="too large"):
        _decode_child_ip(b"1" * (_MAX_CHILD_IP_BYTES + 1))


@pytest.mark.parametrize("protocol", [0, 1, 2, pickle.HIGHEST_PROTOCOL])
def test_malicious_pickle_payload_is_rejected_without_executing(protocol):
    """The regression this module exists for.

    A pickled payload must be rejected as a malformed IP. Critically, it must be
    rejected *without* being deserialized -- the marker file proves no code ran.
    """
    payload = pickle.dumps(_RceProbe(), protocol=protocol)
    if os.path.exists(_RceProbe.MARKER):
        os.remove(_RceProbe.MARKER)

    try:
        with pytest.raises((ValueError, UnicodeDecodeError)):
            _decode_child_ip(payload)

        assert not os.path.exists(_RceProbe.MARKER), "pickle payload was executed -- deserialization still reachable"
    finally:
        if os.path.exists(_RceProbe.MARKER):
            os.remove(_RceProbe.MARKER)


class _CompactRceProbe:
    """Worst case for the validator: a pickle small and plain enough to reach ip_address().

    Protocol 0 serialises to pure ASCII, and with a short command the payload stays
    under the size cap -- so neither the length check nor the utf-8 decode rejects it.
    That leaves ipaddress.ip_address() as the only thing standing between this payload
    and execution, which is exactly the property worth pinning down.
    """

    MARKER = "/tmp/lm_rce"

    def __reduce__(self):
        return (os.system, (f"touch {self.MARKER}",))


def test_compact_ascii_pickle_reaches_and_is_stopped_by_ip_validation():
    payload = pickle.dumps(_CompactRceProbe(), protocol=0)

    # Preconditions: this payload really does slip past the two cheaper checks.
    assert len(payload) <= _MAX_CHILD_IP_BYTES, f"payload {len(payload)}B no longer exercises the ip_address() path"
    assert payload.isascii(), "payload must be valid utf-8 to exercise the ip_address() path"
    payload.decode("utf-8")

    if os.path.exists(_CompactRceProbe.MARKER):
        os.remove(_CompactRceProbe.MARKER)
    try:
        with pytest.raises(ValueError):
            _decode_child_ip(payload)
        assert not os.path.exists(_CompactRceProbe.MARKER), "pickle payload was executed"
    finally:
        if os.path.exists(_CompactRceProbe.MARKER):
            os.remove(_CompactRceProbe.MARKER)
