"""Attention interfaces and compatibility exports, with backend-owned implementations."""

from lightllm.backends import get_operator_backend


from .gdn import LinearAttBackend, LinearAttPrefillAttState, LinearAttDecodeAttState

_BACKEND_EXPORTS = {
    "FlashQlaLinearAttBackend": "attention.linear.FlashQlaLinearAttBackend",
    "TritonLinearAttBackend": "attention.linear.TritonLinearAttBackend",
}
__all__ = [
    "LinearAttBackend",
    "LinearAttPrefillAttState",
    "LinearAttDecodeAttState",
    "FlashQlaLinearAttBackend",
    "TritonLinearAttBackend",
]


def __getattr__(name):
    if name not in _BACKEND_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = get_operator_backend().resolve(_BACKEND_EXPORTS[name])
    globals()[name] = value
    return value
