"""Lightweight control-plane types for activation exchange."""

from .errors import ConsumerExecutionError
from .protocol import (
    ActivationKind,
    ActivationPayload,
    ActivationRef,
    ActivationTransport,
    HiddenStatesActivation,
    HiddenStatesDescriptor,
    KVActivation,
    PrefixKVHandle,
)

__all__ = [
    "ActivationKind",
    "ActivationPayload",
    "ActivationRef",
    "ActivationTransport",
    "ConsumerExecutionError",
    "HiddenStatesActivation",
    "HiddenStatesDescriptor",
    "KVActivation",
    "PrefixKVHandle",
]
