"""Operator resolution is separate from memory, streams and synchronization."""

from abc import ABC, abstractmethod
from typing import Callable


class OperatorBackend(ABC):
    supports_autotune = False

    def greedy_sample(self, logits, requests, temperatures, length_penalty, mask_eos, eos_ids, manager):
        """Optional fused greedy path; None retains the existing sampler unchanged."""
        return None

    @abstractmethod
    def resolve(self, name: str) -> Callable:
        """Bind a named operation once; return the actual implementation."""
        raise NotImplementedError

    @abstractmethod
    def selection_module(self, family: str):
        """Return this backend's full, linear or vision attention selector."""
        raise NotImplementedError

    @abstractmethod
    def select_weight_forward(self, op) -> Callable:
        """Select a weight operation with the existing fallback policy."""
        raise NotImplementedError

    @abstractmethod
    def initialize_linear_attention(self):
        """Install the kernel allocator required by hybrid attention models."""
        raise NotImplementedError
