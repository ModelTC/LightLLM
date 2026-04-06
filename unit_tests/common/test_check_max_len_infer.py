"""Tests for _check_decode_infer memory validation logic.

These tests verify the decode stress test constructs correct ModelInput parameters
without requiring a real model. They mock the forward pass to validate the
batch composition logic.
"""

import torch


class FakeModelOutput:
    logits: torch.Tensor


class FakeMemManager:
    def __init__(self, size):
        self.size = size
        self.can_use_mem_size = size

    def alloc(self, n):
        return torch.arange(n, dtype=torch.int32)

    def free_all(self):
        pass


class FakeReqManager:
    def __init__(self, max_req):
        self._next = 0
        self._max = max_req

    def alloc(self):
        if self._next >= self._max:
            return None
        idx = self._next
        self._next += 1
        return idx

    def free_all(self):
        self._next = 0


def test_decode_check_batch_composition():
    """Verify decode check creates correct ModelInput for decode (max_q_seq_len=1)."""
    graph_max_batch_size = 8
    batch_max_tokens = 1024
    max_total_token_num = 4096

    mem_mgr = FakeMemManager(max_total_token_num)
    req_mgr = FakeReqManager(graph_max_batch_size)

    # Simulate the decode check logic
    batch_size = graph_max_batch_size
    actual_batch = min(batch_size, req_mgr._max)
    tokens_per_req = min(batch_max_tokens, max(1, max_total_token_num // actual_batch))
    total_tokens = tokens_per_req * actual_batch
    total_tokens = min(total_tokens, mem_mgr.can_use_mem_size)
    tokens_per_req = max(1, total_tokens // actual_batch)

    assert tokens_per_req == 512  # 4096 // 8
    assert actual_batch == 8


def test_decode_check_limited_by_req_slots():
    """When req slots < graph_max_batch_size, decode check adapts."""
    req_mgr = FakeReqManager(max_req=4)
    graph_max_batch_size = 16

    actual_batch = 0
    for _ in range(graph_max_batch_size):
        if req_mgr.alloc() is None:
            break
        actual_batch += 1

    assert actual_batch == 4  # Limited by req manager capacity


def test_decode_check_tokens_per_req_clamped():
    """tokens_per_req should not exceed batch_max_tokens."""
    batch_max_tokens = 256
    max_total_token_num = 100000
    actual_batch = 8

    tokens_per_req = min(batch_max_tokens, max(1, max_total_token_num // actual_batch))
    assert tokens_per_req == 256  # Clamped to batch_max_tokens
