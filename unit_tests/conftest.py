"""Shared fixtures for tests that construct runtime resources."""

from __future__ import annotations

import logging
import os
from contextlib import suppress
from uuid import uuid4

import pytest


@pytest.fixture(autouse=True)
def single_rank_environment(monkeypatch: pytest.MonkeyPatch):
    """Default standalone tests to one rank without overriding explicit settings."""
    for key, value in {
        "LIGHTLLM_CURRENT_DEVICE_ID": "0",
        "LIGHTLLM_CURRENT_RANK_IN_DP": "0",
        "LIGHTLLM_CURRENT_RANK_IN_NODE": "0",
        "LIGHTLLM_DP_WORLD_SIZE": "1",
    }.items():
        if key not in os.environ:
            monkeypatch.setenv(key, value)


@pytest.fixture
def service_identity(monkeypatch: pytest.MonkeyPatch):
    """Give shared-memory tests a unique service namespace."""
    from lightllm.utils.envs_utils import get_unique_server_name

    monkeypatch.setenv("LIGHTLLM_UNIQUE_SERVICE_NAME_ID", "unit_test_" + uuid4().hex)
    get_unique_server_name.cache_clear()
    yield
    get_unique_server_name.cache_clear()


@pytest.fixture
def capture_lightllm_logs(caplog):
    """Capture loggers that deliberately disable propagation in production."""
    attached = []

    def capture(name: str):
        logger = logging.getLogger(name)
        logger.addHandler(caplog.handler)
        attached.append(logger)

    yield capture
    for logger in attached:
        logger.removeHandler(caplog.handler)


@pytest.fixture
def kv_allocator_environment(monkeypatch: pytest.MonkeyPatch):
    """Provide a service identity and reclaim the allocator's real shared counters."""
    from lightllm.common.attention_state import AttentionStateManager
    from lightllm.common.attention_state.allocators import full_att_kv_allocator
    from lightllm.utils.envs_utils import get_unique_server_name

    monkeypatch.setenv("LIGHTLLM_CURRENT_RANK_IN_NODE", "0")
    monkeypatch.setenv("LIGHTLLM_UNIQUE_SERVICE_NAME_ID", "test_kv_allocator_" + uuid4().hex)
    get_unique_server_name.cache_clear()
    counters = []
    shared_int = full_att_kv_allocator.SharedInt

    def track_counter(name: str):
        counter = shared_int(name)
        counters.append(counter)
        return counter

    monkeypatch.setattr(full_att_kv_allocator, "SharedInt", track_counter)
    yield
    AttentionStateManager._get_states.cache_clear()
    names = set()
    for counter in counters:
        if counter.shm.name not in names:
            with suppress(FileNotFoundError):
                counter.shm.unlink()
            names.add(counter.shm.name)
        counter.shm.close()
    get_unique_server_name.cache_clear()
