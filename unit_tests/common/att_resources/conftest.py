from __future__ import annotations

import json
import os
from dataclasses import asdict

import pytest


@pytest.fixture(autouse=True)
def allocator_environment(kv_allocator_environment: None, monkeypatch: pytest.MonkeyPatch):
    """独立资源测试也需要启动参数，以选择页池的共享分配策略。"""
    from lightllm.server.core.objs.start_args_type import StartArgs
    from lightllm.utils.envs_utils import get_env_start_args

    if "LIGHTLLM_START_ARGS" not in os.environ:
        monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(asdict(StartArgs())))
    get_env_start_args.cache_clear()
    yield
    get_env_start_args.cache_clear()


@pytest.fixture
def pd_args(monkeypatch: pytest.MonkeyPatch):
    """为独立资源测试提供与生产路径一致的全局 PD 配置。"""
    from lightllm.server.core.objs.start_args_type import StartArgs
    from lightllm.utils.envs_utils import get_env_start_args

    monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(asdict(StartArgs(pd_kv_page_size=8, cpu_cache_token_page_size=8))))
    get_env_start_args.cache_clear()
    yield get_env_start_args()
    get_env_start_args.cache_clear()


@pytest.fixture
def cpu_args(monkeypatch: pytest.MonkeyPatch):
    """为独立资源测试提供与生产路径一致的全局 CPU 页配置。"""
    from lightllm.server.core.objs.start_args_type import StartArgs
    from lightllm.utils.envs_utils import get_env_start_args

    monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(asdict(StartArgs(cpu_cache_token_page_size=8))))
    get_env_start_args.cache_clear()
    yield get_env_start_args()
    get_env_start_args.cache_clear()
