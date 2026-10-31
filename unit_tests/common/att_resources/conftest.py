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
    # 方法缓存以 State 实例为 key；测试会反复创建实例和替换参数，结束后释放缓存引用。
    from lightllm.common.attention_state.full_att import FullAttentionState
    from lightllm.common.attention_state.bounded_att import BoundedAttentionState

    state_classes = [FullAttentionState, BoundedAttentionState]
    for state_class in state_classes:
        state_classes.extend(state_class.__subclasses__())
        for getter in (state_class.get_cpu_transfer_regions, state_class.get_pd_transfer_regions):
            if hasattr(getter, "cache_clear"):
                getter.cache_clear()
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
