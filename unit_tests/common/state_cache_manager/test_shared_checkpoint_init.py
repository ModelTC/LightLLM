from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import numpy as np
import torch
import torch.distributed as dist

from lightllm.server.router.model_infer.mode_backend import generic_pre_process  # noqa: F401
from lightllm.common.kv_cache_mem_manager import qwen3next_mem_manager as memory_module
from lightllm.common.state_cache_manager import LayerCache, LinearAttCacheManager
from lightllm.server.core.objs import atomic_lock
from lightllm.utils import shm_utils, dist_utils, envs_utils


def _config():
    return NS(
        conv_state_dtype=torch.bfloat16,
        ssm_state_dtype=torch.float32,
        linear_layer_num=2,
        get_conv_state_shape=lambda: (4, 3),
        get_ssm_state_shape=lambda: (2, 4, 3),
    )


@pytest.mark.parametrize(
    "manager_type",
    [
        memory_module.Qwen3NextMemManager,
        memory_module.FP8StaticPerHeadQuantQwen3NextMemManager,
        memory_module.FP8StaticPerTensorQuantQwen3NextMemManager,
    ],
)
@pytest.mark.parametrize("dp,nnodes", [(1, 1), (2, 1), (2, 2), (4, 2), (1, 2)])
def test_big_pool_initializes_in_one_call(monkeypatch, manager_type, dp, nnodes):
    args = NS(dp=dp, nnodes=nnodes, linear_att_page_block_num=4, linear_att_hash_page_size=64)
    monkeypatch.setattr(memory_module, "get_env_start_args", lambda: args)
    monkeypatch.setattr(envs_utils, "get_env_start_args", lambda: args)
    group_factory, barrier = Mock(), Mock()
    monkeypatch.setattr(dist_utils, "create_new_group_for_current_node", group_factory)
    monkeypatch.setattr(memory_module, "get_dp_rank_in_node", lambda: 0)
    monkeypatch.setattr(dist_utils, "get_dp_rank_in_node", lambda: 0)
    monkeypatch.setattr(dist_utils, "get_current_rank_in_dp", lambda: 0)
    monkeypatch.setattr(dist, "barrier", barrier)
    dp_size_in_node = max(1, dp // nnodes)
    pool_constructor = Mock()

    def construct(pool, **kwargs):
        pool.__dict__.update(kwargs)
        pool_constructor(**kwargs)

    monkeypatch.setattr(LinearAttCacheManager, "__init__", construct)
    zeros = Mock()
    monkeypatch.setattr(torch, "zeros", zeros)
    manager = object.__new__(manager_type)
    manager.size, manager.linear_config = 2048, _config()
    manager._init_linear_att_buffers()
    assert manager.CPU_CACHE_BIG_PAGE_LOAD_TEMP_BUFFER_ID == 8 * dp_size_in_node
    assert manager.CPU_CACHE_BIG_PAGE_OFFLOAD_TEMP_BUFFER_ID == 8 * dp_size_in_node + 1
    if dp_size_in_node > 1:
        pool_constructor.assert_called_once_with(
            size=10 * dp_size_in_node,
            linear_config=manager.linear_config,
            keep_num=2 * dp_size_in_node,
            shm_name="dp_big_checkpoints_tp_0",
            shm_create=True,
        )
        assert isinstance(manager.big_page_buffers, LinearAttCacheManager)
        group_factory.assert_called_once_with("gloo")
        assert barrier.call_count == 2
        zeros.assert_not_called()
    else:
        group_factory.assert_not_called()
        barrier.assert_not_called()
        pool_constructor.assert_called_once_with(size=10, keep_num=2, linear_config=manager.linear_config)
        assert isinstance(manager.big_page_buffers, LinearAttCacheManager)


@pytest.mark.parametrize("create", [False, True])
def test_shared_pool_constructor_never_allocates_private_storage(monkeypatch, create):
    zeros, share_memory = Mock(), Mock()
    monkeypatch.setattr(torch, "zeros", zeros)
    monkeypatch.setattr(LayerCache, "init_share_memory", share_memory)
    metadata = np.array([15, 1, *range(16), 1, *([0] * 15)], dtype=np.int64)
    original_metadata = metadata.copy()
    metadata_shm = NS(buf=metadata.data)
    create_or_link_shm = Mock(return_value=metadata_shm)
    shared_lock = Mock()
    monkeypatch.setattr(shm_utils, "create_or_link_shm", create_or_link_shm)
    monkeypatch.setattr(atomic_lock, "AtomicShmLock", shared_lock)
    pool = LinearAttCacheManager(20, _config(), keep_num=4, shm_name="checkpoints", shm_create=create)
    zeros.assert_not_called()
    assert pool.metadata_shm is metadata_shm
    create_or_link_shm.assert_called_once_with(
        "checkpoints_metadata", 34 * 8, force_mode="create" if create else "link"
    )
    shared_lock.assert_called_once_with("checkpoints_lock")
    if create:
        assert pool.metadata[0] == 16
        assert (pool.refs == 0).all()
    else:
        np.testing.assert_array_equal(pool.metadata, original_metadata)
    assert [call.args for call in share_memory.call_args_list] == [
        ("checkpoints_conv", 20, create),
        ("checkpoints_ssm", 20, create),
    ]


def test_private_small_pool_still_allocates_pinned_buffers(monkeypatch):
    zeros = Mock()
    monkeypatch.setattr(torch, "zeros", zeros)
    pool = LinearAttCacheManager(3, _config())
    assert pool.metadata_shm is None
    assert zeros.call_count == 2
    assert all(call.kwargs["pin_memory"] for call in zeros.call_args_list)


@pytest.mark.parametrize("rank", range(4))
def test_shared_pool_groups_corresponding_tp_shards_and_reserves_per_dp_scratch(monkeypatch, rank):
    args = NS(dp=2, nnodes=1, linear_att_page_block_num=4, linear_att_hash_page_size=64)
    monkeypatch.setattr(memory_module, "get_env_start_args", lambda: args)
    monkeypatch.setattr(envs_utils, "get_env_start_args", lambda: args)
    monkeypatch.setattr(dist_utils, "create_new_group_for_current_node", lambda backend: None)
    monkeypatch.setattr(memory_module, "get_dp_rank_in_node", lambda: rank // 2)
    monkeypatch.setattr(dist_utils, "get_dp_rank_in_node", lambda: rank // 2)
    monkeypatch.setattr(dist_utils, "get_current_rank_in_dp", lambda: rank % 2)
    manager = object.__new__(memory_module.Qwen3NextMemManager)
    manager.size, manager.linear_config = 768, _config()
    events = []
    pool_constructor = Mock()
    pool_constructor.side_effect = lambda **kwargs: events.append("construct") or pool_constructor.return_value

    def construct(pool, **kwargs):
        pool.__dict__.update(kwargs)
        pool_constructor(**kwargs)

    monkeypatch.setattr(LinearAttCacheManager, "__init__", construct)
    monkeypatch.setattr(dist, "barrier", lambda group: events.append("barrier"))
    manager._init_linear_att_buffers()
    capacity = 6
    pool_constructor.assert_called_once_with(
        size=capacity + 4,
        linear_config=manager.linear_config,
        keep_num=4,
        shm_name=f"dp_big_checkpoints_tp_{rank % 2}",
        shm_create=rank < 2,
    )
    assert isinstance(manager.big_page_buffers, LinearAttCacheManager)
    assert events == (["construct", "barrier", "barrier"] if rank < 2 else ["barrier", "construct", "barrier"])
    assert manager.CPU_CACHE_BIG_PAGE_LOAD_TEMP_BUFFER_ID == capacity + (rank // 2) * 2
    assert manager.CPU_CACHE_BIG_PAGE_OFFLOAD_TEMP_BUFFER_ID == capacity + (rank // 2) * 2 + 1
