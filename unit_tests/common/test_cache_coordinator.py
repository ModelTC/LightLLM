from types import SimpleNamespace

import pytest
import torch

from lightllm.common.cache_coordinator import HybridCacheCoordinator


@pytest.mark.parametrize(
    "hit", ["big", "small", "small_only", "small_no_ancestor", "fallback", "fallback_no_ancestor", "cold", "existing"]
)
def test_restore_prefix(hit):
    events = []
    big = SimpleNamespace(node_prefix_total_len=8, big_page_buffer_idx=1, is_big_page_node=lambda: True)
    small = SimpleNamespace(node_prefix_total_len=12, small_page_buffer_idx=2, is_big_page_node=lambda: False)
    req = SimpleNamespace(
        req_idx=1,
        args=SimpleNamespace(
            linear_att_hash_page_size=4,
            linear_att_page_block_num=4 if hit == "small_only" else 2,
            max_req_total_len=12 if hit == "small_only" else 64,
            page_size=4,
        ),
        sampling_param=SimpleNamespace(disable_prompt_cache=False),
        shm_req=SimpleNamespace(
            hybrid_token_hash_list=SimpleNamespace(get_all=lambda: [1, 2, 3], size=3),
            shm_prompt_ids=SimpleNamespace(arr=list(range(13))),
            prompt_cache_len=0,
        ),
        cur_kv_len=0,
        get_cur_total_len=lambda: 13,
        shared_kv_node=None,
        tail_small_page_buffer_id=None,
        mm_input_spans=[],
        prompt_selected_logprobs=SimpleNamespace(
            copy_capture_slots_if_needed=lambda **kwargs: events.append("logprobs")
        ),
    )
    if hit == "existing":
        req.cur_kv_len = 8
        req.shared_kv_node = big
        req.shm_req.prompt_cache_len = 4
    if hit.endswith("no_ancestor"):
        small.node_prefix_total_len = 4
    source = None if hit == "cold" else big if hit == "big" else small

    def match(key, **kwargs):
        assert kwargs["update_refs"]
        assert key.tolist() == list(range(12))
        events.append("match")
        count = source.node_prefix_total_len if source is not None else 0
        return source, count, torch.arange(count, dtype=torch.int32)

    def restore(req_idx, conv, ssm):
        assert req_idx == req.req_idx
        expected_slot = 1 if hit in {"big", "fallback"} else 2
        assert conv == ssm == expected_slot
        if hit in {"small", "small_no_ancestor"}:
            assert req.shared_kv_node is None
        events.append("restore")

    manager = HybridCacheCoordinator(
        SimpleNamespace(
            restore_checkpoint=lambda req_idx, length, pool, slot: restore(req_idx, *pool.get_state_cache(slot)),
            reset=lambda req_idx: events.append("reset"),
            conv_state=None,
            ssm_state=None,
        ),
        SimpleNamespace(reset=lambda req_idx: events.append("tail"), state=None),
    )
    manager.small_page_buffers = SimpleNamespace(get_state_cache=lambda slot: (slot, slot))
    mem = SimpleNamespace(
        allocator=SimpleNamespace(can_use_mem_size=4 if hit.startswith("fallback") else 8),
        alloc=lambda need_size: torch.arange(20, 20 + need_size, dtype=torch.int32),
        operator=SimpleNamespace(copy_mem_to_mem=lambda src, dst: events.append("copy")),
    )
    manager.mem_manager = mem
    manager.big_page_buffers = SimpleNamespace(get_state_cache=lambda slot: (slot, slot))
    requests = SimpleNamespace(mem_manager=mem, req_to_token_indexs=torch.full((2, 16), -1, dtype=torch.int32))
    radix = SimpleNamespace(
        match_prefix=match,
        mem_manager=mem,
        get_tree_total_tokens_num=lambda: 12,
        get_refed_tokens_num=lambda: 12,
        free_radix_cache_to_get_enough_token=lambda **kwargs: events.append("reserve"),
        deref_to_first_big_page_node=lambda **kwargs: None if hit.endswith("no_ancestor") else big,
    )
    manager.restore_prefix(req, requests, radix)

    if hit == "existing":
        assert req.cur_kv_len == req.hold_kv_len == req.shm_req.shm_cur_kv_len == 8
        assert req.shared_kv_node is big
        assert req.shm_req.prompt_cache_len == 4
        assert not events
        return
    length = (
        0
        if hit in {"cold", "fallback_no_ancestor"}
        else 8
        if hit in {"big", "fallback"}
        else 4
        if hit == "small_no_ancestor"
        else 12
    )
    assert req.cur_kv_len == req.hold_kv_len == req.shm_req.shm_cur_kv_len == length
    assert req.shm_req.prompt_cache_len == length
    assert req.shared_kv_node is (
        None if hit in {"cold", "small_no_ancestor", "fallback_no_ancestor"} else small if hit == "small_only" else big
    )
    if hit == "small_no_ancestor":
        assert requests.req_to_token_indexs[1, :4].tolist() == list(range(20, 24))
        assert events == ["match", "reserve", "copy", "logprobs", "restore", "tail"]
    elif hit == "small":
        assert requests.req_to_token_indexs[1, :12].tolist() == list(range(8)) + list(range(20, 24))
        assert events == ["match", "reserve", "copy", "logprobs", "restore", "tail"]
    elif hit in {"cold", "fallback_no_ancestor"}:
        assert events == ["match", "reset", "tail"]
        assert (requests.req_to_token_indexs == -1).all()
    else:
        assert requests.req_to_token_indexs[1, :length].tolist() == list(range(length))
        assert events == ["match", "restore", "tail"]


def test_memory_binding_rejects_another_pool():
    manager = HybridCacheCoordinator()
    mem = SimpleNamespace()
    manager.bind_mem_manager(mem)
    manager.bind_mem_manager(mem)
    with pytest.raises(ValueError, match="different memory manager"):
        manager.bind_mem_manager(SimpleNamespace())
    assert manager.mem_manager is mem


@pytest.mark.parametrize("kv_type", ["None", "fp8kv_sph", "fp8kv_spt"])
def test_qwen_runtime_and_cache_share_config(monkeypatch, kv_type):
    from lightllm.models.qwen3next import model as qwen
    from lightllm.common.state_cache_manager import linear_att
    from lightllm.server.core.objs.start_args_type import StartArgs

    args = StartArgs(tp=4, dp=1, data_type="bfloat16", llm_kv_type=kv_type)
    monkeypatch.setattr(qwen, "get_env_start_args", lambda: args)
    model = object.__new__(qwen.Qwen3NextTpPartModel)
    model.args = args
    model.config = dict(
        n_layer=8,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=8,
        linear_key_head_dim=8,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        full_attention_interval=4,
    )
    model.max_req_num = 2
    model.tp_world_size_ = 4
    model.num_kv_heads = 1
    model.data_type = torch.bfloat16
    model.max_total_token_num = 32
    model.mem_fraction = 0.9
    events = []

    def runtime(size, config, args):
        events.append("runtime")
        return SimpleNamespace(linear_config=config, conv_state=object(), ssm_state=object())

    def memory(**kwargs):
        events.append("memory")
        assert events == ["runtime", "memory"]
        return SimpleNamespace(
            **kwargs, kv_buffer=object(), set_kv_layer_mapping=lambda mapping: events.append(mapping)
        )

    monkeypatch.setattr(linear_att, "get_added_mtp_kv_layer_num", lambda: 1)
    monkeypatch.setattr(qwen, "LinearStateManager", runtime)
    for name in (
        "MemoryManager",
        "FP8StaticPerHeadQuantMemManager",
        "FP8StaticPerTensorQuantMemManager",
    ):
        monkeypatch.setattr(qwen, name, memory)
    model._init_cache_coordinator()
    model._init_mem_manager()
    monkeypatch.setattr("lightllm.common.cache_coordinator.get_env_start_args", lambda: args)
    pools = []

    def checkpoint_pool(size, config, keep_num):
        pools.append((size, config, keep_num))
        return SimpleNamespace(size=size)

    monkeypatch.setattr(
        model.cache_coordinator.state_manager,
        "create_checkpoint_pool",
        lambda size, keep_num=0: checkpoint_pool(size, model.linear_config, keep_num),
        raising=False,
    )
    monkeypatch.setattr(model.cache_coordinator.state_manager, "create_cpu_transfer", lambda *args: None, raising=False)
    model.cache_coordinator.bind_mem_manager(model.mem_manager)
    pool = model.cache_coordinator.big_page_buffers
    model.cache_coordinator.bind_mem_manager(model.mem_manager)
    assert model.cache_coordinator.big_page_buffers is pool
    assert len(pools) == 1
    assert pools[0][1] is model.linear_config
    assert pools[0][2] == 2
    assert not hasattr(model.mem_manager, "big_page_buffers")
    assert not hasattr(model.mem_manager, "req_to_conv_state")
    config = model.linear_config
    assert config is model.cache_coordinator.linear_state_manager.linear_config
    assert events[-1] == config.get_full_att_kv_layer_mapping()
    assert config.num_linear_k_heads == 1
    assert config.num_linear_v_heads == 2
    assert config.full_att_dtype == (torch.bfloat16 if kv_type == "None" else torch.uint8)
    assert config.get_full_att_kv_layer_index(8) == 2


@pytest.mark.parametrize("mismatch", [None, "requests", "radix"])
def test_context_requires_shared_memory(monkeypatch, mismatch):
    from lightllm.server.router.model_infer import infer_batch

    monkeypatch.setattr(infer_batch, "get_env_start_args", lambda: SimpleNamespace(output_constraint_mode="none"))
    mem = SimpleNamespace()
    coordinator = HybridCacheCoordinator()
    coordinator.bind_mem_manager(mem)
    args = dict(
        backend=SimpleNamespace(),
        req_manager=SimpleNamespace(
            mem_manager=object() if mismatch == "requests" else mem, req_sampling_params_manager=None
        ),
        radix_cache=SimpleNamespace(mem_manager=object() if mismatch == "radix" else mem),
        shm_req_manager=None,
        vocab_size=1,
        cache_coordinator=coordinator,
    )
    context = infer_batch.InferenceContext()
    if mismatch:
        with pytest.raises(ValueError, match="share the memory manager"):
            context.register(**args)
    else:
        context.register(**args)
        assert context.cache_coordinator is coordinator
    args.pop("cache_coordinator")
    with pytest.raises(TypeError, match="cache_coordinator"):
        context.register(**args)


@pytest.mark.parametrize("pool", ["big_page_buffers", "small_page_buffers"])
def test_radix_cannot_use_a_different_checkpoint_pool(pool):
    coordinator = HybridCacheCoordinator(SimpleNamespace())
    coordinator.mem_manager = object()
    coordinator.big_page_buffers = object()
    coordinator.small_page_buffers = object()
    radix = SimpleNamespace(
        mem_manager=coordinator.mem_manager,
        big_page_buffers=coordinator.big_page_buffers,
        small_page_buffers=coordinator.small_page_buffers,
    )
    coordinator.validate_radix_cache(radix)
    setattr(radix, pool, object())
    with pytest.raises(ValueError, match="checkpoint pools"):
        coordinator.validate_radix_cache(radix)


def test_checkpoint_pool_cannot_be_replaced():
    coordinator = HybridCacheCoordinator()
    pool = coordinator.small_page_buffers = SimpleNamespace(size=8)
    assert coordinator.create_small_page_cache_manager(8) is pool
    with pytest.raises(ValueError, match="replace"):
        coordinator.create_small_page_cache_manager(16)


def test_full_attention_uses_the_coordinator_for_restore_and_release():
    from lightllm.common.cache_coordinator import CacheCoordinator

    coordinator = CacheCoordinator()
    assert coordinator.linear_state_manager is coordinator.indexer_state_manager is None
    node = SimpleNamespace(node_prefix_total_len=4)
    references = []
    radix = SimpleNamespace(
        match_prefix=lambda *args, **kwargs: (node, 4, torch.tensor([10, 11, 12, 13])),
        insert=lambda key, value: (6, node),
        dec_node_ref_counter=references.append,
    )
    reqs = SimpleNamespace(req_to_token_indexs=torch.arange(20, 28).reshape(1, 8))
    req = SimpleNamespace(
        req_idx=0,
        args=SimpleNamespace(page_size=2),
        cur_kv_len=0,
        hold_kv_len=0,
        shared_kv_node=None,
        mm_input_spans=[],
        sampling_param=SimpleNamespace(disable_prompt_cache=False),
        shm_req=SimpleNamespace(shm_prompt_ids=SimpleNamespace(arr=list(range(9)))),
        get_cur_total_len=lambda: 9,
        get_input_token_ids=lambda: list(range(9)),
    )
    coordinator.restore_prefix(req, reqs, radix)
    assert req.cur_kv_len == req.hold_kv_len == req.shm_req.prompt_cache_len == 4
    req.cur_kv_len, req.hold_kv_len = 7, 8
    released = coordinator.release_request_cache(req, reqs, radix, req.args, True)
    assert [part.tolist() for part in released] == [[26, 27], [24, 25]]
    assert references == [node]
    assert req.shared_kv_node is None


@pytest.mark.parametrize("local_prefix", [0, 8])
def test_peer_runtime_without_intermediate_checkpoints_is_not_published(local_prefix):
    from unittest.mock import Mock

    coordinator = HybridCacheCoordinator()
    coordinator.big_page_buffers = SimpleNamespace(free_state_cache=Mock())
    coordinator.small_page_buffers = SimpleNamespace(free_state_cache=Mock())
    shared = SimpleNamespace(node_prefix_total_len=local_prefix) if local_prefix else None
    req = SimpleNamespace(
        req_idx=0,
        cur_kv_len=28,
        hold_kv_len=28,
        shared_kv_node=shared,
        hybrid_cache_len=28,
        hybrid_len_to_big_page_id={24: 3},
        tail_small_page_buffer_id=7,
    )
    args = SimpleNamespace(linear_att_hash_page_size=4, linear_att_page_block_num=2)
    requests = SimpleNamespace(req_to_token_indexs=torch.arange(28).reshape(1, 28))
    radix = SimpleNamespace(insert=Mock(), dec_node_ref_counter=Mock())
    assert coordinator.get_cacheable_prefix_len(req, args) == local_prefix
    freed = coordinator.release_request_cache(req, requests, radix, args, True)
    assert torch.cat(freed).tolist() == list(range(local_prefix, 28))
    radix.insert.assert_not_called()
    coordinator.big_page_buffers.free_state_cache.assert_called_once_with([3])
    coordinator.small_page_buffers.free_state_cache.assert_called_once_with([7])
    assert not req.hybrid_len_to_big_page_id and req.tail_small_page_buffer_id is None


def test_release_keeps_only_the_checkpoint_complete_prefix():
    from unittest.mock import Mock

    coordinator = HybridCacheCoordinator()
    coordinator.big_page_buffers = SimpleNamespace(free_state_cache=Mock())
    coordinator.small_page_buffers = SimpleNamespace(free_state_cache=Mock())
    req = SimpleNamespace(
        req_idx=0,
        cur_kv_len=28,
        hold_kv_len=28,
        shared_kv_node=None,
        hybrid_cache_len=28,
        hybrid_len_to_big_page_id={8: 2, 24: 3},
        tail_small_page_buffer_id=7,
        get_input_token_ids=lambda: list(range(28)),
        shm_req=SimpleNamespace(hybrid_token_hash_list=SimpleNamespace(get_all=lambda: list(range(7)))),
    )
    args = SimpleNamespace(linear_att_hash_page_size=4, linear_att_page_block_num=2)
    requests = SimpleNamespace(req_to_token_indexs=torch.arange(28).reshape(1, 28))

    def insert(key, value, **kwargs):
        assert len(key) == len(value) == 8
        assert kwargs["len_to_big_page_id"] == {8: 2}
        kwargs["len_to_big_page_id"].clear()
        return 0, None

    assert coordinator.get_cacheable_prefix_len(req, args) == 8
    freed = coordinator.release_request_cache(req, requests, SimpleNamespace(insert=insert), args, True)
    assert torch.cat(freed).tolist() == list(range(8, 28))
    coordinator.big_page_buffers.free_state_cache.assert_called_once_with([3])
    coordinator.small_page_buffers.free_state_cache.assert_called_once_with([7])
