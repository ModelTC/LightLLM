"""V4 State 独立验证：packed 字段、逻辑快照及不同 MTP ring 的 PD 恢复。"""

from dataclasses import replace

import pytest
import torch

from lightllm.models.deepseek_v4.config import DSV4AttentionConfig
from lightllm.common.attention_state.full_att.compressed_att_state import (
    CompressedAttentionConfig,
    CompressedAttentionState,
)
from lightllm.common.attention_state.bounded_att import (
    BoundedAttentionState,
    FP8E8M0SlidingWindowConfig,
    FP8E8M0SlidingWindowState,
    WindowCompressorStateConfig,
    WindowCompressorState,
)
from lightllm.common.attention_state.types import StateTransferLayout


def make_config() -> DSV4AttentionConfig:
    return DSV4AttentionConfig(
        logical_layer_ids=(2, 5, 8),
        compress_ratios=(0, 4, 128),
        head_dim=512,
        rope_head_dim=64,
        indexer_head_dim=128,
        sliding_window=128,
    )


def make_full(ratio: int, device: str = "meta") -> CompressedAttentionState:
    config = make_config()
    layout = CompressedAttentionConfig(
        logical_layer_groups=((5 if ratio == 4 else 8,),),
        compress_ratio=ratio,
        kv_dim=config.head_dim,
        rope_dim=config.rope_head_dim,
        block_token_size=config.compressed_kv_token_alignment,
        indexer_dim=config.indexer_head_dim if ratio == 4 else 0,
    )
    return CompressedAttentionState(layout_config=layout, page_size=256, device=device)


def pd_layout(state: BoundedAttentionState | CompressedAttentionState) -> StateTransferLayout:
    regions = state.get_pd_transfer_regions()
    return StateTransferLayout(
        state_name=state.state_name,
        tokens_per_page=None if isinstance(state, BoundedAttentionState) else 256,
        byte_start=0,
        byte_end=regions[-1].byte_end,
        regions=regions,
    )


def gather_swa_reference(state: FP8E8M0SlidingWindowState, req_idx: int, length: int) -> torch.Tensor:
    """按格式的独立参考计算读回窗口，覆盖数据/scale 分区和环绕。"""
    config = state.config
    packed = state.swa_buffer[:, req_idx].cpu().reshape(len(config.logical_layer_ids), -1)
    result = torch.zeros((len(config.logical_layer_ids), config.window_size, config.kv_bytes), dtype=torch.uint8)
    for window, position in enumerate(range(length - config.window_size, length)):
        if position < 0:
            continue
        slot = position % state.swa_capacity_tokens
        page = slot // 128 * config.get_packed_kv_bytes(128)
        token = slot % 128
        data = page + token * config.kv_data_bytes
        scale = page + 128 * config.kv_data_bytes + token * (config.kv_bytes - config.kv_data_bytes)
        result[:, window, : config.kv_data_bytes] = packed[:, data : data + config.kv_data_bytes]
        result[:, window, config.kv_data_bytes :] = packed[:, scale : scale + config.kv_bytes - config.kv_data_bytes]
    return result


def test_compressed_meta_geometry() -> None:
    config = make_config()
    swa_config = config.create_swa_config()
    assert swa_config.kv_bytes == 584 and swa_config.get_packed_kv_bytes(128) // 128 == 585
    for ratio, slot_bytes in ((4, 180), (128, 9)):
        state = make_full(ratio=ratio)
        assert state.config.token_slot_bytes == slot_bytes
        assert state.storage is None
        state.allocate_storage(token_capacity=512)
        assert state.storage.shape == (1, 768, 1, slot_bytes)
        view = state.get_layer_view(logical_layer_id=5 if ratio == 4 else 8)
        assert view.kv.shape[0] == 3
        assert (view.indexer_k is not None) == (ratio == 4)
        attention_view = state.config.get_attention_view(packed_kv=view.kv)
        assert attention_view.shape == (3, 256 // ratio, 1, 584)
        assert attention_view.stride(0) == view.kv.stride(0)
        with pytest.raises(NotImplementedError, match="fused compressor"):
            state.write_full_att_kv(
                logical_layer_id=5 if ratio == 4 else 8,
                token_indices=torch.empty(0, dtype=torch.int64, device="meta"),
                kv=torch.empty(0, device="meta"),
            )


def test_compressed_factory_uses_independent_config() -> None:
    from lightllm.common.attention_state.full_att import create_compressed_att_state

    definition = make_full(ratio=4)
    assert type(definition.config).__bases__ == (object,)
    state = create_compressed_att_state(config=definition.config, page_size=256, device="meta")
    assert isinstance(state, CompressedAttentionState)
    assert state.storage is None and state.get_token_bytes() == 180
    assert state.get_state_type() == "compressed_c4_fp8_group64_e8m0"


@pytest.mark.parametrize("page_tokens", [0, 128, 384])
def test_compressed_transfer_pages_require_complete_blocks(page_tokens: int, monkeypatch: pytest.MonkeyPatch) -> None:
    from lightllm.utils.envs_utils import get_env_start_args

    args = get_env_start_args()
    monkeypatch.setattr(args, "cpu_cache_token_page_size", page_tokens)
    monkeypatch.setattr(args, "pd_kv_page_size", page_tokens)
    state = make_full(ratio=4)
    with pytest.raises(AssertionError, match="page token capacity"):
        state.get_cpu_transfer_regions()
    with pytest.raises(AssertionError, match="page token capacity"):
        state.get_pd_transfer_regions()


@pytest.mark.parametrize("write", [True, False])
@pytest.mark.parametrize("invalid", ["page_tokens", "page_bytes", "byte_start", "page_index"])
def test_compressed_cpu_page_boundaries(
    write: bool,
    invalid: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """错误的布局或物理页在提交 GPU kernel 前失败，避免跨页写入其他缓存数据。"""
    from lightllm.utils.envs_utils import get_env_start_args

    monkeypatch.setattr(get_env_start_args(), "cpu_cache_token_page_size", 256)
    state = make_full(ratio=4)
    regions = state.get_cpu_transfer_regions()
    layout = StateTransferLayout(
        state_name=state.state_name,
        tokens_per_page=256,
        byte_start=0,
        byte_end=regions[-1].byte_end,
        regions=regions,
    )
    pages = torch.empty((2, layout.byte_end), dtype=torch.uint8)
    page_indices = [0]
    match invalid:
        case "page_tokens":
            layout = replace(layout, tokens_per_page=128)
        case "page_bytes":
            pages = pages[:, :-1]
        case "byte_start":
            layout = replace(layout, byte_start=-1)
        case "page_index":
            page_indices = [2]
    # 只需索引数量，不创建 CUDA Tensor；所有非法输入必须在读取实际存储前被拦截。
    indices = torch.empty(256, dtype=torch.int32, device="meta")
    with pytest.raises(AssertionError, match="CPU page|CPU pages"):
        if write:
            state.write_to_cpu(
                token_indices=indices,
                cpu_page_buffers=pages,
                cpu_page_indices=page_indices,
                layout=layout,
            )
        else:
            state.read_from_cpu(
                cpu_page_buffers=pages,
                cpu_page_indices=page_indices,
                token_indices=indices,
                layout=layout,
            )


@pytest.mark.parametrize("ratio,block_tokens,indexer_dim,slot_bytes", [(4, 256, 0, 153), (8, 512, 32, 81)])
def test_atomic_compressed_config_controls_layout(
    ratio: int,
    block_tokens: int,
    indexer_dim: int,
    slot_bytes: int,
) -> None:
    """无需模型配置即可创建 State；indexer 是否存在独立于压缩比例。"""
    from lightllm.common.attention_state.full_att import create_compressed_att_state

    config = CompressedAttentionConfig(
        logical_layer_groups=((0, 4), (8,)),
        compress_ratio=ratio,
        kv_dim=512,
        rope_dim=64,
        block_token_size=block_tokens,
        indexer_dim=indexer_dim,
        tp_rank=1,
        tp_world_size=2,
    )
    state = create_compressed_att_state(config=config, page_size=block_tokens, device="meta")
    assert not hasattr(config, "cache_config")
    assert config.token_slot_bytes == slot_bytes
    assert state.get_token_bytes() == 2 * slot_bytes
    state.allocate_storage(token_capacity=2 * block_tokens)
    assert state.storage.shape == (2, 3 * block_tokens, 1, slot_bytes)
    assert state.get_storage_layer_index(0) == state.get_storage_layer_index(4)
    view = state.get_layer_view(logical_layer_id=4)
    assert view.kv.shape == (3, config.kv_block_bytes)
    if indexer_dim:
        assert view.indexer_k.shape == (3, block_tokens // ratio * (indexer_dim + 4))
    else:
        assert view.indexer_k is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("ratio", [4, 128])
def test_pd_compressed_kv_preserves_partial_block(ratio: int, monkeypatch: pytest.MonkeyPatch) -> None:
    from lightllm.utils.envs_utils import get_env_start_args

    monkeypatch.setattr(get_env_start_args(), "pd_kv_page_size", 256)
    source, target = make_full(ratio=ratio, device="cuda"), make_full(ratio=ratio, device="cuda")
    source.allocate_storage(token_capacity=512)
    target.allocate_storage(token_capacity=512)
    source.storage.random_(0, 256)
    target.storage.zero_()
    layout = pd_layout(source)
    pages = torch.zeros((1, layout.byte_end), dtype=torch.uint8, device="cuda")
    indices = torch.arange(17, dtype=torch.int32, device="cuda")
    source.write_to_pd_page(token_indices=indices, pd_page_buffers=pages, pd_page_index=0, layout=layout)
    target.read_from_pd_page(pd_page_buffers=pages, token_indices=indices, pd_page_index=0, layout=layout)
    torch.cuda.synchronize()
    assert torch.equal(source.storage[:, :256], target.storage[:, :256])
    assert not target.storage[:, 256:].any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("ratio", [4, 128])
def test_compressed_cpu_and_token_copy(ratio: int, monkeypatch: pytest.MonkeyPatch) -> None:
    from lightllm.utils.envs_utils import get_env_start_args

    monkeypatch.setattr(get_env_start_args(), "cpu_cache_token_page_size", 256)
    source, target = make_full(ratio=ratio, device="cuda"), make_full(ratio=ratio, device="cuda")
    source.allocate_storage(token_capacity=512)
    target.allocate_storage(token_capacity=768)
    source.storage.random_(0, 256)
    regions = source.get_cpu_transfer_regions()
    layout = StateTransferLayout(
        state_name=source.state_name,
        tokens_per_page=256,
        byte_start=0,
        byte_end=regions[-1].byte_end,
        regions=regions,
    )
    cpu_pages = torch.zeros((3, layout.byte_end), dtype=torch.uint8, pin_memory=True)
    src_indices = torch.arange(256, 512, dtype=torch.int32, device="cuda")
    dst_indices = torch.arange(256, dtype=torch.int64, device="cuda")
    source.write_to_cpu(
        token_indices=src_indices,
        cpu_page_buffers=cpu_pages,
        cpu_page_indices=[2],
        layout=layout,
    )
    target.read_from_cpu(
        cpu_page_buffers=cpu_pages,
        cpu_page_indices=[2],
        token_indices=dst_indices,
        layout=layout,
    )
    target.copy_tokens_from_dp(
        src_token_indices=src_indices,
        dst_token_indices=dst_indices + 512,
        remote_dp_full_att_state=source,
    )
    torch.cuda.synchronize()
    assert torch.equal(source.storage[:, 256:512], target.storage[:, :256])
    assert torch.equal(source.storage[:, 256:512], target.storage[:, 512:768])
    assert not target.storage[:, 256:512].any()
    assert not cpu_pages[:2].any()
    target.copy_full_att_tokens(src_indices=dst_indices, dst_indices=dst_indices + 256)
    torch.cuda.synchronize()
    assert torch.equal(source.storage[:, 256:512], target.storage[:, 256:512])
    assert not target.storage[:, 768:].any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize("token_count", [128, 257])
@pytest.mark.parametrize("from_dp", [False, True])
def test_compressed_token_copy_rejects_partial_blocks(ratio: int, token_count: int, from_dp: bool) -> None:
    """即使长度已按压缩比例对齐，也不能拆开含 KV/scale/indexer 的物理块。"""
    source, target = make_full(ratio=ratio, device="cuda"), make_full(ratio=ratio, device="cuda")
    source.allocate_storage(token_capacity=512)
    target.allocate_storage(token_capacity=512)
    indices = torch.arange(token_count, dtype=torch.int64, device="cuda")
    with pytest.raises(AssertionError, match="complete compressed blocks"):
        if from_dp:
            target.copy_tokens_from_dp(
                src_token_indices=indices,
                dst_token_indices=indices,
                remote_dp_full_att_state=source,
            )
        else:
            target.copy_full_att_tokens(src_indices=indices, dst_indices=indices)


def make_atomic_state(kind: str, mtp_step: int = 0, device: str = "meta") -> BoundedAttentionState:
    if kind == "swa":
        return FP8E8M0SlidingWindowState(
            config=FP8E8M0SlidingWindowConfig(logical_layer_ids=(2, 5), window_size=128, kv_dim=512, rope_dim=64),
            request_capacity=3,
            mtp_step=mtp_step,
            device=device,
        )
    window, compress_stride, candidate_dim, target = {
        "kv_w8": (8, 4, 1024, "kv"),
        "indexer_w8": (8, 4, 256, "indexer"),
        "kv_w128": (128, 128, 512, "kv"),
        "generic_w12": (12, 4, 24, "kv"),
    }[kind]
    return WindowCompressorState(
        config=WindowCompressorStateConfig(
            logical_layer_ids=(2, 5),
            window_size=window,
            compress_stride=compress_stride,
            candidate_dim=candidate_dim,
            target=target,
        ),
        request_capacity=3,
        mtp_step=mtp_step,
        device=device,
    )


def runtime_buffer(state: BoundedAttentionState) -> torch.Tensor:
    return state.swa_buffer if isinstance(state, FP8E8M0SlidingWindowState) else state.buffer


@pytest.mark.parametrize("kind", ["swa", "kv_w8", "indexer_w8", "kv_w128", "generic_w12"])
def test_atomic_window_definitions(kind: str) -> None:
    source, target = make_atomic_state(kind), make_atomic_state(kind, mtp_step=5)
    assert source.get_pd_transfer_regions() == target.get_pd_transfer_regions()
    cpu_regions = source.get_cpu_transfer_regions()
    copied_regions = source.get_cpu_transfer_regions()
    assert copied_regions == cpu_regions
    if cpu_regions:
        assert copied_regions is not cpu_regions and copied_regions[0] is not cpu_regions[0]
    assert source.get_layer_view(2).shape[0] == 3
    assert not source.contains_logical_layer(8)
    assert not hasattr(source.config, "compress_ratios")
    if isinstance(source, WindowCompressorState):
        assert source.config.overlap_tokens == source.config.window_size - source.config.compress_stride
        assert (source.get_cpu_transfer_regions() == ()) == (source.config.overlap_tokens == 0)
        source.initialize_page_pools(big_page_capacity=3, device="meta", pin_memory=False)
        assert (source.big_pages_pool is None) == (source.config.overlap_tokens == 0)
    else:
        assert source.swa_capacity_tokens == 128 and target.swa_capacity_tokens == 256
        for state in (source, target):
            attention_view = state.config.get_attention_view(packed_kv=state.get_layer_view(2))
            assert attention_view.shape == (3 * state.swa_capacity_tokens // 128, 128, 1, 584)
            assert attention_view.stride(0) == state.config.get_packed_kv_bytes(128)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", ["swa", "kv_w8", "indexer_w8", "kv_w128", "generic_w12"])
@pytest.mark.parametrize("length", [0, 1, 5, 127, 128, 129, 255, 260])
@pytest.mark.parametrize("source_step,target_step", [(0, 5), (5, 0)])
def test_atomic_pd_windows_ignore_mtp_capacity(kind: str, length: int, source_step: int, target_step: int) -> None:
    source = make_atomic_state(kind, mtp_step=source_step, device="cuda")
    target = make_atomic_state(kind, mtp_step=target_step, device="cuda")
    runtime_buffer(source).random_(0, 127)
    target.reset_bounded_att_state([1])
    layout = pd_layout(source)
    pages = torch.full((2, layout.byte_end + 32), 53, dtype=torch.uint8, device="cuda")
    source.write_to_pd_page(req_idx=0, pd_page_buffers=pages, pd_page_index=1, state_token_len=length, layout=layout)
    target.read_from_pd_page(pd_page_buffers=pages, req_idx=1, pd_page_index=1, state_token_len=length, layout=layout)
    torch.cuda.synchronize()
    if isinstance(source, FP8E8M0SlidingWindowState):
        assert torch.equal(gather_swa_reference(source, 0, length), gather_swa_reference(target, 1, length))
    else:
        src, dst = runtime_buffer(source), runtime_buffer(target)
        for position in range(max(0, length - source.config.window_size), length):
            assert torch.equal(src[:, 0, position % src.shape[2]], dst[:, 1, position % dst.shape[2]])
    assert (pages[0] == 53).all() and (pages[1, layout.byte_end :] == 53).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", ["swa", "kv_w8", "indexer_w8", "kv_w128", "generic_w12"])
def test_atomic_page_cpu_and_dp_roundtrip(kind: str) -> None:
    source = make_atomic_state(kind, mtp_step=3, device="cuda")
    target = make_atomic_state(kind, mtp_step=3, device="cuda")
    source.initialize_page_pools(big_page_capacity=3)
    target.initialize_page_pools(big_page_capacity=3)
    runtime_buffer(source).random_(0, 127)
    target.reset_bounded_att_state([0, 1])
    source.save_bounded_att_runtime_to_pages(
        req_indices=[0], state_token_lens=[512], page_kind="big", page_slot_ids=[0]
    )
    target.copy_runtime_state_from_dp(src_req_indices=[0], dst_req_indices=[1], remote_dp_bounded_state=source)
    regions = tuple(
        replace(region, byte_start=region.byte_start + 64, byte_end=region.byte_end + 64)
        for region in source.get_cpu_transfer_regions()
    )
    if not regions:
        assert source.big_pages_pool is None
        with pytest.raises(AssertionError, match="stride"):
            source.restore_bounded_att_runtime_from_pages([0], [513], "big", [0])
        return
    layout = StateTransferLayout(
        state_name=None, tokens_per_page=256, byte_start=64, byte_end=regions[-1].byte_end, regions=regions
    )
    pages = torch.full((3, layout.byte_end + 32), 53, dtype=torch.uint8, pin_memory=True)
    source.write_state_pages_to_cpu(
        page_kind="big",
        page_slot_ids=[0, 0],
        cpu_page_buffers=pages,
        cpu_page_indices=[1, 2],
        layout=layout,
        cpu_page_ready_flags=[True, False],
    )
    target.read_state_pages_from_cpu(
        cpu_page_buffers=pages, cpu_page_indices=[2], big_page_slot_ids=[0], layout=layout
    )
    target.copy_big_pages_from_dp(src_page_ids=[0], dst_page_ids=[2], remote_dp_bounded_state=source)
    target.restore_bounded_att_runtime_from_pages(
        req_indices=[0], state_token_lens=[512], page_kind="big", page_slot_ids=[0]
    )
    torch.cuda.synchronize()
    assert torch.equal(source.big_pages_pool.buffer[0], target.big_pages_pool.buffer[0])
    assert torch.equal(source.big_pages_pool.buffer[0], target.big_pages_pool.buffer[2])
    assert torch.equal(runtime_buffer(source)[:, 0], runtime_buffer(target)[:, 1])
    assert (pages[:2] == 53).all() and (pages[2, :64] == 53).all() and (pages[2, layout.byte_end :] == 53).all()
    if isinstance(source, FP8E8M0SlidingWindowState):
        assert torch.equal(gather_swa_reference(source, 0, 512), gather_swa_reference(target, 0, 512))
    else:
        for position in range(512 - source.config.overlap_tokens, 512):
            ring = source.buffer.shape[2]
            assert torch.equal(source.buffer[:, 0, position % ring], target.buffer[:, 0, position % ring])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_manager_joint_cpu_page_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    from lightllm.common.attention_state import AttentionStateManager
    from lightllm.utils.envs_utils import get_env_start_args

    args = get_env_start_args()
    monkeypatch.setattr(args, "cpu_cache_token_page_size", 256)
    monkeypatch.setattr(args, "pd_kv_page_size", 256)
    manager = AttentionStateManager(request_capacity=3)
    full_states = [make_full(ratio=ratio) for ratio in (4, 128)]
    bounded = [
        make_atomic_state(kind, mtp_step=2, device="cuda") for kind in ("swa", "kv_w8", "indexer_w8", "kv_w128")
    ]
    for state in (*full_states, *bounded):
        manager.register_state(model_role="main", state=state)
    manager.finish_state_registration()
    manager.allocate_storage(token_capacity=512, page_size=256, big_page_tokens=256, device="cuda")
    for full in full_states:
        full.storage[:, :256].random_(0, 256)
    for state in bounded:
        runtime_buffer(state).random_(0, 127)
    manager.save_bounded_att_runtime_to_pages([0], [256], "big", [0])
    expected = gather_swa_reference(bounded[0], 0, 256)
    layouts = manager.get_cpu_page_layouts()
    assert bounded[-1].state_name not in layouts
    pages = torch.zeros((2, manager.get_cpu_page_bytes()), dtype=torch.uint8, pin_memory=True)
    manager.write_req_to_cpu(
        token_indices=torch.arange(256, dtype=torch.int64, device="cuda"),
        cpu_page_buffers=pages,
        cpu_page_indices=[1],
        layouts=layouts,
        big_page_slot_ids=[0],
    )
    manager.read_req_from_cpu(
        cpu_page_buffers=pages,
        cpu_page_indices=[1],
        token_indices=torch.arange(256, 512, dtype=torch.int64, device="cuda"),
        layouts=layouts,
        big_page_slot_ids=[1],
        req_idx=1,
        state_token_len=256,
    )
    torch.cuda.synchronize()
    for full in full_states:
        assert torch.equal(full.storage[:, :256], full.storage[:, 256:512])
    assert torch.equal(expected, gather_swa_reference(bounded[0], 1, 256))
    assert not pages[0].any()
    ipc = manager.create_pd_ipc_copy()
    assert all(state.big_pages_pool is None for state in ipc.get_bounded_attention_states())
    assert bounded[0].big_pages_pool is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_atomic_dp_ipc_preserves_shared_pools(monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    from lightllm.common.attention_state import AttentionStateManager
    from lightllm.utils.envs_utils import get_env_start_args

    monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps({"enable_dp_prompt_cache_fetch": True}))
    get_env_start_args.cache_clear()
    try:
        manager = AttentionStateManager(request_capacity=3)
        for kind in ("swa", "kv_w8", "kv_w128"):
            state = make_atomic_state(kind, device="cuda")
            state.initialize_page_pools(big_page_capacity=2, small_page_capacity=1)
            manager.register_state("main", state)
        manager.finish_state_registration()
        exported = manager.create_dp_ipc_copy()
        for state in exported.get_bounded_attention_states():
            for pool in (state.big_pages_pool, state.small_pages_pool):
                if pool is not None:
                    assert pool.buffer.is_shared()
        exported.register_page_pools_pin_memory()
    finally:
        get_env_start_args.cache_clear()
