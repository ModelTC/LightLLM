"""GQA PD 直接搬运：原始位模式、物理页选择、TP 切片及有效 token 范围。"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
import torch

from lightllm.common.attention_state.full_att import FullAttentionConfig, GQAFullAttentionState
from lightllm.common.attention_state.types import StateTransferLayout

if TYPE_CHECKING:
    from easydict import EasyDict
    from multiprocessing.connection import Connection


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("token_major", [False, True])
@pytest.mark.parametrize("token_count", [1, 3])
def test_gqa_pd_direct_copy_between_tp_layouts(
    dtype: torch.dtype,
    token_major: bool,
    token_count: int,
    pd_args: EasyDict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pd_args.pd_kv_page_size = 4
    # P 保存完整 heads，D 只恢复 rank 1 的一半 heads；两侧外部页格式相同。
    source = GQAFullAttentionState(
        layout_config=FullAttentionConfig(
            logical_layer_groups=((2,), (5,)),
            global_head_num=4,
            head_dim=128,
            dtype=dtype,
        ),
        page_size=4,
        device="cuda",
    )
    target = GQAFullAttentionState(
        layout_config=replace(source.config, tp_world_size=2, tp_rank=1),
        page_size=4,
        device="cuda",
    )
    for state in (source, target):
        state.allocate_storage(token_capacity=8)
    for tensor in (source.k_storage, source.v_storage):
        tensor.view(torch.uint8).random_(0, 256)
    for tensor in (target.k_storage, target.v_storage):
        tensor.view(torch.uint8).fill_(173)
    layouts = []
    for state in (source, target):
        regions = []
        for region in state.get_pd_transfer_regions():
            axes = ("token", "layer", "head", "dim") if token_major else region.axes
            regions.append(replace(
                region,
                byte_start=region.byte_start + 16,
                byte_end=region.byte_end + 16,
                axes=axes,
                shape=tuple(region.shape[region.axes.index(axis)] for axis in axes),
            ))
        layouts.append(StateTransferLayout(
            state_name=state.state_name,
            tokens_per_page=4,
            byte_start=16,
            byte_end=max(region.byte_end for region in regions),
            regions=tuple(regions),
        ))
    layout, target_layout = layouts
    pages = torch.full(
        size=(3, layout.byte_end + 64),
        fill_value=91,
        dtype=torch.uint8,
        device="cuda",
    )
    page_view = pages[:, 32:-32]  # 页池视图的真实跨度大于可见宽度。
    src_ids = torch.tensor(data=[6, 9, 1, 9, 7, 9], dtype=torch.int32, device="cuda")[::2][:token_count]
    dst_ids = torch.tensor(data=[2, 6, 4], dtype=torch.int64, device="cuda")[:token_count]
    expected = [tensor[:, src_ids, 2:].contiguous().view(torch.uint8) for tensor in (source.k_storage, source.v_storage)]

    def reject_gather(*args: object, **kwargs: object) -> None:
        pytest.fail("PD transfer must not gather or scatter intermediate KV tensors")

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "index_select", reject_gather)
        patch.setattr(torch.Tensor, "index_copy_", reject_gather)
        patch.setattr(torch.Tensor, "tolist", reject_gather)
        with torch.cuda.stream(stream):
            source.write_to_pd_page(
                token_indices=src_ids,
                pd_page_buffers=page_view,
                pd_page_index=1,
                layout=layout,
            )
            target.read_from_pd_page(
                pd_page_buffers=page_view,
                token_indices=dst_ids,
                pd_page_index=1,
                layout=target_layout,
            )
        stream.synchronize()
    for tensor, expected_bytes in zip((target.k_storage, target.v_storage), expected):
        assert torch.equal(tensor[:, dst_ids].contiguous().view(torch.uint8), expected_bytes)
        assert (tensor[:, [index for index in range(12) if index not in [2, 6, 4][:token_count]]].contiguous().view(torch.uint8) == 173).all()
    assert (pages[[0, 2]] == 91).all()
    assert (pages[:, :48] == 91).all() and (pages[:, -32:] == 91).all()
    for region in layout.regions:
        external = page_view[1, region.byte_start:region.byte_end].view(dtype).view(region.shape)
        unused = [slice(None)] * 4
        unused[region.axes.index("token")] = slice(token_count, None)
        assert (external[tuple(unused)].contiguous().view(torch.uint8) == 91).all()

    # 反向验证分片写出：rank 1 只修改全局 heads [2, 4)，rank 0 区域保持原始字节。
    pages.fill_(91)
    target.write_to_pd_page(
        token_indices=dst_ids,
        pd_page_buffers=page_view,
        pd_page_index=2,
        layout=target_layout,
    )
    for region, expected_bytes in zip(target_layout.regions, expected):
        external = page_view[2, region.byte_start:region.byte_end].view(dtype).view(region.shape)
        external = external.permute(tuple(region.axes.index(axis) for axis in ("layer", "token", "head", "dim")))
        assert torch.equal(external[:, :token_count, 2:].contiguous().view(torch.uint8), expected_bytes)
        assert (external[:, :, :2].contiguous().view(torch.uint8) == 91).all()
        assert (external[:, token_count:].contiguous().view(torch.uint8) == 91).all()
    assert (pages[:2] == 91).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gqa_pd_head_slice_and_replica_reads(pd_args: EasyDict) -> None:
    pd_args.pd_kv_page_size = 4
    # MQA 的 KV head 在 TP ranks 间复制，非首 rank 禁止写出但仍能读回。
    state = GQAFullAttentionState(
        layout_config=FullAttentionConfig(
            logical_layer_groups=((0,), (2,)),
            global_head_num=1,
            head_dim=128,
            dtype=torch.bfloat16,
            tp_world_size=2,
            tp_rank=1,
        ),
        page_size=4,
        device="cuda",
    )
    state.allocate_storage(token_capacity=8)
    state.k_storage.fill_(19)
    state.v_storage.fill_(19)
    regions = state.get_pd_transfer_regions()
    layout = StateTransferLayout(
        state_name=state.state_name,
        tokens_per_page=4,
        byte_start=0,
        byte_end=max(region.byte_end for region in regions),
        regions=regions,
    )
    pages = torch.zeros(size=(2, layout.byte_end), dtype=torch.uint8, device="cuda")
    for region, value in zip(regions, (3, 7)):
        pages[1, region.byte_start:region.byte_end].view(region.dtype).fill_(value)
    snapshot = pages.clone()
    ids = torch.tensor(data=[5, 1], dtype=torch.int32, device="cuda")
    state.write_to_pd_page(
        token_indices=ids,
        pd_page_buffers=pages,
        pd_page_index=1,
        layout=layout,
    )
    state.read_from_pd_page(
        pd_page_buffers=pages,
        token_indices=ids,
        pd_page_index=1,
        layout=layout,
    )
    assert torch.equal(pages, snapshot)
    assert (state.k_storage[:, ids] == 3).all() and (state.v_storage[:, ids] == 7).all()
    assert (state.k_storage[:, [0, 2, 3, 4, 6, 7]] == 19).all()
    assert (state.v_storage[:, [0, 2, 3, 4, 6, 7]] == 19).all()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices required")
def test_gqa_pd_direct_copy_through_peer_ipc(pd_args: EasyDict, monkeypatch: pytest.MonkeyPatch) -> None:
    """使用生产的 IPC peer 映射，从传输 GPU 直接读写另一块 GPU 上的 K/V。"""
    from multiprocessing.reduction import ForkingPickler
    import torch.multiprocessing as mp

    from lightllm.server.router.model_infer.mode_backend.pd.p2p_fix import reduce_tensor

    if not torch.cuda.can_device_access_peer(0, 1):
        pytest.skip("CUDA peer access required")
    pd_args.pd_kv_page_size = 4
    state = GQAFullAttentionState(
        layout_config=FullAttentionConfig(
            logical_layer_groups=((0,), (2,)),
            global_head_num=2,
            head_dim=128,
            dtype=torch.bfloat16,
        ),
        page_size=4,
        device="cuda:1",
    )
    state.allocate_storage(token_capacity=8)
    state.k_storage.zero_()
    state.v_storage.zero_()
    state.k_storage[:, 1].fill_(3)
    state.v_storage[:, 1].fill_(7)
    torch.cuda.synchronize(device=1)
    regions = state.get_pd_transfer_regions()
    layout = StateTransferLayout(
        state_name=state.state_name,
        tokens_per_page=4,
        byte_start=0,
        byte_end=max(region.byte_end for region in regions),
        regions=regions,
    )
    monkeypatch.setattr(mp.reductions.reduce_tensor, "__code__", reduce_tensor.__code__)
    payload = bytes(ForkingPickler.dumps(state))
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_peer_pd_worker, args=(child, payload, layout))
    process.start()
    child.close()
    try:
        assert parent.poll(45), "PD peer worker did not finish"
        error = parent.recv()
        assert error is None, error
        assert (state.k_storage[:, 5] == 3).all() and (state.v_storage[:, 5] == 7).all()
        parent.send(None)
        process.join(10)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
        parent.close()


def _peer_pd_worker(connection: Connection, payload: bytes, layout: StateTransferLayout) -> None:
    from multiprocessing.reduction import ForkingPickler
    import traceback

    try:
        torch.cuda.set_device(device=0)
        state = ForkingPickler.loads(payload)
        pages = torch.zeros(size=(2, layout.byte_end), dtype=torch.uint8, device="cuda:0")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            state.write_to_pd_page(
                token_indices=torch.tensor(data=[1], dtype=torch.int32, device="cuda:0"),
                pd_page_buffers=pages,
                pd_page_index=1,
                layout=layout,
            )
            state.read_from_pd_page(
                pd_page_buffers=pages,
                token_indices=torch.tensor(data=[5], dtype=torch.int64, device="cuda:0"),
                pd_page_index=1,
                layout=layout,
            )
        stream.synchronize()
        assert not pages[0].any()
        connection.send(None)
        connection.recv()  # 保持导入的存储存活，直到父进程检查原 GPU 上的数据。
    except Exception:
        connection.send(traceback.format_exc())
    finally:
        connection.close()
