"""各 full-attention 编码的统一 CPU/PD kernel：无中间 KV、跳步循环与字段完整性。"""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from importlib import import_module
from typing import TYPE_CHECKING

import pytest
import torch

from lightllm.common.att_resources.full_att import StaticFP8FullAttentionState
from lightllm.common.att_resources.types import StateTransferLayout
from unit_tests.common.att_resources.model_helpers import full_state_test_buffers, full_state_test_fields, rebuild_full_state
from unit_tests.common.att_resources.test_capacity import definition

if TYPE_CHECKING:
    from easydict import EasyDict
    from multiprocessing.connection import Connection


SCHEMES = {
    "int4": "group_quantized",
    "int8": "group_quantized",
    "static_head": "static_fp8",
    "static_tensor": "static_fp8",
    "mla": "mla",
    "packed": "packed_mla",
    "packed_fp8": "packed_mla",
    "packed_glm": "packed_mla",
}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", list(SCHEMES))
@pytest.mark.parametrize("transfer", ["cpu", "pd"])
@pytest.mark.parametrize("grid_num", [1, 36])
@pytest.mark.parametrize("token_count", [1, 513])
def test_full_att_pages_copy_all_fields_with_bounded_grid(
    kind: str,
    transfer: str,
    grid_num: int,
    token_count: int,
    pd_args: EasyDict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page_tokens = 128 if transfer == "cpu" else 1024
    pd_args.cpu_cache_token_page_size = pd_args.pd_kv_page_size = page_tokens
    template = definition(kind)
    source = rebuild_full_state(template, token_capacity=1024, device="cuda")
    target = rebuild_full_state(template, token_capacity=1024, device="cuda")
    source_buffers = full_state_test_buffers(source)
    target_buffers = full_state_test_buffers(target)
    for buffer in source_buffers.values():
        buffer.view(torch.uint8).random_(0, 256)
    for buffer in target_buffers.values():
        buffer.view(torch.uint8).fill_(173)
    calibration = []
    if isinstance(source, StaticFP8FullAttentionState):
        for state in (source, target):
            for tensor in (state.scales, state.q_scales):
                if tensor is not None:
                    calibration.append((tensor, tensor.clone()))

    regions = source.get_cpu_transfer_regions() if transfer == "cpu" else source.get_pd_transfer_regions()
    if grid_num == 1:
        # 调换层/token 轴，验证寻址使用真实 stride；region 顺序也不要求与存储成员相同。
        regions = tuple(replace(
            region,
            axes=("token", "layer", *region.axes[2:]),
            shape=(region.shape[1], region.shape[0], *region.shape[2:]),
        ) for region in reversed(regions))
    layout = StateTransferLayout(
        resource_id=source.resource_id,
        tokens_per_page=page_tokens,
        byte_start=0,
        byte_end=max(region.byte_end for region in regions),
        regions=regions,
    )
    page_count = (token_count + page_tokens - 1) // page_tokens
    page_indices = [3, 0, 4, 1, 6][:page_count]
    pages = torch.full(
        size=(8, layout.byte_end + 32),
        fill_value=91,
        dtype=torch.uint8,
        device="cpu" if transfer == "cpu" else "cuda",
        pin_memory=transfer == "cpu",
    )
    page_view = pages[:, 16:-16]
    src_ids = torch.randperm(n=1024, device="cuda")[:token_count].to(dtype=torch.int32)
    dst_order = torch.randperm(n=1024, device="cuda")
    dst_ids = dst_order[:token_count]
    expected = {name: tensor[:, dst_ids].clone() for name, tensor in target_buffers.items()}
    source_fields = full_state_test_fields(source, source_buffers)
    expected_fields = full_state_test_fields(target, expected)
    for name, field in source_fields.items():
        expected_fields[name].tensor.copy_(other=field.tensor[:, src_ids])

    # 直接覆盖 State 调用的入口参数，读写仍通过生产 State 接口提交。
    scheme = SCHEMES[kind]
    state_module = import_module(name=type(source).__module__)
    transfer_name = f"transfer_{scheme}_pages"
    monkeypatch.setattr(state_module, transfer_name, partial(getattr(state_module, transfer_name), grid_num=grid_num))

    def reject_gather(*args: object, **kwargs: object) -> None:
        pytest.fail("page transfers must not gather/scatter KV or copy token indices to host")

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "index_select", reject_gather)
        patch.setattr(torch.Tensor, "index_copy_", reject_gather)
        patch.setattr(torch.Tensor, "tolist", reject_gather)
        with torch.cuda.stream(stream):
            if transfer == "cpu":
                source.write_to_cpu(token_indices=src_ids, cpu_page_buffers=page_view, cpu_page_indices=page_indices, layout=layout)
                target.read_from_cpu(cpu_page_buffers=page_view, cpu_page_indices=page_indices, token_indices=dst_ids, layout=layout)
            else:
                source.write_to_pd_buffer(token_indices=src_ids, pd_page_buffers=page_view, pd_page_index=3, layout=layout)
                target.read_from_pd_buffer(pd_page_buffers=page_view, token_indices=dst_ids, pd_page_index=3, layout=layout)
        stream.synchronize()

    # 比较整行原始字节：packed padding 沿用目标填充值，字段数据来自源存储。
    for name, tensor in target_buffers.items():
        assert torch.equal(tensor[:, dst_ids].contiguous().view(torch.uint8), expected[name].view(torch.uint8))
        assert (tensor[:, dst_order[token_count:]].contiguous().view(torch.uint8) == 173).all()
        assert (tensor[:, 1024:].view(torch.uint8) == 173).all()
    for tensor, before in calibration:
        assert torch.equal(tensor, before)
    assert (pages[[index for index in range(8) if index not in page_indices]] == 91).all()
    assert (pages[:, :16] == 91).all() and (pages[:, -16:] == 91).all()
    last_count = token_count - (page_count - 1) * page_tokens
    for region in regions:
        last_page = page_view[page_indices[-1], region.byte_start:region.byte_end].view(region.dtype).view(region.shape)
        unused = [slice(None)] * 4
        unused[region.axes.index("token")] = slice(last_count, None)
        assert (last_page[tuple(unused)].contiguous().view(torch.uint8) == 91).all()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices required")
def test_all_full_att_schemes_transfer_through_peer_ipc(
    pd_args: EasyDict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """各编码字段通过生产 IPC peer 映射跨 GPU 搬运，padding 保留目标值。"""
    from multiprocessing.reduction import ForkingPickler
    import torch.multiprocessing as mp

    from lightllm.server.router.model_infer.mode_backend.pd.p2p_fix import reduce_tensor

    if not torch.cuda.can_device_access_peer(0, 1):
        pytest.skip("CUDA peer access required")
    pd_args.pd_kv_page_size = 4
    states, layouts, expected_buffers = [], [], []
    for kind in SCHEMES:
        state = rebuild_full_state(definition(kind), token_capacity=8, device="cuda:1")
        buffers = full_state_test_buffers(state)
        for tensor in buffers.values():
            tensor.view(torch.uint8).random_(0, 256)
            tensor[:, 5].view(torch.uint8).fill_(173)
        expected = {name: tensor[:, 5:6].clone() for name, tensor in buffers.items()}
        source_fields = full_state_test_fields(state, buffers)
        expected_fields = full_state_test_fields(state, expected)
        for name, field in source_fields.items():
            expected_fields[name].tensor.copy_(other=field.tensor[:, 1:2])
        regions = state.get_pd_transfer_regions()
        layouts.append(StateTransferLayout(
            resource_id=state.resource_id,
            tokens_per_page=4,
            byte_start=0,
            byte_end=max(region.byte_end for region in regions),
            regions=regions,
        ))
        states.append(state)
        expected_buffers.append(expected)
    torch.cuda.synchronize(device=1)
    monkeypatch.setattr(mp.reductions.reduce_tensor, "__code__", reduce_tensor.__code__)
    payload = bytes(ForkingPickler.dumps(states))
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_peer_transfer_worker, args=(child, payload, layouts))
    process.start()
    child.close()
    try:
        assert parent.poll(60), "PD peer worker did not finish"
        error = parent.recv()
        assert error is None, error
        for state, expected in zip(states, expected_buffers):
            for name, tensor in full_state_test_buffers(state).items():
                assert torch.equal(tensor[:, 5:6].contiguous().view(torch.uint8), expected[name].view(torch.uint8))
        parent.send(None)
        process.join(10)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
        parent.close()


def _peer_transfer_worker(
    connection: Connection, payload: bytes, layouts: list[StateTransferLayout],
) -> None:
    from multiprocessing.reduction import ForkingPickler
    import traceback

    try:
        torch.cuda.set_device(device=0)
        states = ForkingPickler.loads(payload)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            source_ids = torch.tensor(data=[1], dtype=torch.int32, device="cuda:0")
            target_ids = torch.tensor(data=[5], dtype=torch.int64, device="cuda:0")
            for state, layout in zip(states, layouts):
                pages = torch.empty(size=(2, layout.byte_end), dtype=torch.uint8, device="cuda:0")
                state.write_to_pd_buffer(
                    token_indices=source_ids,
                    pd_page_buffers=pages,
                    pd_page_index=1,
                    layout=layout,
                )
                state.read_from_pd_buffer(
                    pd_page_buffers=pages,
                    token_indices=target_ids,
                    pd_page_index=1,
                    layout=layout,
                )
        stream.synchronize()
        connection.send(None)
        connection.recv()
    except Exception:
        connection.send(traceback.format_exc())
    finally:
        connection.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", ["gqa", *SCHEMES])
@pytest.mark.parametrize("readies", [[False, True, False], [True, False, True], [True, True, True], [False, False, False]])
@pytest.mark.parametrize("grid_num", [1, 36])
def test_cpu_ready_pages_preserve_fields_and_logical_token_positions(
    kind: str, readies: list[bool], grid_num: int, pd_args: EasyDict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pd_args.cpu_cache_token_page_size = 5
    state = rebuild_full_state(definition(kind), token_capacity=32, device="cuda")
    # 单 program 验证跳过 ready 块后仍会处理后续未就绪块；默认 grid 保持生产配置。
    state_module = import_module(name=type(state).__module__)
    scheme = "gqa" if kind == "gqa" else SCHEMES[kind]
    transfer_name = f"transfer_{scheme}_pages"
    monkeypatch.setattr(state_module, transfer_name, partial(getattr(state_module, transfer_name), grid_num=grid_num))
    for buffer in full_state_test_buffers(state).values():
        buffer.view(torch.uint8).random_(0, 256)
    regions = state.get_cpu_transfer_regions()
    layout = StateTransferLayout(
        resource_id=state.resource_id, tokens_per_page=5, byte_start=0,
        byte_end=max(region.byte_end for region in regions), regions=regions,
    )
    # 页编号不连续，token 索引带 stride；最后一页只写三个 token。
    page_indices = [3, 0, 4]
    token_indices = torch.arange(26, dtype=torch.int32, device="cuda")[::2]
    baseline = torch.full((6, layout.byte_end + 32), 173, dtype=torch.uint8, pin_memory=True)
    pages = torch.full_like(baseline, 173, pin_memory=True)
    state.write_to_cpu(token_indices, baseline[:, 16:-16], page_indices, layout=layout)
    torch.cuda.synchronize()
    expected = pages.clone()
    for page, ready in zip(page_indices, readies):
        if not ready:
            expected[page].copy_(baseline[page])

    def reject_allocation(*args: object, **kwargs: object) -> None:
        pytest.fail("all-ready writes must not allocate index or ready tensors")

    with monkeypatch.context() as patch:
        if all(readies):
            patch.setattr(torch, "tensor", reject_allocation)
        state.write_to_cpu(
            token_indices, pages[:, 16:-16], page_indices, layout=layout, cpu_page_readies=readies,
        )
    torch.cuda.synchronize()
    # 整页比较同时覆盖 KV、量化 scale、packed indexer 字段及前后 padding。
    assert torch.equal(pages, expected)
