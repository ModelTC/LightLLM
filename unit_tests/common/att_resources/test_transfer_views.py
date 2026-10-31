"""传输视图整理必须保留存储别名、位模式、未参与区域和独立寻址轴。"""

import pytest
import torch

from lightllm.common.attention_state.utils import merge_contiguous_tail_dims, view_max_copy_dtype


@pytest.mark.parametrize(
    "layout, expected_shape, start_dim",
    [
        ("dense", (48,), 0),
        ("source_gap", (2, 3, 8), 0),
        ("destination_gap", (2, 3, 8), 0),
        ("strided_tail", (48,), 0),
        ("singleton_gap", (2, 8), 0),
        ("protected_window", (2, 5, 24), 2),
    ],
)
def test_merge_tail_preserves_copy_regions(
    layout: str, expected_shape: tuple[int, ...], start_dim: int,
) -> None:
    source_shape = (2, 5, 3, 8) if layout == "protected_window" else (2, 3, 8)
    destination_shape = source_shape
    if layout == "source_gap":
        source_shape = (2, 3, 10)
    elif layout == "destination_gap":
        destination_shape = (2, 3, 10)
    elif layout == "strided_tail":
        source_shape = (2, 3, 16)
    elif layout == "singleton_gap":
        source_shape, destination_shape = (2, 1, 10), (2, 1, 12)

    source_storage = torch.arange(torch.Size(source_shape).numel(), dtype=torch.float32).view(source_shape)
    destination_storage = torch.full(size=destination_shape, fill_value=-1, dtype=torch.float32)
    source = source_storage[..., ::2] if layout == "strided_tail" else source_storage[..., :8]
    destination = destination_storage[..., :8]
    # 直接按原始视图执行参考复制，保留目标 padding，随后比较整个底层存储。
    expected = destination_storage.clone()
    expected.as_strided(
        size=destination.shape,
        stride=destination.stride(),
        storage_offset=destination.storage_offset(),
    ).copy_(source)

    copies = [(source, destination)]
    merged = merge_contiguous_tail_dims(copies=copies, start_dim=start_dim)
    source_view, destination_view = merged[0]
    assert source_view.shape == destination_view.shape == expected_shape
    assert source_view.data_ptr() == source.data_ptr()
    assert destination_view.data_ptr() == destination.data_ptr()
    if expected_shape == source.shape:
        assert merged is copies
    destination_view.copy_(source_view)
    assert torch.equal(destination_storage, expected)


def test_merge_request_slices_preserves_batch_offsets() -> None:
    source = torch.arange(2 * 4 * 3 * 8, dtype=torch.float32).view(2, 4, 3, 8)
    destination = torch.full(size=(4, 2, 3, 8), fill_value=-1, dtype=torch.float32)
    expected = destination.clone()
    copies = []
    for req_idx in (0, 2):
        copies.append((source[:, req_idx], destination[req_idx]))
        expected[req_idx].copy_(source[:, req_idx])

    for source_view, destination_view in merge_contiguous_tail_dims(copies=copies):
        assert source_view.shape == destination_view.shape == (2, 24)
        destination_view.copy_(source_view)
    assert torch.equal(destination, expected)


@pytest.mark.parametrize("destination_prefix", [(2, 11), (3, 2, 5)])
@pytest.mark.parametrize("keepdim", [False, True])
def test_merge_tail_preserves_different_prefix_layouts(destination_prefix: tuple[int, ...], keepdim: bool) -> None:
    source = torch.arange(2 * 7 * 4 * 2, dtype=torch.float32).view(2, 7, 4, 2)
    destination_shape = destination_prefix + (4, 2)
    destination = torch.arange(torch.Size(destination_shape).numel(), dtype=torch.float32).view(destination_shape)
    source_view, destination_view = merge_contiguous_tail_dims(
        copies=[(source, destination)],
        start_dim=-2,
        keepdim=keepdim,
    )[0]
    tail_shape = (1, 8) if keepdim else (8,)
    assert source_view.shape == (2, 7) + tail_shape
    assert destination_view.shape == destination_prefix + tail_shape
    assert source_view.data_ptr() == source.data_ptr()
    assert destination_view.data_ptr() == destination.data_ptr()
    assert torch.equal(source_view.flatten(), source.flatten())
    assert torch.equal(destination_view.flatten(), destination.flatten())


@pytest.mark.parametrize("gap_side", ["source", "destination"])
def test_merge_tail_keeps_head_padding(gap_side: str) -> None:
    source = torch.empty(size=(2, 7, 4, 3 if gap_side == "source" else 2))[..., :2]
    destination = torch.empty(size=(3, 2, 5, 4, 3 if gap_side == "destination" else 2))[..., :2]
    copies = [(source, destination)]
    assert merge_contiguous_tail_dims(copies=copies, start_dim=-2, keepdim=True) is copies


@pytest.mark.parametrize("torch_version", ["2.9.0", "2.10.0", "2.12.0", "3.0.0"])
def test_merge_tail_rejects_unverified_torch_version(torch_version: str, monkeypatch: pytest.MonkeyPatch) -> None:
    source = torch.empty(size=(2, 4))
    destination = torch.empty_like(source)
    monkeypatch.setattr(torch, "__version__", torch_version)
    with pytest.raises(AssertionError, match="supports PyTorch 2.11.x"):
        merge_contiguous_tail_dims(copies=[(source, destination)])


@pytest.mark.parametrize("torch_version", ["2.11.0+cu130", "2.11.1"])
def test_merge_tail_accepts_supported_torch_version(torch_version: str, monkeypatch: pytest.MonkeyPatch) -> None:
    source = torch.empty(size=(2, 4))
    destination = torch.empty_like(source)
    monkeypatch.setattr(torch, "__version__", torch_version)
    source_view, destination_view = merge_contiguous_tail_dims(copies=[(source, destination)])[0]
    assert source_view.shape == destination_view.shape == (8,)


@pytest.mark.parametrize(
    "dtype, width, src_offset, dst_offset, expected_dtype",
    [
        (torch.bfloat16, 8, 0, 0, torch.int64),
        (torch.bfloat16, 6, 0, 0, torch.int32),
        (torch.int8, 2, 0, 0, torch.int16),
        (torch.float32, 3, 0, 0, torch.float32),
        (torch.int64, 8, 0, 0, torch.int64),
        (torch.bfloat16, 8, 1, 0, torch.bfloat16),
        (torch.bfloat16, 8, 0, 1, torch.bfloat16),
        (torch.bfloat16, 8, 2, 0, torch.int32),
    ],
)
def test_max_copy_dtype_preserves_data_and_aliases(
    dtype: torch.dtype,
    width: int,
    src_offset: int,
    dst_offset: int,
    expected_dtype: torch.dtype,
) -> None:
    source = torch.arange(32).to(dtype=dtype)
    destination = torch.full(size=(32,), fill_value=-1, dtype=dtype)
    expected = destination.clone()
    source_view = source[src_offset:src_offset + width]
    destination_view = destination[dst_offset:dst_offset + width]
    expected[dst_offset:dst_offset + width].copy_(source_view)

    copies = [(source_view, destination_view)]
    converted = view_max_copy_dtype(copies=copies)
    copy_source, copy_destination = converted[0]
    assert copy_source.dtype == copy_destination.dtype == expected_dtype
    assert copy_source.data_ptr() == source_view.data_ptr()
    assert copy_destination.data_ptr() == destination_view.data_ptr()
    assert copy_source.ndim == source_view.ndim
    if expected_dtype == dtype:
        assert converted is copies
    copy_destination.copy_(copy_source)
    assert torch.equal(destination, expected)


@pytest.mark.parametrize(
    "src_padding, dst_padding, expected_dtype",
    [(1, 0, torch.bfloat16), (0, 2, torch.int32), (0, 0, torch.int64)],
)
def test_max_copy_dtype_preserves_row_padding(
    src_padding: int, dst_padding: int, expected_dtype: torch.dtype,
) -> None:
    source = torch.arange(2 * (8 + src_padding)).to(dtype=torch.bfloat16).view(2, 8 + src_padding)
    destination = torch.full(size=(2, 8 + dst_padding), fill_value=-1, dtype=torch.bfloat16)
    expected = destination.clone()
    expected[:, :8].copy_(source[:, :8])

    source_view, destination_view = view_max_copy_dtype(copies=[(source[:, :8], destination[:, :8])])[0]
    assert source_view.dtype == destination_view.dtype == expected_dtype
    assert source_view.shape[0] == destination_view.shape[0] == 2
    destination_view.copy_(source_view)
    assert torch.equal(destination, expected)


def test_max_copy_dtype_falls_back_for_middle_batch_offset() -> None:
    source = torch.arange(64).to(dtype=torch.bfloat16)
    destination = torch.full(size=(64,), fill_value=-1, dtype=torch.bfloat16)
    copies = [(source[start:start + 8], destination[end:end + 8]) for start, end in ((0, 0), (9, 16), (24, 32))]
    assert view_max_copy_dtype(copies=copies) is copies


def test_max_copy_dtype_keeps_strided_last_axis() -> None:
    source = torch.arange(32, dtype=torch.float32)[::2]
    destination = torch.empty(size=(16,), dtype=torch.float32)
    copies = [(source, destination)]
    assert view_max_copy_dtype(copies=copies) is copies
