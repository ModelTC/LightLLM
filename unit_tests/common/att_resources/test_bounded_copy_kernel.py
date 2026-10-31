"""聚合复制必须保持位模式、stride、流顺序和未参与区域。"""

import pytest
import torch
import triton.language as tl

from lightllm.common.attention_state.bounded_att.copy_kernel import copy_bounded_att_tensors
from lightllm.common.attention_state.bounded_att.utils import register_page_pool_tensor


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("direction", ["cuda_to_host", "host_to_cuda", "host_to_host", "cuda_to_cuda"])
@pytest.mark.parametrize("grid_num", [1, 5, 16])
@pytest.mark.parametrize("dtype", [torch.int8, torch.bfloat16, torch.float32, torch.float16, torch.int64])
def test_aggregate_same_dtype_and_strides(direction: str, grid_num: int, dtype: torch.dtype) -> None:
    src_device = "cuda" if direction.startswith("cuda") else "cpu"
    dst_device = "cuda" if direction.endswith("cuda") else "cpu"
    sources, targets, copies, expected_targets = [], [], [], []
    # Conv 窗口裁剪和 SSM 轴交换均保留间隙；使用非 2 次幂的边界覆盖最后一个 block。
    # 同批视图保持相同 shape，验证 grid 跳步和跨视图 block 分配。
    for _ in range(41):
        shape = (3, 5, 7, 131)
        source = torch.empty(
            size=shape,
            dtype=dtype,
            device=src_device,
            pin_memory=src_device == "cpu",
        )
        source.copy_(torch.arange(source.numel()).reshape(shape) % 127)
        target_shape = (shape[1], shape[0], shape[2] + 2, shape[3] + 4)
        target = torch.full(
            size=target_shape,
            fill_value=-17,
            dtype=dtype,
            device=dst_device,
            pin_memory=dst_device == "cpu",
        )
        source_view = source[..., :shape[3] - 1]
        target_view = target.permute(1, 0, 2, 3)[:, :, 1:shape[2] + 1, 2:shape[3] + 1]
        expected = target.cpu().clone()
        expected.permute(1, 0, 2, 3)[:, :, 1:shape[2] + 1, 2:shape[3] + 1].copy_(source_view.cpu())
        sources.append(source)
        targets.append(target)
        copies.append((source_view, target_view))
        expected_targets.append(expected)
    copy_bounded_att_tensors(copies=copies, grid_num=grid_num)
    torch.cuda.current_stream().synchronize()
    for target, expected in zip(targets, expected_targets):
        assert torch.equal(target.cpu(), expected)


def test_registered_shared_pages_and_stream_dependency() -> None:
    source = torch.empty(size=(4099,), dtype=torch.float32).share_memory_()
    target = torch.empty(size=(4099,), dtype=torch.float32).share_memory_()
    register_page_pool_tensor(tensor=source)
    register_page_pool_tensor(tensor=target)
    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    values = torch.arange(source.numel(), dtype=source.dtype, device="cuda")
    with torch.cuda.stream(producer):
        copy_bounded_att_tensors(copies=[(values, source)])
    with torch.cuda.stream(consumer):
        consumer.wait_stream(producer)
        # 这里不经过主机 synchronize；CPU→CPU 传输必须在 GPU stream 上等待上游。
        copy_bounded_att_tensors(copies=[(source, target)])
        restored = torch.empty_like(values)
        copy_bounded_att_tensors(copies=[(target, restored)])
    consumer.synchronize()
    assert torch.equal(restored, values)
    assert torch.equal(target, values.cpu())


@pytest.mark.parametrize("float_dtype,int_dtype,patterns", [
    (torch.float16, torch.int16, [0x7E01, -32768, 0x7C00, 1]),
    (torch.bfloat16, torch.int16, [0x7FC1, -32768, 0x7F80, 1]),
    (torch.float32, torch.int32, [0x7FC00001, -2147483648, 0x7F800000, 1]),
    (torch.float64, torch.int64, [0x7FF8000000000001, -9223372036854775808, 0x7FF0000000000000, 1]),
])
def test_exact_float_bit_patterns(float_dtype: torch.dtype, int_dtype: torch.dtype, patterns: list[int]) -> None:
    # 包含 NaN payload、负零和无穷，等宽整数复制不能触发精度转换或规范化。
    bits = torch.tensor(data=patterns, dtype=int_dtype, device="cuda")
    target = torch.empty(size=(4,), dtype=float_dtype, pin_memory=True)
    copy_bounded_att_tensors(copies=[(bits.view(float_dtype), target)])
    torch.cuda.synchronize()
    assert torch.equal(target.view(int_dtype), bits.cpu())


def test_one_kernel_for_same_dimension_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    from lightllm.common.attention_state.bounded_att import copy_kernel

    actual_kernel = copy_kernel._copy_bounded_att_1d
    launches = []

    class CountLaunches:
        def __getitem__(self, grid: tuple[int]):
            launches.append(grid)
            return actual_kernel[grid]

    monkeypatch.setattr(copy_kernel, "_copy_bounded_att_1d", CountLaunches())
    source = torch.ones(size=(32, 67), device="cuda")
    target = torch.zeros_like(source)
    copy_bounded_att_tensors(copies=[(source[i], target[i]) for i in range(32)])
    assert launches == [(24,)]
    assert torch.equal(source, target)
    copy_bounded_att_tensors(copies=[])
    with pytest.raises(AssertionError, match="copy tensors must be nonempty"):
        copy_bounded_att_tensors(copies=[(source[:0], target[:0])])
    assert len(launches) == 1


@pytest.mark.parametrize("ndim", [1, 2, 3, 4, 5, 6])
def test_up_to_six_dimensions(ndim: int) -> None:
    # 非整 block 的大小，覆盖每个维度算子的跳步循环与尾部 mask。
    shape = (2,) * (ndim - 1) + (2051,)
    source = torch.arange(2 ** (ndim - 1) * 2051, dtype=torch.float32, device="cuda")
    source = source.reshape(shape)
    if ndim >= 2:
        source = source.transpose(0, ndim - 1)
    target = torch.full(
        size=source.shape,
        fill_value=-1,
        dtype=source.dtype,
        pin_memory=True,
    )
    copy_bounded_att_tensors(copies=[(source, target)])
    torch.cuda.synchronize()
    assert torch.equal(source.cpu(), target)


@pytest.mark.parametrize("ndim", range(1, 7))
@pytest.mark.parametrize("num_stages", [1, 2, 4, 8])
def test_same_shape_batch_uses_one_kernel(
    ndim: int,
    num_stages: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightllm.common.attention_state.bounded_att import copy_kernel

    launches = []
    kernel_name = f"_copy_bounded_att_{ndim}d"
    actual_kernel = getattr(copy_kernel, kernel_name)

    class CountLaunches:
        def __getitem__(self, grid: tuple[int]):
            launches.append(grid)
            return actual_kernel[grid]

    monkeypatch.setattr(copy_kernel, kernel_name, CountLaunches())
    copies, targets, expected_targets = [], [], []
    for _ in range(3):
        shape = (2,) * (ndim - 1) + (517,)
        storage_shape = (4,) * (ndim - 1) + (1034,)
        source = torch.arange(torch.Size(storage_shape).numel(), dtype=torch.float32, device="cuda").reshape(storage_shape)
        # 每个轴都保留间隙，保证降维后仍需调用对应的 1～6 维算子。
        source_view = source[(slice(None, None, 2),) * ndim]
        target = torch.full(size=(*shape[:-1], 1034), fill_value=-17, dtype=source.dtype, pin_memory=True)
        target_view = target[..., ::2]
        copies.append((source_view, target_view))
        expected = target.clone()
        expected[..., ::2].copy_(source_view.cpu())
        targets.append(target)
        expected_targets.append(expected)

    copy_bounded_att_tensors(copies=copies, grid_num=5, num_stages=num_stages)
    torch.cuda.synchronize()
    assert launches == [(5,)]
    for target, expected in zip(targets, expected_targets):
        assert torch.equal(target, expected)


@pytest.mark.parametrize("layout", ["ssm", "conv_mtp", "dense", "width4", "middle_unaligned", "prefix", "singletons", "wide_singleton", "wide_singleton_prefix", "all_singletons"])
def test_transfer_views_preserve_gaps_and_alignment(layout: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from lightllm.common.attention_state.bounded_att import copy_kernel

    launches = []

    class CaptureLaunch:
        def __init__(self, kernel, ndim: int) -> None:
            self.kernel = kernel
            self.ndim = ndim

        def __getitem__(self, grid: tuple[int]):
            def launch(**kwargs):
                launches.append((self.ndim, kwargs["SHAPE"], kwargs["DTYPE"]))
                return self.kernel[grid](**kwargs)
            return launch

    for ndim in range(1, 7):
        name = f"_copy_bounded_att_{ndim}d"
        monkeypatch.setattr(copy_kernel, name, CaptureLaunch(getattr(copy_kernel, name), ndim))

    if layout == "ssm":
        source = torch.arange(2 * 4 * 3 * 128, dtype=torch.float32, device="cuda").view(2, 4, 3, 128)
        target = torch.full(size=(4, 2, 3, 128), fill_value=-1, dtype=source.dtype, pin_memory=True)
        copies = [(source[:, index], target[index]) for index in (0, 2)]
        expected = target.clone()
        for index in (0, 2):
            expected[index].copy_(source[:, index].cpu())
        expected_launch = (2, (2, 192), tl.uint64)
    elif layout == "conv_mtp":
        source = torch.arange(2 * 4 * 16 * 7, dtype=torch.float32, device="cuda").to(torch.bfloat16).view(2, 4, 16, 7)
        target = torch.full(size=(4, 2, 16, 4), fill_value=-1, dtype=source.dtype, pin_memory=True)
        copies = [(source[:, index, :, :4], target[index]) for index in (0, 2)]
        expected = target.clone()
        for index in (0, 2):
            expected[index].copy_(source[:, index, :, :4].cpu())
        expected_launch = (3, (2, 16, 4), tl.uint16)
    elif layout in ("prefix", "singletons", "wide_singleton", "wide_singleton_prefix", "all_singletons"):
        if layout == "prefix":
            shape = (2, 3, 4, 7)
            target_shape = (2, 3, 4, 9)
            width = 5
            expected_launch = (4, (2, 3, 4, 5), tl.uint32)
        elif layout == "singletons":
            shape = (1, 2, 1, 3, 1, 7)
            target_shape = (1, 2, 1, 3, 1, 9)
            width = 5
            # 尾部单元素轴可并入 width；前面的 padding 边界仍然保留。
            expected_launch = (5, (1, 2, 1, 3, 5), tl.uint32)
        elif layout == "wide_singleton_prefix":
            # dtype 扩宽只改变末轴长度，前后的单元素轴均保持不变。
            shape = (1, 3, 4)
            target_shape = (1, 3, 6)
            width = 2
            expected_launch = (3, (1, 3, 1), tl.uint64)
        elif layout == "wide_singleton":
            shape = (3, 4)
            target_shape = (3, 6)
            width = 2
            expected_launch = (2, (3, 1), tl.uint64)
        else:
            shape = target_shape = (1, 1, 1)
            width = 1
            expected_launch = (1, (1,), tl.uint32)
        source = torch.arange(torch.Size(shape).numel(), dtype=torch.float32, device="cuda").view(shape)
        target = torch.full(size=target_shape, fill_value=-1, dtype=source.dtype, pin_memory=True)
        copies = [(source[..., :width], target[..., :width])]
        expected = target.clone()
        expected[..., :width].copy_(source[..., :width].cpu())
    elif layout == "middle_unaligned":
        source = torch.arange(64, dtype=torch.bfloat16, device="cuda")
        target = torch.full(size=(64,), fill_value=-1, dtype=source.dtype, pin_memory=True)
        expected = target.clone()
        copies = []
        # 首尾地址都满足 8 字节对齐，中间请求偏移不对齐；整批必须保留 2 字节搬运。
        for src_offset, dst_offset in ((0, 0), (9, 16), (24, 32)):
            copies.append((source[src_offset:src_offset + 8], target[dst_offset:dst_offset + 8]))
            expected[dst_offset:dst_offset + 8].copy_(source[src_offset:src_offset + 8].cpu())
        expected_launch = (1, (8,), tl.uint16)
    else:
        shape = (4, 2, 3, 128) if layout == "dense" else (4, 6)
        source = torch.arange(torch.Size(shape).numel(), dtype=torch.float32, device="cuda").to(torch.bfloat16).view(shape)
        target = torch.full(size=shape, fill_value=-1, dtype=source.dtype, pin_memory=True)
        copies = [(source[index], target[index]) for index in (0, 2)]
        expected = target.clone()
        for index in (0, 2):
            expected[index].copy_(source[index].cpu())
        expected_launch = (1, (192,), tl.uint64) if layout == "dense" else (1, (3,), tl.uint32)

    copy_bounded_att_tensors(copies=copies)
    torch.cuda.synchronize()
    assert torch.equal(target, expected)
    assert launches == [expected_launch]
