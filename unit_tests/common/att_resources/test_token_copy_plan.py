from dataclasses import replace
from math import prod

import pytest
import torch

from lightllm.common.attention_state.token_copy import TokenPageCopyPlan, TransferBatch
from lightllm.common.attention_state.types import AxisMapping, StateTransferLayout, TensorField, TransferRegion


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("pages_device", ["cpu", "cuda"])
@pytest.mark.parametrize("tail", [(), (3,), (2, 8), (2, 3, 5)])
def test_arbitrary_fields_reorder_axes_and_preserve_bits_tail_and_guards(device, pages_device, tail):
    if device == "cpu" and pages_device == "cuda":
        pytest.skip("device copies require device token indices")
    if "cuda" in (device, pages_device) and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(36)
    axes = ("layer", "token", *(f"axis{i}" for i in range(len(tail))))
    source = torch.empty((2, 11, *tail), dtype=torch.bfloat16, device=device)
    source.view(torch.uint8).random_(0, 256)
    destination = torch.full_like(source, 0)
    fields = {"new_encoding_aux": TensorField(source, axes)}

    shape = (3, 2, *tail)
    mappings = ()
    if tail:
        shape = (3, 2, tail[0] + 2, *tail[1:])
        mappings = (AxisMapping("axis0", 0, 1, tail[0]),)
    size = prod(shape) * 2
    region = TransferRegion(
        "new_encoding_aux", 2, 2 + size, torch.bfloat16, ("token", "layer", *axes[2:]), shape, mappings
    )
    layout = StateTransferLayout("custom", 3, 0, size + 4, (region,))
    pages = torch.full(
        (5, size + 4),
        173,
        dtype=torch.uint8,
        device=pages_device,
        pin_memory=pages_device == "cpu" and device == "cuda",
    )
    src_ids = torch.tensor([8, 0, 2, 0, 7, 0, 3, 0, 6, 0], device=device, dtype=torch.int32)[::2]
    dst_ids = torch.tensor([1, 9, 0, 8, 5], device=device)
    writer = TokenPageCopyPlan.bind(fields, layout, pages, True)
    reader = TokenPageCopyPlan.bind({"new_encoding_aux": TensorField(destination, axes)}, layout, pages, False)
    writer.run(TransferBatch.prepare(src_ids, [3, 1]))
    reader.run(TransferBatch.prepare(dst_ids, [3, 1]))
    if device == "cuda":
        torch.cuda.synchronize()
    assert torch.equal(
        source[:, src_ids.long()].contiguous().view(torch.uint8), destination[:, dst_ids].contiguous().view(torch.uint8)
    )
    assert (pages[:, :2] == 173).all() and (pages[:, -2:] == 173).all()
    assert (pages[[0, 2, 4]] == 173).all()
    last = pages[1, 2:-2].view(torch.bfloat16).view(shape)
    assert (last[-1].view(torch.uint8) == 173).all()
    before = pages.clone()
    source.zero_()
    writer.run(TransferBatch.prepare(src_ids, [3, 1], [True, False]))
    if device == "cuda":
        torch.cuda.synchronize()
    assert torch.equal(pages[3], before[3])
    assert not torch.equal(pages[1], before[1])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_token_copy_preserves_noncontiguous_auxiliary_fields(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    tensors = [torch.empty(2, 13, 3, 10, dtype=torch.bfloat16, device=device) for _ in range(2)]
    tensors[0].view(torch.uint8).random_(0, 256)
    tensors[1].view(torch.uint8).fill_(173)
    fields = [dict(aux=TensorField(t[..., ::2], ("layer", "token", "group", "code"))) for t in tensors]
    plan = TokenPageCopyPlan.for_tokens(*fields)
    src, dst = [
        torch.tensor(ids, dtype=torch.int64, device=device)[::2] for ids in ([7, 0, 1, 0, 10, 0], [2, 1, 6, 1, 0, 1])
    ]
    plan.run(TransferBatch(src, dst))
    if device == "cuda":
        torch.cuda.synchronize()
    assert torch.equal(
        tensors[0][:, src, :, ::2].contiguous().view(torch.uint8),
        tensors[1][:, dst, :, ::2].contiguous().view(torch.uint8),
    )
    assert (tensors[1][..., 1::2].contiguous().view(torch.uint8) == 173).all()


def test_invalid_field_mapping_fails_during_binding():
    axes = ("layer", "token", "head", "dim")
    field = TensorField(torch.empty(1, 4, 2, 4), axes)
    region = TransferRegion("aux", 0, 64, torch.float32, axes, (1, 2, 2, 4), (AxisMapping("head", 1, 0, 2),))
    layout = StateTransferLayout("custom", 2, 0, 64, (region,))
    pages = torch.full((2, 64), 173, dtype=torch.uint8)
    with pytest.raises(ValueError, match="outside storage"):
        TokenPageCopyPlan.bind({"aux": field}, layout, pages, True)
    region = replace(region, axis_mappings=(AxisMapping("head", 0, 0, 2, local_circular=True),))
    with pytest.raises(ValueError, match="non-circular"):
        TokenPageCopyPlan.bind({"aux": field}, replace(layout, regions=(region,)), pages, True)
    assert (pages == 173).all()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_packed_tail_is_extended_in_the_copy_without_new_device_indices(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    source = torch.arange(2 * 20 * 12, dtype=torch.int64, device=device).to(torch.uint8).view(2, 20, 12)
    destination = torch.full_like(source, 173)
    axes = ("layer", "token", "byte")
    region = TransferRegion("packed", 16, 208, torch.uint8, axes, (2, 8, 12), ())
    layout = StateTransferLayout("packed", 8, 0, 224, (region,))
    pages = torch.full((3, 224), 173, dtype=torch.uint8, device=device)
    fields = {"packed": TensorField(source, axes, 4)}
    targets = {"packed": TensorField(destination, axes, 4)}
    src = torch.tensor([8, 0, 9, 0, 10, 0, 11, 0, 0, 0, 1, 0], device=device)[::2]
    dst = torch.tensor([4, 5, 6, 7, 12, 13], device=device)
    with pytest.raises(ValueError, match="complete token blocks"):
        TokenPageCopyPlan.bind(fields, layout, pages, True).run(TransferBatch.prepare(src, [1]))
    assert (pages == 173).all()
    writer = TokenPageCopyPlan.bind(fields, layout, pages, True, pad_tail=True)
    reader = TokenPageCopyPlan.bind(targets, layout, pages, False, pad_tail=True)
    batch = TransferBatch.prepare(src, [1])
    writer.run(batch)
    reader.run(TransferBatch.prepare(dst, [1]))
    assert batch.token_indices is src
    src_full, dst_full = [8, 9, 10, 11, 0, 1, 2, 3], [4, 5, 6, 7, 12, 13, 14, 15]
    assert torch.equal(source[:, src_full], destination[:, dst_full])
    assert (pages[[0, 2]] == 173).all()
    assert (pages[:, :16] == 173).all() and (pages[:, 208:] == 173).all()
    assert (destination[:, [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19]] == 173).all()
    with pytest.raises(ValueError, match="block geometry"):
        TokenPageCopyPlan.for_tokens(fields, {"packed": TensorField(destination, axes, 8)})


def test_cpu_copy_checks_all_block_constraints_before_writing_any_field():
    from types import SimpleNamespace
    from lightllm.common.attention_state.cpu_cache_operator import CpuCacheOperator

    axes = ("layer", "token", "byte")
    pages = torch.zeros(1, 16, dtype=torch.uint8)
    plain, packed = torch.full((1, 8, 1), 173, dtype=torch.uint8), torch.full((1, 8, 1), 173, dtype=torch.uint8)
    plans = []
    for i, (tensor, block_size) in enumerate(((plain, 1), (packed, 4))):
        region = TransferRegion("data", i * 8, (i + 1) * 8, torch.uint8, axes, (1, 8, 1), ())
        layout = StateTransferLayout(str(i), 8, i * 8, (i + 1) * 8, (region,))
        plans.append(TokenPageCopyPlan.bind({"data": TensorField(tensor, axes, block_size)}, layout, pages, False))
    operator = CpuCacheOperator(SimpleNamespace(get_bounded_attention_states=lambda: ()))
    operator._bound_token_plans = lambda *args: plans
    with pytest.raises(ValueError, match="complete token blocks"):
        operator.read_req(pages, [0], torch.tensor([0, 1]), {})
    assert (plain == 173).all() and (packed == 173).all()
