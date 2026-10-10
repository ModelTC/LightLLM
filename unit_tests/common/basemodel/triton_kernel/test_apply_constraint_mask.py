import pytest
import torch

from lightllm.common.basemodel.triton_kernel.post_process.apply_constraint_mask import apply_constraint_mask


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Constraint mask kernel requires CUDA")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("vocab_size,padding", [(1, 0), (31, 5), (32, 0), (33, 31), (257, 7), (8195, 43)])
@pytest.mark.parametrize("mtp", [False, True])
def test_request_indexed_mask_matches_reference(dtype, vocab_size, padding, mtp):
    generator = torch.Generator().manual_seed(2026)
    masks = torch.randint(
        -(1 << 31),
        (1 << 31) - 1,
        (6, 4, (vocab_size + 31) // 32),
        dtype=torch.int32,
        generator=generator,
        pin_memory=True,
    )
    enabled = torch.tensor([False, True, False, False, True, False], pin_memory=True)
    req_ids = torch.tensor([4, 1, 4, 2, 1], dtype=torch.int32)
    positions = torch.tensor([2, 0, 1, 3, 3], dtype=torch.int32) if mtp else torch.zeros(5, dtype=torch.int32)
    # Exercise a padded row stride as well as unused vocabulary columns.
    width = vocab_size + padding
    storage = torch.randn(5, width + 11, dtype=dtype, device="cuda")
    logits = storage[:, :width]
    expected = logits.cpu().clone()
    token_ids = torch.arange(vocab_size)
    for prediction, (req_id, position) in enumerate(zip(req_ids, positions)):
        if enabled[req_id]:
            words = masks[req_id, position, token_ids // 32]
            allowed = ((words >> (token_ids % 32)) & 1).bool()
            expected[prediction, :vocab_size].masked_fill_(~allowed, float("-inf"))
    expected[:, vocab_size:] = float("-inf")
    untouched_padding = storage[:, width:].clone()

    apply_constraint_mask(
        logits,
        req_ids.cuda(),
        masks,
        enabled,
        vocab_size,
        positions.cuda() if mtp else None,
    )

    torch.testing.assert_close(logits.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(storage[:, width:], untouched_padding, rtol=0, atol=0)


def test_pinned_masks_can_be_reused_after_sampling_completion():
    masks = torch.zeros(3, 2, 2, dtype=torch.int32, pin_memory=True)
    enabled = torch.tensor([True, True, False], pin_memory=True)
    masks[0, 0, 0] = -(1 << 31)
    masks[1, 1, 1] = 1
    req_ids = torch.tensor([1, 0], dtype=torch.int32, device="cuda")
    positions = torch.tensor([1, 0], dtype=torch.int32, device="cuda")
    logits = torch.zeros(2, 33, device="cuda")
    apply_constraint_mask(logits, req_ids, masks, enabled, 33, positions)
    # This represents the existing sampling/post_handle completion boundary.
    done = torch.cuda.Event()
    done.record()
    done.synchronize()
    assert torch.isfinite(logits[0]).nonzero().flatten().tolist() == [32]
    assert torch.isfinite(logits[1]).nonzero().flatten().tolist() == [31]

    # Reuse one slot for an ordinary request, and update the other's prefix.
    enabled[1] = False
    masks[0].fill_(-1)
    logits.fill_(1.25)
    apply_constraint_mask(logits, req_ids, masks, enabled, 33, positions)
    torch.testing.assert_close(logits, torch.full_like(logits, 1.25), rtol=0, atol=0)


def test_empty_batch_needs_no_kernel_launch():
    apply_constraint_mask(
        torch.empty(0, 33, device="cuda"),
        torch.empty(0, dtype=torch.int32, device="cuda"),
        torch.empty(1, 1, 2, dtype=torch.int32, pin_memory=True),
        torch.zeros(1, dtype=torch.bool, pin_memory=True),
        33,
    )
