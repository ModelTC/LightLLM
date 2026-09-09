import pytest
import torch

from lightllm.common.basemodel.triton_kernel.sliding_window_state import copy_sliding_window_checkpoint

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("dtype,payload", [(torch.bfloat16, (4, 32)), (torch.float16, (2, 64)), (torch.uint8, (584,))])
@pytest.mark.parametrize("seq_len", [0, 1, 17, 32, 97])
def test_checkpoint_preserves_ring_byte_layout_with_scattered_runtime(dtype, payload, seq_len):
    window, capacity = 32, 200
    table = torch.zeros((2, 128), device="cuda", dtype=torch.int32)
    pool = torch.randint(0, 100, (3, capacity, *payload), device="cuda", dtype=dtype)
    checkpoint = torch.empty((3, window, *payload), dtype=dtype, pin_memory=True)
    window_len = min(window, seq_len)
    # Slot 0 is ordinary KV; force the checkpoint to include it.
    free = torch.arange(capacity - 1, device="cuda", dtype=torch.int32)
    free[1:] = torch.randperm(capacity - 2, device="cuda", dtype=torch.int32) + 1
    slots, restored = free[:window_len], free[window_len : 2 * window_len]
    table[1, seq_len - window_len : seq_len] = slots
    table[0, seq_len - window_len : seq_len] = restored

    copy_sliding_window_checkpoint(pool, table, seq_len, 1, checkpoint)
    positions = torch.arange(seq_len - window_len, seq_len, device="cuda")
    expected = torch.zeros_like(checkpoint, device="cuda")
    expected[:, positions % window] = pool[:, slots.long()]
    torch.cuda.synchronize()
    torch.testing.assert_close(checkpoint.cuda(), expected, atol=0, rtol=0)

    # Restore the same checkpoint to different physical slots.
    copy_sliding_window_checkpoint(pool, table, seq_len, 0, checkpoint, restore=True)
    torch.testing.assert_close(pool[:, restored.long()], pool[:, slots.long()], atol=0, rtol=0)
