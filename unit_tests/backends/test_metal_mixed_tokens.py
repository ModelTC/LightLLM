"""Mixed token gather preserves prefill inputs and reads GPU next-token slots."""

from unittest.mock import patch

import pytest
import torch

from lightllm.backends.metal.kernels import gather_token_prefill_decode_mixed
from lightllm.backends.metal.tokens import gather_mixed


pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires MPS")


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("batch", [0, 1, 16, 300])
@pytest.mark.parametrize("mode", ["prefill", "decode", "mixed-alias"])
def test_mixed_token_gather_strides_offsets_masks_and_aliases(dtype, batch, mode):
    tokens = max(7, batch * 3)
    host_input = torch.arange(2 * tokens + 1, dtype=dtype)
    host_table = torch.arange(35 * 9, dtype=dtype).view(35, 9) + 1000
    requests = torch.arange(batch, dtype=torch.int64) % 17
    mtp = torch.arange(batch, dtype=torch.int32) % 4
    flags = torch.full((batch,), mode != "prefill", dtype=torch.bool)
    locations = torch.arange(batch, dtype=torch.int64) * 3
    if mode == "mixed-alias":
        flags[::3] = False
        if batch > 4:
            locations[-1] = locations[1]
            locations[-2] = -tokens + int(locations[1])
    # False rows have placeholders; they must not read these invalid indices.
    requests[~flags] = 9999
    mtp[~flags] = 9999
    locations[~flags] = -9999
    expected = host_input.clone()
    table_view = host_table[1::2, 1::2]
    view = expected[1::2]
    for row in range(batch):
        if flags[row]:
            view[locations[row]] = table_view[requests[row], mtp[row]]

    def strided(host):
        storage = torch.empty(2 * host.numel() + 1, dtype=host.dtype, device="mps")
        target = storage[1::2]
        target.copy_(host)
        return target

    actual = host_input.to("mps")
    table = host_table.to("mps")
    metadata = [strided(t) for t in (requests, mtp, flags, locations)]
    with patch.object(torch.Tensor, "tolist", side_effect=AssertionError("CPU metadata readback")):
        result = gather_token_prefill_decode_mixed(actual[1::2], table[1::2, 1::2], *metadata)
    assert result.data_ptr() == actual[1::2].data_ptr()
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
    torch.testing.assert_close(table.cpu(), host_table, atol=0, rtol=0)


def test_unsupported_mixed_gather_is_readonly_before_fallback():
    source = torch.arange(12, device="mps", dtype=torch.int64)
    table = torch.arange(8, device="mps", dtype=torch.int64).view(4, 2)
    requests = torch.tensor([1, 2], device="mps", dtype=torch.int32)
    mtp = torch.zeros(2, device="mps", dtype=torch.int32)
    flags = torch.tensor([True, False], device="mps")
    locations = torch.tensor([3, 6], device="mps", dtype=torch.int32)
    before = source.cpu().clone()
    assert not gather_mixed(source, table.float(), requests, mtp, flags, locations)
    assert not gather_mixed(source, table, requests.cpu(), mtp, flags, locations)
    assert not gather_mixed(source, table, requests, mtp, flags.to(torch.int32), locations)
    torch.testing.assert_close(source.cpu(), before, atol=0, rtol=0)
    # A floating table retains the existing Torch conversion path.
    result = gather_token_prefill_decode_mixed(source, table.float(), requests, mtp, flags, locations)
    before[3] = 2
    torch.testing.assert_close(result.cpu(), before, atol=0, rtol=0)
