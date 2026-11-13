import pytest
import torch
import torch.nn.functional as F

from lightllm.common.basemodel.triton_kernel.linear_att.causal_conv1d_mtp import causal_conv1d_update


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def reference(x, state, weight, bias, activation, slots, accepted, starts):
    output, next_state = x.clone(), state.clone()
    width = weight.shape[1]
    for i, slot in enumerate(slots.tolist()):
        start, end = starts[i : i + 2].tolist()
        if slot == -1 or start == end:
            continue
        offset = accepted[i].item() - 1
        history = state[slot, :, offset : offset + width - 1]
        window = torch.cat((history, x[start:end].T), dim=1)
        values = (
            F.conv1d(
                window.float().unsqueeze(0),
                weight.float().unsqueeze(1),
                bias=None if bias is None else bias.float(),
                groups=x.shape[1],
            )
            .squeeze(0)
            .T
        )
        output[start:end] = F.silu(values) if activation else values
        saved = window[:, 1:]
        next_state[slot].zero_()
        next_state[slot, :, : saved.shape[1]] = saved
    return output, next_state


@pytest.mark.parametrize("width", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("mtp_step", [0, 1, 2, 4])
@pytest.mark.parametrize(
    "dtype,dim,has_bias,activation",
    [
        (torch.float16, 64, False, None),
        (torch.float16, 128, True, "silu"),
        (torch.bfloat16, 65, False, "silu"),
        (torch.bfloat16, 256, True, None),
    ],
)
def test_output_and_state_match_convolution(width, mtp_step, dtype, dim, has_bias, activation):
    torch.manual_seed(42)
    lengths = [mtp_step + 1, max(1, mtp_step), 0, mtp_step + 1]
    starts = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    slots = torch.tensor([2, 0, 3, -1], dtype=torch.int32, device="cuda")
    accepted = torch.tensor([1, mtp_step + 1, 1, 1], dtype=torch.int32, device="cuda")
    x = torch.randn(sum(lengths), dim, dtype=dtype, device="cuda")
    state = torch.randn(5, dim, width - 1 + mtp_step, dtype=dtype, device="cuda")
    weight = torch.randn(dim, width, dtype=dtype, device="cuda")
    bias = torch.randn(dim, dtype=dtype, device="cuda") if has_bias else None
    expected, next_state = reference(x, state, weight, bias, activation, slots, accepted, starts)

    output = causal_conv1d_update(
        x,
        state,
        weight,
        mtp_step=mtp_step,
        bias=bias,
        activation=activation,
        conv_state_indices=slots,
        num_accepted_tokens=accepted,
        query_start_loc=starts,
    )

    assert output.data_ptr() == x.data_ptr()
    tolerance = 2e-2 if dtype == torch.bfloat16 else 1e-2
    torch.testing.assert_close(output, expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(state, next_state, rtol=0, atol=0)
    torch.testing.assert_close(output[starts[-2] :], expected[starts[-2] :], rtol=0, atol=0)


@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("mtp_step", [0, 1, 2, 3])
def test_repeated_decode_preserves_accepted_history(width, mtp_step):
    torch.manual_seed(77)
    dim, length = 32, mtp_step + 1
    weight = torch.randn(dim, width, dtype=torch.float16, device="cuda")
    bias = torch.randn(dim, dtype=torch.float16, device="cuda")
    state = torch.randn(1, dim, width - 1 + mtp_step, dtype=torch.float16, device="cuda")
    expected_state = state.clone()
    slots = torch.tensor([0], dtype=torch.int32, device="cuda")
    starts = torch.tensor([0, length], dtype=torch.int32, device="cuda")
    for step in range(5):
        accepted = torch.tensor([1 + step % length], dtype=torch.int32, device="cuda")
        x = torch.randn(length, dim, dtype=torch.float16, device="cuda")
        expected, expected_state = reference(x, expected_state, weight, bias, "silu", slots, accepted, starts)
        output = causal_conv1d_update(
            x,
            state,
            weight,
            mtp_step=mtp_step,
            bias=bias,
            activation="silu",
            conv_state_indices=slots,
            num_accepted_tokens=accepted,
            query_start_loc=starts,
        )
        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
