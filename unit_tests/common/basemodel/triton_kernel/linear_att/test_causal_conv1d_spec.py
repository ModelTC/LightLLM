from typing import Optional

import pytest
import torch

from lightllm.common.basemodel.triton_kernel.linear_att.causal_conv1d_spec import causal_conv1d_update


def causal_conv1d_ref(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    mtp_step: int,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    pad_slot_id: int = -1,
) -> torch.Tensor:
    """
    Pure-PyTorch reference for causal_conv1d_update.

    The kernel's algorithm per token iteration:
      acc = bias
      for j in range(width):
          if j == 0:        acc += col0      * weight[:, 0]
          elif j < width-1: acc += col{j}    * weight[:, j]
          else:             acc += x[t]      * weight[:, width-1]
      shift: col0=col1, col1=col2, ..., col_{w-3}=col_{w-2}, col_{w-2}=x[t]
    """
    assert conv_state_indices is not None
    batch = conv_state_indices.size(0)
    dim = x.size(1)
    assert x.size(0) % batch == 0
    seqlen = x.size(0) // batch
    _, width = weight.shape
    _, _, state_len = conv_state.size()
    assert state_len == width - 1 + mtp_step

    out = x.clone()
    conv_state = conv_state.clone()

    for b in range(batch):
        slot = conv_state_indices[b].item()
        if slot == pad_slot_id:
            continue

        if query_start_loc is not None:
            start = query_start_loc[b].item()
            end = query_start_loc[b + 1].item()
            local_seqlen = end - start
        else:
            start = b * seqlen
            local_seqlen = seqlen

        accepted = 1
        if num_accepted_tokens is not None:
            accepted = num_accepted_tokens[b].item()
        offset = accepted - 1

        state_3d = conv_state[slot]

        # STEP 1: initial columns from conv_state starting at offset
        cols = [state_3d[:, offset + k].clone() for k in range(width - 1)]

        # STEP 2: update conv_state in the reference too
        new_state = []
        for k in range(width - 2):
            new_state.append(state_3d[:, offset + k + 1].clone())
        x_chunk = x[start : start + local_seqlen, :]
        for t_ in range(local_seqlen):
            new_state.append(x_chunk[t_, :].clone())
        while len(new_state) < state_len:
            new_state.append(torch.zeros(dim, device=x.device, dtype=x.dtype))
        stacked = torch.stack(new_state, dim=1)
        write_len = min(state_len, stacked.size(1))
        conv_state[slot, :, :write_len] = stacked[:, :write_len]

        # STEP 3-5: compute output
        for t_ in range(local_seqlen):
            acc = torch.zeros(dim, device=x.device, dtype=torch.float32)
            if bias is not None:
                acc += bias.float()

            for k in range(width):
                if k == 0:
                    v = cols[0].float()
                elif k == width - 1:
                    v = x_chunk[t_, :].float()
                else:
                    v = cols[k].float()
                w = weight[:, k].float()
                acc += v * w

            if activation in ("silu", "swish"):
                acc = acc / (1 + torch.exp(-acc))

            out[start + t_, :] = acc.to(x.dtype)

            # shift cols: col0=col1, col1=col2, ..., col_{w-2}=x[t]
            for k in range(width - 2):
                cols[k] = cols[k + 1].clone()
            if width >= 2:
                cols[width - 2] = x_chunk[t_, :].clone()

    return out


def make_tensors(
    batch: int,
    dim: int,
    width: int,
    mtp_step: int,
    has_bias: bool,
    num_slots: int = None,
    dtype: torch.dtype = torch.float16,
    seed: int = 42,
    device: str = "cuda",
):
    torch.manual_seed(seed)
    if num_slots is None:
        num_slots = batch

    state_len = width - 1 + mtp_step
    x = torch.randn(batch * (mtp_step + 1), dim, device=device, dtype=dtype)
    weight = torch.randn(dim, width, device=device, dtype=dtype)
    bias = torch.randn(dim, device=device, dtype=dtype) if has_bias else None
    conv_state = torch.randn(num_slots, dim, state_len, device=device, dtype=dtype)
    conv_state.add_(0.5)

    conv_state_indices = torch.arange(batch, device=device, dtype=torch.int32) % num_slots
    num_accepted_tokens = torch.ones(batch, device=device, dtype=torch.int32)
    query_start_loc = torch.arange(batch + 1, device=device, dtype=torch.int32) * (mtp_step + 1)

    return x, conv_state, weight, bias, conv_state_indices, num_accepted_tokens, query_start_loc


@pytest.mark.parametrize("width", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("mtp_step", [0, 1, 2])
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("activation", [None, "silu"])
def test_single_request(width, dim, mtp_step, has_bias, activation):
    """Single request, no pad slots, basic functionality."""
    batch = 1
    x, conv_state, weight, bias, conv_state_indices, num_accepted_tokens, query_start_loc = make_tensors(
        batch=batch,
        dim=dim,
        width=width,
        mtp_step=mtp_step,
        has_bias=has_bias,
        dtype=torch.float16,
    )

    x_orig = x.clone()
    conv_state_ref = conv_state.clone()

    out_triton = causal_conv1d_update(
        x,
        conv_state,
        weight,
        mtp_step=mtp_step,
        bias=bias,
        activation=activation,
        conv_state_indices=conv_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        query_start_loc=query_start_loc,
    )

    out_ref = causal_conv1d_ref(
        x_orig,
        conv_state_ref,
        weight,
        mtp_step=mtp_step,
        bias=bias,
        activation=activation,
        conv_state_indices=conv_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        query_start_loc=query_start_loc,
    )

    rtol, atol = 1e-2, 1e-2
    assert torch.allclose(out_triton, out_ref, rtol=rtol, atol=atol), (
        f"Output mismatch: width={width}, dim={dim}, mtp_step={mtp_step}, "
        f"bias={has_bias}, activation={activation}\n"
        f"max diff={torch.abs(out_triton - out_ref).max().item():.6f}"
    )


@pytest.mark.parametrize("batch", [2, 4])
@pytest.mark.parametrize("width", [3, 4, 5])
@pytest.mark.parametrize("dim", [64, 256])
@pytest.mark.parametrize("mtp_step", [0, 1, 2])
def test_multi_request(batch, width, dim, mtp_step):
    """Multiple requests."""
    x, conv_state, weight, bias, conv_state_indices, num_accepted_tokens, query_start_loc = make_tensors(
        batch=batch,
        dim=dim,
        width=width,
        mtp_step=mtp_step,
        has_bias=True,
        dtype=torch.float16,
    )
    x_orig = x.clone()
    conv_state_ref = conv_state.clone()

    out_triton = causal_conv1d_update(
        x,
        conv_state,
        weight,
        mtp_step=mtp_step,
        bias=bias,
        activation="silu",
        conv_state_indices=conv_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        query_start_loc=query_start_loc,
    )
    out_ref = causal_conv1d_ref(
        x_orig,
        conv_state_ref,
        weight,
        mtp_step=mtp_step,
        bias=bias,
        activation="silu",
        conv_state_indices=conv_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        query_start_loc=query_start_loc,
    )

    rtol, atol = 1e-2, 1e-2
    assert torch.allclose(out_triton, out_ref, rtol=rtol, atol=atol), (
        f"Output mismatch: batch={batch}, width={width}, dim={dim}, mtp_step={mtp_step}\n"
        f"max diff={torch.abs(out_triton - out_ref).max().item():.6f}"
    )


@pytest.mark.parametrize("width", [3, 4])
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("mtp_step", [0, 1])
def test_pad_slots(width, dim, mtp_step):
    """Some slots are padded, should produce same output as reference."""
    batch = 4
    num_slots = 4
    x, conv_state, weight, bias, conv_state_indices, num_accepted_tokens, query_start_loc = make_tensors(
        batch=batch,
        dim=dim,
        width=width,
        mtp_step=mtp_step,
        has_bias=True,
        num_slots=num_slots,
        dtype=torch.float16,
    )

    conv_state_indices[2:] = -1
    x_orig = x.clone()
    conv_state_ref = conv_state.clone()

    out_triton = causal_conv1d_update(
        x,
        conv_state,
        weight,
        mtp_step=mtp_step,
        bias=bias,
        activation="silu",
        conv_state_indices=conv_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        query_start_loc=query_start_loc,
        pad_slot_id=-1,
    )
    out_ref = causal_conv1d_ref(
        x_orig,
        conv_state_ref,
        weight,
        mtp_step=mtp_step,
        bias=bias,
        activation="silu",
        conv_state_indices=conv_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        query_start_loc=query_start_loc,
        pad_slot_id=-1,
    )

    rtol, atol = 1e-2, 1e-2
    assert torch.allclose(out_triton, out_ref, rtol=rtol, atol=atol), (
        f"Output mismatch with pad slots: width={width}, dim={dim}, mtp_step={mtp_step}\n"
        f"max diff={torch.abs(out_triton - out_ref).max().item():.6f}"
    )

    for b in [2, 3]:
        if conv_state_indices[b] == -1:
            start = query_start_loc[b].item()
            end = query_start_loc[b + 1].item()
            for t_ in range(start, end):
                assert torch.allclose(
                    x_orig[t_], out_triton[t_], rtol=1e-6, atol=1e-6
                ), f"Padded entry should not be modified at batch={b}, token={t_ - start}"


@pytest.mark.parametrize("width", [2, 3, 4, 5])
@pytest.mark.parametrize("dim", [64, 256])
@pytest.mark.parametrize("mtp_step", [1, 2])
def test_num_accepted_tokens_varied(width, dim, mtp_step):
    """Varying num_accepted_tokens across batch."""
    batch = 3
    x, conv_state, weight, bias, conv_state_indices, num_accepted_tokens, query_start_loc = make_tensors(
        batch=batch,
        dim=dim,
        width=width,
        mtp_step=mtp_step,
        has_bias=True,
        dtype=torch.float16,
    )

    num_accepted_tokens[0] = 1
    num_accepted_tokens[1] = mtp_step + 1
    num_accepted_tokens[2] = max(1, mtp_step // 2 + 1)

    x_orig = x.clone()
    conv_state_ref = conv_state.clone()

    out_triton = causal_conv1d_update(
        x,
        conv_state,
        weight,
        mtp_step=mtp_step,
        bias=bias,
        activation="silu",
        conv_state_indices=conv_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        query_start_loc=query_start_loc,
    )
    out_ref = causal_conv1d_ref(
        x_orig,
        conv_state_ref,
        weight,
        mtp_step=mtp_step,
        bias=bias,
        activation="silu",
        conv_state_indices=conv_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        query_start_loc=query_start_loc,
    )

    rtol, atol = 1e-2, 1e-2
    assert torch.allclose(out_triton, out_ref, rtol=rtol, atol=atol), (
        f"Output mismatch with varied accepted tokens: width={width}, dim={dim}, mtp_step={mtp_step}\n"
        f"max diff={torch.abs(out_triton - out_ref).max().item():.6f}"
    )


@pytest.mark.parametrize("width", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("mtp_step", [0, 1])
@pytest.mark.parametrize("activation", [None, "silu"])
def test_conv_state_update_correctness(width, dim, mtp_step, activation):
    """Verify that conv_state is updated correctly after the forward pass."""
    batch = 1
    x, conv_state, weight, bias, conv_state_indices, num_accepted_tokens, query_start_loc = make_tensors(
        batch=batch,
        dim=dim,
        width=width,
        mtp_step=mtp_step,
        has_bias=True,
        dtype=torch.float16,
    )

    x_orig = x.clone()
    conv_state_before = conv_state.clone()

    causal_conv1d_update(
        x,
        conv_state,
        weight,
        mtp_step=mtp_step,
        bias=bias,
        activation=activation,
        conv_state_indices=conv_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        query_start_loc=query_start_loc,
    )

    seqlen = mtp_step + 1
    state_len = width - 1 + mtp_step
    slot = conv_state_indices[0].item()

    expected = torch.zeros_like(conv_state_before[slot])
    offset = num_accepted_tokens[0].item() - 1

    for k in range(width - 2):
        src = offset + k + 1
        if src < state_len:
            expected[:, k] = conv_state_before[slot, :, src]

    for t_ in range(seqlen):
        idx = (width - 2) + t_
        if idx < state_len:
            expected[:, idx] = x_orig[t_, :]

    actual = conv_state[slot]
    rtol, atol = 1e-2, 1e-2
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol), (
        f"Conv state mismatch: width={width}, dim={dim}, mtp_step={mtp_step}\n"
        f"max diff={torch.abs(actual - expected).max().item():.6f}"
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dtype_consistency(dtype):
    """Test that the function runs correctly with different dtypes."""
    width = 4
    dim = 128
    mtp_step = 1

    x, conv_state, weight, bias, conv_state_indices, num_accepted_tokens, query_start_loc = make_tensors(
        batch=2,
        dim=dim,
        width=width,
        mtp_step=mtp_step,
        has_bias=True,
        dtype=dtype,
    )
    x_orig = x.clone()
    conv_state_ref = conv_state.clone()

    out_triton = causal_conv1d_update(
        x,
        conv_state,
        weight,
        mtp_step=mtp_step,
        bias=bias,
        activation="silu",
        conv_state_indices=conv_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        query_start_loc=query_start_loc,
    )

    out_ref = causal_conv1d_ref(
        x_orig,
        conv_state_ref,
        weight,
        mtp_step=mtp_step,
        bias=bias,
        activation="silu",
        conv_state_indices=conv_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        query_start_loc=query_start_loc,
    )

    rtol, atol = (1e-2, 1e-2) if dtype == torch.float16 else (2e-2, 2e-2)
    assert torch.allclose(out_triton, out_ref, rtol=rtol, atol=atol), (
        f"Output mismatch for dtype {dtype}: " f"max diff={torch.abs(out_triton - out_ref).max().item():.6f}"
    )


@pytest.mark.parametrize("width", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("activation", [None, "silu"])
@pytest.mark.parametrize("has_bias", [True, False])
def test_known_values(width, activation, has_bias):
    """Deterministic known values to verify numerically."""
    batch = 1
    dim = 4
    mtp_step = 1
    device = "cuda"

    torch.manual_seed(999)
    x = torch.randn(batch * (mtp_step + 1), dim, device=device, dtype=torch.float16)
    weight = torch.randn(dim, width, device=device, dtype=torch.float16)
    bias = torch.randn(dim, device=device, dtype=torch.float16) if has_bias else None
    state_len = width - 1 + mtp_step
    conv_state = torch.randn(1, dim, state_len, device=device, dtype=torch.float16)
    conv_state.add_(0.5)

    conv_state_indices = torch.zeros(batch, device=device, dtype=torch.int32)
    num_accepted_tokens = torch.ones(batch, device=device, dtype=torch.int32)
    qsl = torch.arange(batch + 1, device=device, dtype=torch.int32) * (mtp_step + 1)

    x_orig = x.clone()
    conv_state_ref = conv_state.clone()
    out_triton = causal_conv1d_update(
        x,
        conv_state,
        weight,
        mtp_step=mtp_step,
        bias=bias,
        activation=activation,
        conv_state_indices=conv_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        query_start_loc=qsl,
    )
    out_ref = causal_conv1d_ref(
        x_orig,
        conv_state_ref,
        weight,
        mtp_step=mtp_step,
        bias=bias,
        activation=activation,
        conv_state_indices=conv_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        query_start_loc=qsl,
    )

    rtol, atol = 1e-2, 1e-2
    assert torch.allclose(out_triton, out_ref, rtol=rtol, atol=atol), (
        f"Known value mismatch: width={width}, activation={activation}, bias={has_bias}\n"
        f"max diff={torch.abs(out_triton - out_ref).max().item():.6f}"
    )


# =============================================================================
# Edge-case tests for kernel simplification verification
# =============================================================================


@pytest.mark.parametrize("width", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("mtp_step", [0, 1, 2, 4])
def test_single_step_kernel_vs_pytorch_conv(width, mtp_step):
    """
    Verify the triton kernel against a direct torch.nn.functional.conv1d
    reference (transposed to per-channel depthwise conv). This test ensures
    the causal_conv1d kernel computes the mathematically correct causal conv.
    """
    batch = 1
    dim = 32
    seqlen = mtp_step + 1
    state_len = width - 1 + mtp_step
    device = "cuda"

    torch.manual_seed(123)
    x = torch.randn(seqlen, dim, device=device, dtype=torch.float32)
    weight = torch.randn(dim, width, device=device, dtype=torch.float32)
    bias = torch.randn(dim, device=device, dtype=torch.float32)
    conv_state = torch.randn(1, dim, state_len, device=device, dtype=torch.float32)

    x_orig = x.clone()
    conv_state_ref = conv_state.clone()

    conv_state_indices = torch.zeros(batch, device=device, dtype=torch.int32)
    num_accepted_tokens = torch.ones(batch, device=device, dtype=torch.int32)
    qsl = torch.tensor([0, seqlen], device=device, dtype=torch.int32)

    out_triton = causal_conv1d_update(
        x.clone().half(),
        conv_state.clone().half(),
        weight.half(),
        mtp_step=mtp_step,
        bias=bias.half(),
        activation=None,
        conv_state_indices=conv_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        query_start_loc=qsl,
    ).float()

    # PyTorch causal conv1d reference: pad input with state on the left,
    # then run conv1d and slice output
    history = conv_state_ref[0, :, : width - 1]
    padded_input = torch.cat(
        [
            history.T.contiguous(),
            x_orig,
        ],
        dim=0,
    )

    w_3d = weight.unsqueeze(1).float()
    pytorch_out = (
        torch.nn.functional.conv1d(
            padded_input.T.unsqueeze(0).float(),
            w_3d,
            bias=bias.float(),
            groups=dim,
            padding=0,
            stride=1,
        )
        .squeeze(0)
        .T[:seqlen, :]
    )

    rtol, atol = 1e-2, 1e-2
    assert torch.allclose(out_triton, pytorch_out, rtol=rtol, atol=atol), (
        f"Conv1d mismatch against torch.nn.functional.conv1d: "
        f"width={width}, mtp_step={mtp_step}\n"
        f"max diff={torch.abs(out_triton - pytorch_out).max().item():.6f}"
    )


@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("num_steps", [2, 3, 5])
def test_multi_step_decode_sliding_window(width, num_steps):
    """
    Simulate multiple consecutive decode steps. Each step produces one output
    token and shifts a sliding window of conv_state. The outputs over multi-step
    must match a single forward pass that sees the full sequence at once.
    """
    batch = 1
    dim = 32
    mtp_step = 0
    state_len = width - 1 + mtp_step
    device = "cuda"

    torch.manual_seed(77)
    weight = torch.randn(dim, width, device=device, dtype=torch.float32)
    bias = torch.randn(dim, device=device, dtype=torch.float32)
    init_state = torch.zeros(1, dim, state_len, device=device, dtype=torch.float32)

    all_tokens = torch.randn(num_steps, dim, device=device, dtype=torch.float32)

    idxs = torch.zeros(batch, device=device, dtype=torch.int32)
    nat = torch.ones(batch, device=device, dtype=torch.int32)
    qsl1 = torch.tensor([0, 1], device=device, dtype=torch.int32)

    conv_state_step = init_state.clone().half()
    step_outputs = []

    for step in range(num_steps):
        x_step = all_tokens[step : step + 1, :].clone().half()
        step_outputs.append(
            causal_conv1d_update(
                x_step,
                conv_state_step,
                weight.half(),
                mtp_step=0,
                bias=bias.half(),
                activation=None,
                conv_state_indices=idxs,
                num_accepted_tokens=nat,
                query_start_loc=qsl1,
            ).float()
        )

    step_outputs = torch.cat(step_outputs, dim=0)

    # Build full forward: preload state, then conv1d over full sequence
    full_state = init_state.clone().float()
    padded = torch.cat(
        [
            full_state[0, :, : width - 1].T.contiguous(),
            all_tokens,
        ],
        dim=0,
    )

    w_3d = weight.unsqueeze(1).float()
    expected = (
        torch.nn.functional.conv1d(
            padded.T.unsqueeze(0).float(),
            w_3d,
            bias=bias.float(),
            groups=dim,
        )
        .squeeze(0)
        .T[:num_steps, :]
    )

    rtol, atol = 1e-2, 1e-2
    assert torch.allclose(step_outputs, expected, rtol=rtol, atol=atol), (
        f"Multi-step mismatch: width={width}, num_steps={num_steps}\n"
        f"max diff={torch.abs(step_outputs - expected).max().item():.6f}"
    )


@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("mtp_step", [1, 2, 3])
def test_spec_decode_multi_token_per_step(width, mtp_step):
    """
    Spec-decode: each step processes (mtp_step+1) tokens. After each step,
    some tokens are accepted (num_accepted_tokens varies). The conv_state
    sliding window must correctly preserve history across steps.
    """
    batch = 1
    dim = 32
    seqlen = mtp_step + 1
    state_len = width - 1 + mtp_step
    device = "cuda"
    num_steps = 3

    torch.manual_seed(1234)
    weight = torch.randn(dim, width, device=device, dtype=torch.float32)
    bias = torch.randn(dim, device=device, dtype=torch.float32)
    conv_state = torch.randn(1, dim, state_len, device=device, dtype=torch.float32)
    conv_state_ref = conv_state.clone()

    idxs = torch.zeros(batch, device=device, dtype=torch.int32)
    qsl = torch.tensor([0, seqlen], device=device, dtype=torch.int32)

    # Phase 1: one-step spec decode triton
    x_full = torch.randn(seqlen, dim, device=device, dtype=torch.float32)
    out_triton = causal_conv1d_update(
        x_full.clone().half(),
        conv_state.clone().half(),
        weight.half(),
        mtp_step=mtp_step,
        bias=bias.half(),
        activation="silu",
        conv_state_indices=idxs,
        num_accepted_tokens=torch.ones(batch, device=device, dtype=torch.int32),
        query_start_loc=qsl,
    ).float()

    # Reference: torch conv1d with state preload, then silu
    history = conv_state_ref[0, :, : width - 1].float()
    padded = torch.cat([history.T.contiguous(), x_full], dim=0)
    w_3d = weight.unsqueeze(1).float()
    ref_out = (
        torch.nn.functional.conv1d(
            padded.T.unsqueeze(0).float(),
            w_3d,
            bias=bias.float(),
            groups=dim,
        )
        .squeeze(0)
        .T[:seqlen, :]
    )
    ref_out = ref_out / (1 + torch.exp(-ref_out))

    rtol, atol = 1e-2, 1e-2
    assert torch.allclose(out_triton, ref_out, rtol=rtol, atol=atol), (
        f"Spec-decode mismatch: width={width}, mtp_step={mtp_step}\n"
        f"max diff={torch.abs(out_triton - ref_out).max().item():.6f}"
    )

    # Phase 2: multi-step spec decode with varying acceptance
    conv_state_step = conv_state.clone().half()

    for step in range(num_steps):
        x_step = torch.randn(seqlen, dim, device=device, dtype=torch.float32)
        x_step_half = x_step.clone().half()

        accepted = torch.randint(1, seqlen + 1, (batch,), device=device, dtype=torch.int32)
        nat_step = accepted.clone()
        qsl_step = torch.tensor([0, seqlen], device=device, dtype=torch.int32)

        x_orig_step = x_step_half.clone()
        conv_state_before_step = conv_state_step.clone().half()

        step_out = causal_conv1d_update(
            x_step_half,
            conv_state_step,
            weight.half(),
            mtp_step=mtp_step,
            bias=bias.half(),
            activation="silu",
            conv_state_indices=idxs,
            num_accepted_tokens=nat_step,
            query_start_loc=qsl_step,
        ).float()

        ref_step = causal_conv1d_ref(
            x_orig_step,
            conv_state_before_step,
            weight.half(),
            mtp_step=mtp_step,
            bias=bias.half(),
            activation="silu",
            conv_state_indices=idxs,
            num_accepted_tokens=nat_step,
            query_start_loc=qsl_step,
        ).float()

        assert torch.allclose(step_out, ref_step, rtol=rtol, atol=atol), (
            f"Multi-step spec step={step} mismatch: width={width}, mtp_step={mtp_step}\n"
            f"max diff={torch.abs(step_out - ref_step).max().item():.6f}"
        )


@pytest.mark.parametrize("mtp_step", [0, 1, 2, 4])
def test_kernel_width_2_correctness(mtp_step):
    """
    KERNEL_WIDTH=2 is a degenerate case: restore_conv_state_len = 0,
    meaning no history is preserved in the state update (the mask is always
    False). Verify the kernel still computes correct outputs and the single
    history token (col0) is properly read from the state offset.
    """
    width = 2
    batch = 1
    dim = 64
    seqlen = mtp_step + 1
    state_len = width - 1 + mtp_step
    device = "cuda"

    torch.manual_seed(555)
    x = torch.randn(seqlen, dim, device=device, dtype=torch.float32)
    weight = torch.randn(dim, width, device=device, dtype=torch.float32)
    bias = torch.randn(dim, device=device, dtype=torch.float32)

    idxs = torch.zeros(batch, device=device, dtype=torch.int32)
    qsl = torch.tensor([0, seqlen], device=device, dtype=torch.int32)

    # Test with multiple offsets (varying num_accepted_tokens)
    for nac_val in range(1, min(seqlen + 1, mtp_step + 2)):
        conv_state = torch.randn(1, dim, state_len, device=device, dtype=torch.float32)
        x_half = x.clone().half()
        nat = torch.full((batch,), nac_val, device=device, dtype=torch.int32)

        out_triton = causal_conv1d_update(
            x_half,
            conv_state.clone().half(),
            weight.half(),
            mtp_step=mtp_step,
            bias=bias.half(),
            activation=None,
            conv_state_indices=idxs,
            num_accepted_tokens=nat,
            query_start_loc=qsl,
        ).float()

        # Reference: causal conv with history from state at offset nac_val-1
        history = conv_state[0, :, nac_val - 1 : state_len].float()
        history_pad = history[:, : width - 1]
        padded = torch.cat([history_pad.T.contiguous(), x], dim=0)
        w_3d = weight.unsqueeze(1).float()
        ref_out = (
            torch.nn.functional.conv1d(
                padded.T.unsqueeze(0).float(),
                w_3d,
                bias=bias.float(),
                groups=dim,
            )
            .squeeze(0)
            .T[:seqlen, :]
        )

        assert torch.allclose(out_triton, ref_out, rtol=1e-2, atol=1e-2), (
            f"Width=2 mismatch: mtp_step={mtp_step}, nac={nac_val}\n"
            f"max diff={torch.abs(out_triton - ref_out).max().item():.6f}"
        )


def test_kernel_width_2_no_state_history():
    """
    KERNEL_WIDTH=2 with num_accepted_tokens=1 (offset=0): the restore mask
    is always False, so the state update is fully overwritten by x tokens
    (no history preserved). Verify this yields correct output.
    """
    batch, dim, width, mtp_step = 1, 16, 2, 2
    state_len = width - 1 + mtp_step
    seqlen = mtp_step + 1
    device = "cuda"

    torch.manual_seed(42)
    x = torch.randn(seqlen, dim, device=device, dtype=torch.float32)
    weight = torch.randn(dim, width, device=device, dtype=torch.float32)
    bias = torch.randn(dim, device=device, dtype=torch.float32)
    conv_state = torch.zeros(1, dim, state_len, device=device, dtype=torch.float32)
    conv_state[0, :, 0] = 99.0  # Place a known history value at offset 0

    x_orig = x.clone()
    conv_state_ref = conv_state.clone()

    idxs = torch.zeros(batch, device=device, dtype=torch.int32)
    nat = torch.ones(batch, device=device, dtype=torch.int32)
    qsl = torch.tensor([0, seqlen], device=device, dtype=torch.int32)

    out_triton = causal_conv1d_update(
        x.clone().half(),
        conv_state.clone().half(),
        weight.half(),
        mtp_step=mtp_step,
        bias=bias.half(),
        activation=None,
        conv_state_indices=idxs,
        num_accepted_tokens=nat,
        query_start_loc=qsl,
    ).float()

    # Manual reference: for each token t, acc = bias + col0*w0 + x[t]*w1
    # col0 for token 0 is state[0] = 99.0. Then col0 shifts to x[t].
    ref_out = torch.zeros(seqlen, dim, device=device, dtype=torch.float32)
    col0 = conv_state_ref[0, :, 0].clone()
    for t in range(seqlen):
        acc = bias.clone()
        acc += col0 * weight[:, 0]
        acc += x_orig[t] * weight[:, 1]
        ref_out[t] = acc
        col0 = x_orig[t].clone()

    assert torch.allclose(
        out_triton, ref_out, rtol=1e-2, atol=1e-2
    ), f"Width=2 no-history: max diff={torch.abs(out_triton - ref_out).max().item():.6f}"


@pytest.mark.parametrize("width", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("mtp_step", [0, 1, 2])
def test_output_overwrites_x_inplace(width, mtp_step):
    """
    Verify the kernel indeed overwrites x in-place (out == x). Also
    verify that the output values are different from the original input.
    """
    batch = 1
    dim = 64
    x, conv_state, weight, bias, conv_state_indices, num_accepted_tokens, query_start_loc = make_tensors(
        batch=batch,
        dim=dim,
        width=width,
        mtp_step=mtp_step,
        has_bias=True,
        dtype=torch.float16,
    )
    x_before = x.clone()

    out = causal_conv1d_update(
        x,
        conv_state,
        weight,
        mtp_step=mtp_step,
        bias=bias,
        activation="silu",
        conv_state_indices=conv_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        query_start_loc=query_start_loc,
    )

    assert x.data_ptr() == out.data_ptr(), "out must be the same tensor as x"
    assert torch.allclose(x, out, rtol=0, atol=0), "x must equal out after call"
    assert not torch.allclose(
        x, x_before, rtol=1e-6, atol=1e-6
    ), "x must be overwritten (different from original input)"


@pytest.mark.parametrize("width", [3, 4, 5])
@pytest.mark.parametrize("mtp_step", [0, 1, 2])
def test_bias_none_gives_same_shape(width, mtp_step):
    """bias=None should produce the same output shape as bias specified."""
    batch = 1
    dim = 32
    x, conv_state, weight, _, conv_state_indices, num_accepted_tokens, query_start_loc = make_tensors(
        batch=batch,
        dim=dim,
        width=width,
        mtp_step=mtp_step,
        has_bias=False,
        dtype=torch.float16,
    )

    x_orig = x.clone()
    conv_state_ref = conv_state.clone()

    out_triton = causal_conv1d_update(
        x,
        conv_state,
        weight,
        mtp_step=mtp_step,
        bias=None,
        activation=None,
        conv_state_indices=conv_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        query_start_loc=query_start_loc,
    )
    out_ref = causal_conv1d_ref(
        x_orig,
        conv_state_ref,
        weight,
        mtp_step=mtp_step,
        bias=None,
        activation=None,
        conv_state_indices=conv_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        query_start_loc=query_start_loc,
    )

    assert out_triton.shape == out_ref.shape
    assert out_triton.shape == x_orig.shape
    rtol, atol = 1e-2, 1e-2
    assert torch.allclose(out_triton, out_ref, rtol=rtol, atol=atol), (
        f"No-bias mismatch: width={width}, mtp_step={mtp_step}\n"
        f"max diff={torch.abs(out_triton - out_ref).max().item():.6f}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
