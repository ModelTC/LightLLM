import weakref

import pytest
import torch

from lightllm.common.attention.backends._shared.linear.replayssm_fold import fold_prefix
from lightllm.common.attention.backends.kda.triton.kernels.kda_decode import fused_recurrent_kda
from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.attention_state.bounded_att import ConvSSMStateConfig, KDAState
from lightllm.common.attention_state.bounded_att.kda.replay_kda_state import ReplayKDAState
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.models.glm5_next.triton_kernel.kpool import compress_pools

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("mtp_step", [1, 2, 3, 5])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_kda_mtp_matches_single_token_decode_and_restores_accepted_state(mtp_step, state_dtype):
    torch.manual_seed(53)
    width, heads, dim = mtp_step + 1, 2, 128
    # Nonconsecutive requests plus a zero-length graph-padding sequence.
    req_ids = torch.tensor([3, 1, 5], device="cuda", dtype=torch.int32)
    slots = req_ids[:, None] * width + torch.arange(width, device="cuda", dtype=torch.int32)
    states = torch.randn(6 * width, heads, dim, dim, device="cuda", dtype=state_dtype) * 0.1
    accepted = torch.tensor([width, 1, 1], device="cuda", dtype=torch.int32)
    for lengths in ([width, 1, 0], [2, width, 0], [1, 1, 0]):
        cu = torch.tensor(
            [0, lengths[0], sum(lengths), sum(lengths)],
            device="cuda",
            dtype=torch.int32,
        )
        tokens = sum(lengths)
        q, k, v, gate = [torch.randn(1, tokens, heads, dim, device="cuda", dtype=torch.bfloat16) for _ in range(4)]
        beta = torch.randn(1, tokens, heads, device="cuda", dtype=torch.bfloat16)
        a = torch.randn(heads, device="cuda")
        bias = torch.randn(heads * dim, device="cuda")
        reference = states.clone()
        before = states.clone()
        expected = torch.empty_like(v)
        for request, length in enumerate(lengths):
            if not length:
                continue
            start = sum(lengths[:request])
            current = reference[slots[request, accepted[request] - 1]].clone().unsqueeze(0)
            for offset in range(length):
                token = start + offset
                out, _ = fused_recurrent_kda(
                    q[:, token : token + 1],
                    k[:, token : token + 1],
                    v[:, token : token + 1],
                    gate[:, token : token + 1].reshape(1, 1, -1),
                    beta[:, token : token + 1],
                    a,
                    bias,
                    current,
                    torch.zeros(1, device="cuda", dtype=torch.int32),
                )
                expected[:, token : token + 1] = out
                reference[slots[request, offset]] = current[0]
        actual, _ = fused_recurrent_kda(
            q,
            k,
            v,
            gate.reshape(1, tokens, -1),
            beta,
            a,
            bias,
            states,
            slots,
            cu_seqlens=cu,
            num_accepted_tokens=accepted,
        )
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=1e-2)
        torch.testing.assert_close(states, reference, atol=2e-3, rtol=1e-2)
        assert torch.equal(states[slots[2]], before[slots[2]])
        # The next verify must begin at the accepted candidate, not the last written slot.
        accepted = torch.tensor([min(2, lengths[0]), 1, 1], device="cuda", dtype=torch.int32)


@pytest.mark.parametrize("mtp_step", [1, 2, 3, 5])
@pytest.mark.parametrize("prefix", [1, 2, 3, 4, 7, 15])
def test_kpool_verify_and_draft_rewind_match_full_prefill(mtp_step, prefix):
    torch.manual_seed(53)
    width, capacity = mtp_step + 1, 128
    raw = torch.randn(capacity, 256, device="cuda", dtype=torch.bfloat16)
    ape = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    ring = torch.zeros(2, 4 + mtp_step, 256, device="cuda", dtype=torch.bfloat16)
    packed = torch.zeros(capacity, 132, device="cuda", dtype=torch.uint8)
    ragged = torch.arange(capacity, device="cuda", dtype=torch.int32)

    def prefill(values, destination, tail):
        count = values.shape[0]
        compress_pools(
            values,
            tail,
            destination,
            ape,
            torch.arange(1, count + 1, device="cuda", dtype=torch.int32),
            ragged[:count],
            torch.zeros(1, device="cuda", dtype=torch.int32),
            torch.tensor([0, count], device="cuda", dtype=torch.int32),
            torch.tensor([count], device="cuda", dtype=torch.int32),
            count,
        )

    prefill(raw[:prefix], packed, ring)
    # Alternate expanded verification, rewound single-token drafting, and re-verification.
    for start, count in [
        (prefix, width),
        (prefix + 1, 1),
        (prefix + 2, 1),
        (prefix + 1, width),
    ]:
        raw[start : start + count] = torch.randn(count, 256, device="cuda", dtype=torch.bfloat16)
        lengths = torch.arange(start + 1, start + count + 1, device="cuda", dtype=torch.int32)
        compress_pools(
            raw[start : start + count],
            ring,
            packed,
            ape,
            lengths,
            ragged[start : start + count],
            torch.zeros(count, device="cuda", dtype=torch.int32),
            torch.arange(count + 1, device="cuda", dtype=torch.int32),
            lengths,
            1,
            b_mtp_index=torch.arange(count, device="cuda", dtype=torch.int32),
        )
        expected = torch.zeros_like(packed)
        prefill(
            raw[: start + count],
            expected,
            torch.zeros(2, 4, 256, device="cuda", dtype=torch.bfloat16),
        )
        closing = torch.arange(start, start + count, device="cuda")
        closing = closing[(closing + 1) % 4 == 0]
        assert torch.equal(packed[closing], expected[closing])


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("role", ["main", "draft0"])
@pytest.mark.parametrize("mode,mtp_step", [("native", 2), ("replay", 2), ("native", 0), ("replay", 0)])
def test_kda_backend_routes_accepted_conv_and_ssm_states(dynamic, role: str, mode, mtp_step):
    from types import SimpleNamespace

    from lightllm.common.attention.backends._shared.linear.causal_conv1d import causal_conv1d_update
    from lightllm.common.attention.backends.kda.replay.backend import ReplayKDALinearAttBackend
    from lightllm.common.attention.backends.kda.triton.backend import KDALinearAttBackend

    torch.manual_seed(53)
    heads, dim, width = 2, 128, mtp_step + 1
    hidden = heads * dim
    lengths = [2, 1, 3] if dynamic and mtp_step else [width] * 3
    requests = [2, 0, 1]
    request_rows = [req for req, length in zip(requests, lengths) for _ in range(length)]
    offsets = [offset for length in lengths for offset in range(length)]
    if dynamic:
        request_rows += [3, 3]
        offsets += [0, 0]
    tokens = len(request_rows)
    conv = torch.randn(4, 3 * hidden, 3 + mtp_step, device="cuda", dtype=torch.bfloat16)
    ssm = torch.randn(4 * width, heads, dim, dim, device="cuda") * 0.01
    old_conv, expected_ssm = conv.clone(), ssm.clone()
    accepted_offsets = torch.tensor([1, 2, 0, 0] if mtp_step else [0, 0, 0, 0], dtype=torch.int32, device="cuda")
    mixed = torch.randn(tokens, 3 * hidden, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    beta = torch.randn(tokens, heads, device="cuda", dtype=torch.bfloat16)
    conv_weight = torch.randn(3 * hidden, 4, device="cuda", dtype=torch.bfloat16) * 0.1
    a, bias = torch.randn(heads, device="cuda"), torch.randn(hidden, device="cuda")
    expected = []
    row = 0
    zero = torch.zeros(1, dtype=torch.int32, device="cuda")
    for req, length in zip(requests, lengths):
        offset = int(accepted_offsets[req])
        history = old_conv[req : req + 1, :, offset : offset + 3].contiguous()
        state = expected_ssm[req * width + offset].clone().unsqueeze(0)
        for token in range(length):
            convolved = causal_conv1d_update(
                mixed[row : row + 1].clone(),
                history,
                conv_weight,
                bias=None,
                activation="silu",
                conv_state_indices=zero,
            )
            q, k, v = [part.reshape(1, 1, heads, dim) for part in convolved.split(hidden, -1)]
            output, _ = fused_recurrent_kda(
                q,
                k,
                v,
                gate[row : row + 1].reshape(1, 1, hidden),
                beta[row : row + 1].reshape(1, 1, heads),
                a,
                bias,
                state,
                zero,
            )
            expected.append(output.reshape(1, heads, dim))
            expected_ssm[req * width + token] = state[0]
            row += 1
    resources = AttentionStateManager(request_capacity=4)
    config = ConvSSMStateConfig(
        logical_layer_ids=(0,),
        global_k_heads=heads,
        global_v_heads=heads,
        key_dim=dim,
        value_dim=dim,
        conv_width=3,
        conv_dtype=conv.dtype,
        ssm_dtype=ssm.dtype,
    )
    selected = None
    for model_role in ("main", "draft0"):
        state_cls = ReplayKDAState if mode == "replay" else KDAState
        linear = state_cls(config, request_capacity=4, mtp_step=mtp_step, device="cuda")
        if mtp_step:
            linear.mtp_state_index.copy_(accepted_offsets if model_role == role else torch.zeros_like(accepted_offsets))
        if model_role == role:
            linear.conv_state[0].copy_(conv)
            if mode == "native":
                linear.ssm_state[0].copy_(ssm)
            else:
                indexes = torch.arange(4, device="cuda") * width + accepted_offsets
                linear.ssm_state[0].copy_(ssm[indexes])
            selected = linear
        resources.register_state(model_role, linear)
    resources.finish_state_registration()
    manager = SimpleNamespace(HOLD_REQUEST_ID=3)
    infer = InferStateInfo()
    infer.batch_size = tokens
    infer.b_req_idx = torch.tensor(request_rows, device="cuda", dtype=torch.int32)
    infer.b_mtp_index = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    infer.req_manager = manager
    infer.att_state_manager = resources
    infer.att_state_role = role
    from lightllm.common.attention.api.inputs import KDAInputs, KDAParams
    from unit_tests.common.attention.helpers import make_backend, make_batch, make_context, make_layer_attention

    executor = make_backend(
        ReplayKDALinearAttBackend if mode == "replay" else KDALinearAttBackend,
        mtp_step=mtp_step,
        decode_draft_step=mtp_step,
        dynamic_verify=dynamic,
        tp_num_heads=heads,
        head_dim=dim,
        tp_hidden_size=hidden,
        lower_bound=-5.0,
    )
    batch = make_batch(
        (1,) * tokens, device="cuda", b_req_idx=infer.b_req_idx, b_mtp_index=infer.b_mtp_index, hold_req_id=3
    )
    context = make_context(executor, batch)
    executor.update(context)
    attention = make_layer_attention(
        executor,
        context,
        KDAParams(
            *selected.get_layer_view(0),
            conv_weight,
            a,
            bias,
            mtp_state_index=selected.mtp_state_index,
            replay_state=selected if mode == "replay" else None,
        ),
    )
    actual = attention.backend.forward(attention, context, KDAInputs(mixed, gate, beta))
    torch.testing.assert_close(actual.reshape(tokens, heads, dim)[:row], torch.cat(expected), atol=2e-3, rtol=1e-2)
    if mode == "native":
        checked_slots = len(requests) if dynamic and not mtp_step else expected_ssm.shape[0]
        torch.testing.assert_close(
            selected.ssm_state[0, :checked_slots], expected_ssm[:checked_slots], atol=2e-3, rtol=1e-2
        )
    else:
        for req, length in zip(requests, lengths):
            fold_prefix(selected, torch.tensor([req], dtype=torch.int32, device="cuda"))
            torch.testing.assert_close(
                selected.ssm_state[0, req],
                expected_ssm[req * width + length - 1],
                atol=2e-3,
                rtol=1e-2,
            )
    if dynamic and mtp_step:
        assert torch.equal(selected.conv_state[0, 3], old_conv[3])
