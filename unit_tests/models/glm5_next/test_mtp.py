import dataclasses

import pytest
import torch

from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops.kda_decode import fused_recurrent_kda
from lightllm.models.glm5_next.triton_kernel.kpool import compress_pools
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.utils.envs_utils import set_env_start_args


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
        cu = torch.tensor([0, lengths[0], sum(lengths), sum(lengths)], device="cuda", dtype=torch.int32)
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
            torch.zeros(count, device="cuda", dtype=torch.int32),
            ragged,
            torch.zeros(1, device="cuda", dtype=torch.int32),
            torch.tensor([0, count], device="cuda", dtype=torch.int32),
            torch.tensor([count], device="cuda", dtype=torch.int32),
            count,
        )

    prefill(raw[:prefix], packed, ring)
    # Alternate expanded verification, rewound single-token drafting, and re-verification.
    for start, count in [(prefix, width), (prefix + 1, 1), (prefix + 2, 1), (prefix + 1, width)]:
        raw[start : start + count] = torch.randn(count, 256, device="cuda", dtype=torch.bfloat16)
        lengths = torch.arange(start + 1, start + count + 1, device="cuda", dtype=torch.int32)
        compress_pools(
            raw[start : start + count],
            ring,
            packed,
            ape,
            lengths,
            torch.zeros(count, device="cuda", dtype=torch.int32),
            ragged,
            torch.zeros(count, device="cuda", dtype=torch.int32),
            torch.arange(count + 1, device="cuda", dtype=torch.int32),
            lengths,
            1,
            mtp_index=torch.arange(count, device="cuda", dtype=torch.int32),
        )
        expected = torch.zeros_like(packed)
        prefill(raw[: start + count], expected, torch.zeros(2, 4, 256, device="cuda", dtype=torch.bfloat16))
        closing = torch.arange(start, start + count, device="cuda")
        closing = closing[(closing + 1) % 4 == 0]
        assert torch.equal(packed[closing], expected[closing])


def test_glm_post_layer_exposes_normalized_hidden_without_double_norm(monkeypatch):
    from lightllm.models.glm5_next.layer_infer.post_layer_infer import Glm5NextPostLayerInfer
    from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer
    from unittest.mock import patch
    from types import SimpleNamespace

    set_env_start_args(dataclasses.asdict(StartArgs(mtp_mode="eagle_with_att", mtp_step=2)))
    monkeypatch.setenv("LIGHTLLM_CURRENT_RANK_IN_DP", "0")
    monkeypatch.setenv("LIGHTLLM_DP_WORLD_SIZE", "1")
    layer = Glm5NextPostLayerInfer({"rms_norm_eps": 1e-5, "vocab_size": 32, "n_embed": 16})
    hidden = torch.randn(5, 16, device="cuda")
    expected = hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + 1e-5)

    def norm(input, eps, out):
        out.copy_(input * torch.rsqrt(input.square().mean(-1, keepdim=True) + eps))
        return out

    weight = SimpleNamespace(final_norm_weight_=norm)
    with patch.object(LlamaPostLayerInfer, "token_forward", lambda self, x, state, w: self._norm(x, state, w)):
        result = layer.token_forward(hidden, None, weight)
    torch.testing.assert_close(hidden, expected)
    assert result is hidden


@pytest.mark.parametrize("dynamic", [False, True])
def test_kda_backend_routes_accepted_conv_and_ssm_states(dynamic):
    from types import SimpleNamespace

    from lightllm.common.basemodel.attention.base_att import AttControl
    from lightllm.common.basemodel.attention.linear.kda import KDADecodeAttState
    from lightllm.common.basemodel.triton_kernel.linear_att.causal_conv1d import causal_conv1d_update

    torch.manual_seed(53)
    heads, dim, width = 2, 128, 3
    hidden = heads * dim
    lengths = [2, 1, 3] if dynamic else [3, 3, 3]
    requests = [2, 0, 1]
    request_rows = [req for req, length in zip(requests, lengths) for _ in range(length)]
    offsets = [offset for length in lengths for offset in range(length)]
    if dynamic:
        request_rows += [3, 3]
        offsets += [0, 0]
    tokens = len(request_rows)
    conv = torch.randn(4, 3 * hidden, 5, device="cuda", dtype=torch.bfloat16)
    ssm = torch.randn(4 * width, heads, dim, dim, device="cuda") * 0.01
    old_conv, expected_ssm = conv.clone(), ssm.clone()
    accepted_offsets = torch.tensor([1, 2, 0, 0], dtype=torch.int32, device="cuda")
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
    backend = SimpleNamespace(
        mtp_step=2,
        tp_num_heads=heads,
        head_dim=dim,
        tp_hidden_size=hidden,
        lower_bound=-5.0,
        uses_dynamic_spec_verify_layout=lambda: dynamic,
        split_qkv=lambda x: x.split(hidden, -1),
    )
    manager = SimpleNamespace(
        req_to_mtp_state_index=accepted_offsets, HOLD_REQUEST_ID=3, get_mamba_cache=lambda layer: (conv, ssm)
    )
    infer = SimpleNamespace(
        batch_size=tokens,
        b_req_idx=torch.tensor(request_rows, device="cuda", dtype=torch.int32),
        b_mtp_index=torch.tensor(offsets, device="cuda", dtype=torch.int32),
        req_manager=manager,
    )
    weight = SimpleNamespace(
        get_merged_kda_conv_weight=lambda: conv_weight,
        linear_A_log=SimpleNamespace(weight=a),
        linear_dt_bias=SimpleNamespace(weight=bias),
    )
    state = KDADecodeAttState(backend=backend, infer_state=infer)
    state.init_state()
    actual = state.decode_att(
        None,
        None,
        None,
        AttControl(
            linear_att_decode=True,
            linear_att_decode_dict=dict(
                layer_weight=weight, layer_num=0, mixed_qkv=mixed, raw_gate=gate, raw_beta=beta
            ),
        ),
    )
    torch.testing.assert_close(actual.reshape(tokens, heads, dim)[:row], torch.cat(expected), atol=2e-3, rtol=1e-2)
    torch.testing.assert_close(ssm, expected_ssm, atol=2e-3, rtol=1e-2)
    if dynamic:
        assert torch.equal(conv[3], old_conv[3])
