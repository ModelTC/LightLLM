import pytest
import torch

from lightllm.common.basemodel.attention.linear.replay_gdn_decode import get_configs
from unit_tests.common.basemodel.triton_kernel.linear_att.replay_history import (
    accept_replay_window,
    attach_replay_history as ReplaySSMExecutor,
    replay_layer,
)


def ssm_snapshot(backend, req):
    """折进一份 SSM 副本再读出这个请求。不改 canonical SSM，也不动 cursor。"""

    saved = backend.ssm_state
    clone = saved.clone()
    backend.ssm_state = clone
    try:
        fold_prefix(backend, torch.tensor([req], dtype=torch.int32, device=saved.device))
    finally:
        backend.ssm_state = saved
    return clone[:, req]


from lightllm.common.basemodel.triton_kernel.linear_att.replayssm_cursor import advance_cursor
from lightllm.common.basemodel.triton_kernel.linear_att.replayssm_fold import fold_prefix
from lightllm.common.basemodel.triton_kernel.linear_att.ssm_acceptance import commit_ssm_prefix

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def reference(q, k, v, a, b, a_log, bias, state, kda=False, gate_lower_bound=-5.0):
    q, k, v = q.float(), k.float(), v.float()
    q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt() * (q.shape[-1] ** -0.5)
    k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    q = q.repeat_interleave(v.shape[-2] // q.shape[-2], -2)
    k = k.repeat_interleave(v.shape[-2] // k.shape[-2], -2)
    if kda:
        g = gate_lower_bound * torch.sigmoid(a_log.exp()[:, None] * (a.float().view_as(k) + bias.view_as(k)))
        state = state * g.exp()[..., None]
    else:
        g = -a_log.exp() * torch.nn.functional.softplus(a.float() + bias)
        state = state * g.exp()[..., None, None]
    d = b.float().sigmoid()[..., None] * (v - torch.einsum("...kv,...k->...v", state, k))
    state = state + k[..., None] * d[..., None, :]
    return torch.einsum("...kv,...k->...v", state, q), state


@pytest.mark.parametrize("capacity", [4, 8])
@pytest.mark.parametrize("bv", [8, 32], ids=["small_bv_fallback", "precomputed"])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kda", [False, True], ids=["gdn", "kda"])
@pytest.mark.parametrize("activation_dtype", [torch.float16, torch.bfloat16])
def test_fp16_correction_history_preserves_finite_recurrence(capacity, bv, state_dtype, kda, activation_dtype):
    state = torch.zeros(1, 2, 1, 128, 128, dtype=state_dtype, device="cuda")
    ssm_backend = ReplaySSMExecutor(
        state,
        capacity,
        2,
        activation_dtype,
        num_key_heads=1,
        kda=kda,
        compute_mode="precompute",
    )
    reqs = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda")
    accepted = torch.tensor([1, 0], dtype=torch.int32, device="cuda")
    q = torch.zeros(4, 1, 128, dtype=activation_dtype, device="cuda")
    q[:2, 0, 0] = 1
    k = torch.zeros_like(q)
    k[0, 0, 0] = -0.25
    k[1, 0, 0] = 1
    v = torch.zeros_like(q)
    v[:2, 0, 0] = 60000
    a = torch.full((4, 128 if kda else 1), -20, dtype=activation_dtype, device="cuda")
    b = torch.full((4, 1), 20, dtype=activation_dtype, device="cuda")
    alog = torch.zeros(1, device="cuda")
    bias = torch.zeros(128 if kda else 1, device="cuda")
    expected_state = state[0, 0].float().clone()
    for round_idx in range(2):
        # The second round must consume the first round's accepted history.
        # For capacity 4 the next span exactly fills it, so no fold should hide
        # an overflowed history correction by reconstructing from raw inputs.
        assert ssm_backend.cursors[0].item() == 2 * round_idx
        assert not state.any()
        actual = replay_layer(
            ssm_backend,
            0,
            q,
            k,
            v,
            a,
            b,
            alog,
            bias,
            reqs,
            cu,
            run_config={"BV": bv, "num_warps": 1, "num_stages": 3},
        )
        expected = []
        for token in range(2):
            output, expected_state = reference(
                q[token], k[token], v[token], a[token], b[token], alog, bias, expected_state, kda=kda
            )
            expected.append(output)
        assert torch.isfinite(expected_state).all()
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(
            actual[:2].float(),
            torch.stack(expected),
            atol=32 if activation_dtype == torch.bfloat16 else 4,
            rtol=4e-3 if activation_dtype == torch.bfloat16 else 1e-3,
        )
        accept_replay_window(ssm_backend, reqs, accepted, cu)
    # The correction is larger than FP16 can represent even though the
    # recurrent state, inputs, and attention outputs are all finite.
    assert ssm_backend.deltas[0, 0, 0, 1, 0].item() > torch.finfo(torch.float16).max
    assert torch.isfinite(ssm_backend.deltas[0, 0, :, :4]).all()
    assert ssm_backend.cursors[1].item() == 0
    torch.testing.assert_close(ssm_snapshot(ssm_backend, 0)[0].float(), expected_state, atol=128, rtol=0.01)


@pytest.mark.parametrize("capacity", [16, 32, 64])
@pytest.mark.parametrize("width", [1, 3, 5, 16])
@pytest.mark.parametrize("dims", [(32, 64), (128, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_replay_acceptance_flush_and_graph(
    width,
    dims,
    capacity,
    dtype,
    run_config=None,
    compute_mode="inline",
    activation_dtype=torch.bfloat16,
    gate_shift=-3,
    head_geometry=(2, 4),
    kda=False,
    gate_lower_bound=-5.0,
):
    torch.manual_seed(21)
    kdim, vdim = dims
    h, hv = head_geometry
    slots = 5
    state = (torch.randn(2, slots, hv, kdim, vdim, device="cuda") * 0.01).to(dtype)
    ssm_backend = ReplaySSMExecutor(
        state,
        capacity,
        width,
        num_key_heads=h,
        kda=kda,
        gate_lower_bound=gate_lower_bound,
        compute_mode=compute_mode,
        activation_dtype=activation_dtype,
    )
    expected = state.float().clone()
    pending = [0, 0]
    state_atol = 2e-5 if dtype == torch.float32 else 0.002
    # Independent FP32 reductions can straddle a BF16 rounding boundary.
    # Allow one relative BF16 spacing as well as accumulated near-zero error.
    state_rtol = 2e-4 if dtype == torch.float32 else torch.finfo(dtype).eps
    reqs = torch.tensor([2, 0, slots - 1], device="cuda", dtype=torch.int32)
    cu = torch.tensor([0, width, 2 * width - 1, 2 * width - 1], device="cuda", dtype=torch.int32)
    if width == 1:
        cu = torch.tensor([0, 1, 2, 2], device="cuda", dtype=torch.int32)
    tokens = 3 * width
    q = torch.randn(tokens, h, kdim, device="cuda", dtype=activation_dtype)
    k = torch.randn_like(q)
    v = torch.randn(tokens, hv, vdim, device="cuda", dtype=activation_dtype)
    a = torch.randn(tokens, hv * kdim if kda else hv, device="cuda", dtype=activation_dtype) + gate_shift
    b = torch.randn(tokens, hv, device="cuda", dtype=activation_dtype)
    alog = torch.randn(hv, device="cuda") * 0.1
    bias = torch.randn(hv * kdim if kda else hv, device="cuda") * 0.1
    accepted = torch.zeros(slots, dtype=torch.int32, device="cuda")

    def step():
        out = [
            replay_layer(ssm_backend, layer, q, k, v, a, b, alog, bias, reqs, cu, run_config=run_config)
            for layer in range(2)
        ]
        accept_replay_window(ssm_backend, reqs, accepted, cu)
        return out

    # Compile on disposable state, then capture with the HOLD slot only.
    saved = state.clone()
    step()
    state.copy_(saved)
    ssm_backend.cursors.zero_()
    reqs.fill_(slots - 1)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = step()
    reqs.copy_(torch.tensor([2, 0, slots - 1], device="cuda", dtype=torch.int32))
    for iteration in range(3 * capacity + 5):
        counts = [iteration % width + 1, min((iteration + 1) % width + 1, max(width - 1, 1))]
        accepted[2] = counts[0] - 1
        accepted[0] = counts[1] - 1
        for seq, req in enumerate([2, 0]):
            query_len = width if seq == 0 else max(width - 1, 1)
            if pending[seq] + query_len > capacity:
                ssm_backend.fold_accepted_prefix(torch.tensor([req], dtype=torch.int32, device="cuda"))
                expected[:, req] = expected[:, req].to(dtype).float()
                pending[seq] = 0
            pending[seq] += counts[seq]
        graph.replay()
        for layer in range(2):
            for seq, req in enumerate([2, 0]):
                start = seq * width
                length = width if seq == 0 else max(width - 1, 1)
                cur = expected[layer, req].clone()
                for j in range(length):
                    t = start + j
                    out, cur = reference(
                        q[t],
                        k[t],
                        v[t],
                        a[t],
                        b[t],
                        alog,
                        bias,
                        cur,
                        kda=kda,
                        gate_lower_bound=gate_lower_bound,
                    )
                    torch.testing.assert_close(outputs[layer][t].float(), out, atol=0.006, rtol=0.02)
                    if j + 1 == counts[seq]:
                        expected[layer, req].copy_(cur)
        if iteration == 2 * capacity + 1:
            ssm_backend.fold_accepted_prefix(reqs[:2])
            expected = expected.to(dtype).float()
            pending = [0, 0]
            torch.testing.assert_close(state[:, :4].float(), expected[:, :4], atol=state_atol, rtol=state_rtol)
    ssm_backend.fold_accepted_prefix(reqs[:2])
    torch.testing.assert_close(state.float(), expected.to(dtype).float(), atol=state_atol, rtol=state_rtol)


@pytest.mark.parametrize("kda", [False, True], ids=["gdn", "kda"])
@pytest.mark.parametrize("compute_mode", ["inline", "precompute"])
@pytest.mark.parametrize("bv", [32, 128])
@pytest.mark.parametrize("width", [1, 3])
def test_overflow_preserves_shared_keys_across_heads_and_graph(kda, compute_mode, bv, width):
    torch.manual_seed(123)
    batch, kh, hv, dim, capacity = 64, 4, 16, 128, 4
    dtype = torch.bfloat16
    initial = torch.randn(2, batch + 1, hv, dim, dim, device="cuda", dtype=dtype) * 0.05
    config = {"BV": bv, "num_warps": 4 if bv == 128 else 1, "num_stages": 1}
    backend, reference_backend = [
        ReplaySSMExecutor(
            initial.clone(),
            capacity,
            width,
            dtype,
            num_key_heads=kh,
            kda=kda,
            compute_mode=compute_mode,
        )
        for _ in range(2)
    ]
    reqs = torch.arange(batch, device="cuda", dtype=torch.int32)
    reqs[-2:] = batch
    windows = (torch.arange(reqs.numel(), device="cuda", dtype=torch.int32) % width) + 1
    windows[0] = 0
    cu = torch.cat((windows.new_zeros(1), windows.cumsum(0).int()))
    tokens = int(cu[-1])
    q = torch.randn(tokens, kh, dim, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn(tokens, hv, dim, device="cuda", dtype=dtype)
    a = torch.full((tokens, hv * dim if kda else hv), -2.0, device="cuda", dtype=dtype)
    b = torch.randn(tokens, hv, device="cuda", dtype=dtype)
    alog = torch.zeros(hv, device="cuda")
    bias = torch.zeros(hv * dim if kda else hv, device="cuda")

    def step(owner):
        outputs = [
            replay_layer(owner, layer, q, k, v, a, b, alog, bias, reqs, cu, run_config=config) for layer in range(2)
        ]
        accept_replay_window(owner, reqs, cu_seqlens=cu)
        return outputs

    step(backend)
    backend.ssm_state.copy_(initial)
    backend.cursors.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = step(backend)
    backend.ssm_state.copy_(initial)
    backend.cursors.zero_()
    for _ in range(2 * capacity + 1):
        for tensor in (q, k, v, b):
            tensor.normal_()
        lengths = reference_backend.cursors[reqs[:-2].long()]
        overflow = reqs[:-2][(windows[:-2] > 0) & (lengths + windows[:-2] > capacity)]
        if overflow.numel():
            fold_prefix(state=reference_backend, reqs=overflow, run_config=config)
            reference_backend.cursors.index_fill_(0, overflow.long(), 0)
        expected = step(reference_backend)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(backend.ssm_state, reference_backend.ssm_state, rtol=0, atol=0)
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, rtol=torch.finfo(dtype).eps, atol=2e-5)
        assert torch.equal(backend.cursors, reference_backend.cursors)
    torch.testing.assert_close(backend.ssm_state[:, 0], initial[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(backend.ssm_state[:, batch], initial[:, batch], rtol=0, atol=0)


@pytest.mark.parametrize("save_big", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_checkpoint_restores_pending_history_and_accepted_conv(runtime_start_args, save_big, dtype):
    from lightllm.common.attention_state import AttentionStateManager
    from lightllm.common.attention_state.bounded_att import ConvSSMStateConfig
    from lightllm.common.attention_state.bounded_att.gdn.replay_gdn_state import ReplayGDNState

    torch.manual_seed(42)
    state_rtol = 2e-5 if dtype == torch.float32 else torch.finfo(dtype).eps
    config = ConvSSMStateConfig(
        logical_layer_ids=(0, 1),
        global_k_heads=2,
        global_v_heads=2,
        key_dim=32,
        value_dim=64,
        conv_width=3,
        conv_dtype=torch.bfloat16,
        ssm_dtype=dtype,
    )
    state = ReplayGDNState(config, 4, mtp_step=2, device="cuda", ssm_replay_config={"cache_len": 4})
    resources = AttentionStateManager(request_capacity=4)
    resources.register_state("main", state)
    resources.finish_state_registration()
    layout = resources.get_pd_operator().get_pd_bounded_att_page_layouts()[state.state_name]
    state.initialize_page_pools(big_page_capacity=2, small_page_capacity=2)
    state.mtp_state_index[1] = 1
    state.conv_state.normal_()
    state.ssm_state.zero_()
    state.raw_keys.normal_()
    state.raw_values.normal_()
    state.gates.uniform_(-0.1, -0.01)
    state.betas.uniform_(0.1, 0.9)
    active_before = state.ssm_state[:, 1].clone()
    conv_expected = state.conv_state[:, 1, :, :3].clone()
    block = torch.empty((1, layout.byte_end), dtype=torch.uint8, device="cuda")
    kind = "big" if save_big else "small"
    pool = state.big_pages_pool if save_big else state.small_pages_pool
    for count in (2, 1):
        state.cursors[1] = count
        expected = active_before.float().clone()
        for i in range(count):
            key = state.raw_keys[:, 1, :, i].float()
            key /= (key.square().sum(-1, keepdim=True) + 1e-6).sqrt()
            expected *= state.gates[:, 1, :, i].exp()[..., None, None]
            delta = state.betas[:, 1, :, i, None] * (
                state.raw_values[:, 1, :, i].float() - torch.einsum("lhkv,lhk->lhv", expected, key)
            )
            expected += key[..., None] * delta[..., None, :]
        materialized = ssm_snapshot(state, 1)
        torch.testing.assert_close(materialized, expected.to(dtype), rtol=state_rtol, atol=2e-5)
        state.save_bounded_att_runtime_to_pages([1], [8], kind, [0])
        state.write_to_pd_page(1, block, 0, 8, layout)
        torch.cuda.synchronize()
        torch.testing.assert_close(pool.ssm_state[0], active_before.cpu(), rtol=0, atol=0)
        torch.testing.assert_close(pool.conv_state[0], conv_expected.cpu(), rtol=0, atol=0)
        torch.testing.assert_close(state.ssm_state[:, 1], active_before, rtol=0, atol=0)
        assert state.cursors[1].item() == count
        for restore in (
            lambda: state.restore_bounded_att_runtime_from_pages([2], [8], kind, [0]),
            lambda: state.read_from_pd_page(block, 2, 0, 8, layout),
        ):
            state.cursors[2] = 3
            state.mtp_state_index[2] = 2
            restore()
            assert state.cursors[2].item() == state.mtp_state_index[2].item() == 0
            torch.testing.assert_close(state.ssm_state[:, 2], active_before, rtol=0, atol=0)
            torch.testing.assert_close(state.conv_state[:, 2, :, :3], conv_expected, rtol=0, atol=0)
    state.reset_bounded_att_state([2])
    assert not state.ssm_state[:, 2].any()
    assert not state.conv_state[:, 2].any()


@pytest.mark.parametrize("kda", [False, True], ids=["gdn", "kda"])
@pytest.mark.parametrize("compute_mode", ["inline", "precompute"])
def test_mtp_commit_keeps_folded_prefix_length(kda, compute_mode):
    # 最后一层按整窗全接受推进 cursor。放不下时 cursor 变成窗口长度，而不是旧 cursor 加窗口。
    torch.manual_seed(19 + int(kda))
    hv, kh, dim = 2, 2, 32
    dtype = torch.bfloat16
    state = torch.randn(1, 2, hv, dim, dim, device="cuda", dtype=dtype) * 0.05
    backend = ReplaySSMExecutor(
        state,
        4,
        2,
        dtype,
        num_key_heads=kh,
        kda=kda,
        compute_mode=compute_mode,
    )
    reqs = torch.tensor([0], device="cuda", dtype=torch.int32)
    indices = torch.zeros(2, device="cuda", dtype=torch.int32)
    alog = torch.zeros(hv, device="cuda")
    bias = torch.zeros(hv * dim if kda else hv, device="cuda")
    cu = torch.tensor([0, 2], device="cuda", dtype=torch.int32)
    for step in range(3):
        q = torch.randn(2, kh, dim, device="cuda", dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn(2, hv, dim, device="cuda", dtype=dtype)
        a = torch.full((2, hv * dim if kda else hv), -2.0, device="cuda", dtype=dtype)
        b = torch.randn(2, hv, device="cuda", dtype=dtype)
        replay_layer(
            backend,
            0,
            q,
            k,
            v,
            a,
            b,
            alog,
            bias,
            reqs,
            cu,
            run_config={"BV": 32, "num_warps": 1, "num_stages": 3},
        )
        lengths = torch.tensor([2], device="cuda", dtype=torch.int32)
        advance_cursor(backend, reqs, cu)
        commit_ssm_prefix(indices, reqs, lengths, lengths, backend.cursors)
        if step < 2:
            assert int(backend.cursors[0]) == 2 * (step + 1)
        else:
            assert int(backend.cursors[0]) == 2
    assert torch.isfinite(backend.ssm_state).all()
