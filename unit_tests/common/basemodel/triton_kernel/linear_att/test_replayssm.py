import pytest
import torch

from lightllm.common.basemodel.triton_kernel.linear_att.replayssm import ReplaySSMCache
from lightllm.common.basemodel.triton_kernel.linear_att.ssm_autotune import get_configs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def reference(q, k, v, a, b, a_log, bias, state, kda=False, lower_bound=-5.0):
    q, k, v = q.float(), k.float(), v.float()
    q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt() * (q.shape[-1] ** -0.5)
    k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    q = q.repeat_interleave(v.shape[-2] // q.shape[-2], -2)
    k = k.repeat_interleave(v.shape[-2] // k.shape[-2], -2)
    if kda:
        g = lower_bound * torch.sigmoid(a_log.exp()[:, None] * (a.float().view_as(k) + bias.view_as(k)))
        state = state * g.exp()[..., None]
    else:
        g = -a_log.exp() * torch.nn.functional.softplus(a.float() + bias)
        state = state * g.exp()[..., None, None]
    d = b.float().sigmoid()[..., None] * (v - torch.einsum("...kv,...k->...v", state, k))
    state = state + k[..., None] * d[..., None, :]
    return torch.einsum("...kv,...k->...v", state, q), state


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
    projection_mode="inline",
    activation_dtype=torch.bfloat16,
    gate_shift=-3,
    head_geometry=(2, 4),
    kda=False,
    lower_bound=-5.0,
):
    torch.manual_seed(21)
    kdim, vdim = dims
    h, hv = head_geometry
    slots = 5
    state = (torch.randn(2, slots, hv, kdim, vdim, device="cuda") * 0.01).to(dtype)
    cache = ReplaySSMCache(
        state,
        capacity,
        width,
        num_key_heads=h,
        kda=kda,
        lower_bound=lower_bound,
        projection_mode=projection_mode,
        run_config=run_config,
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
    q = torch.randn(1, tokens, h, kdim, device="cuda", dtype=activation_dtype)
    k = torch.randn_like(q)
    v = torch.randn(1, tokens, hv, vdim, device="cuda", dtype=activation_dtype)
    a = torch.randn(tokens, hv * kdim if kda else hv, device="cuda", dtype=activation_dtype) + gate_shift
    b = torch.randn(tokens, hv, device="cuda", dtype=activation_dtype)
    alog = torch.randn(hv, device="cuda") * 0.1
    bias = torch.randn(hv * kdim if kda else hv, device="cuda") * 0.1
    accepted = torch.zeros(slots, dtype=torch.int32, device="cuda")

    def step():
        cache.prepare_decode(reqs, cu)
        out = [cache.forward(layer, q, k, v, a, b, alog, bias, reqs, cu) for layer in range(2)]
        cache.accept_updates(reqs, accepted)
        return out

    # Compile on disposable state, then capture with the HOLD slot only.
    saved = state.clone()
    step()
    state.copy_(saved)
    cache.cursors.zero_()
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
            if pending[seq] + (width if seq == 0 else max(width - 1, 1)) > capacity:
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
                        q[0, t], k[0, t], v[0, t], a[t], b[t], alog, bias, cur, kda=kda, lower_bound=lower_bound
                    )
                    torch.testing.assert_close(outputs[layer][0, t].float(), out, atol=0.006, rtol=0.02)
                    if j + 1 == counts[seq]:
                        expected[layer, req].copy_(cur)
        if iteration == 2 * capacity + 1:
            cache.merge_accepted_updates(reqs[:2])
            expected = expected.to(dtype).float()
            pending = [0, 0]
            torch.testing.assert_close(state[:, :4].float(), expected[:, :4], atol=state_atol, rtol=state_rtol)
    cache.merge_accepted_updates(reqs[:2])
    torch.testing.assert_close(state.float(), expected.to(dtype).float(), atol=state_atol, rtol=state_rtol)


@pytest.mark.parametrize(
    "state_dtype,activation_dtype",
    [(torch.float32, torch.float32), (torch.float32, torch.bfloat16), (torch.bfloat16, torch.bfloat16)],
)
@pytest.mark.parametrize("projection_mode", ["inline", "precompute"])
@pytest.mark.parametrize("dims", [(32, 64), (128, 128)])
@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize("gate_shift", [-3, -10])
def test_fold_matches_reference_across_precisions(
    state_dtype,
    activation_dtype,
    projection_mode,
    dims,
    width,
    gate_shift,
):
    test_replay_acceptance_flush_and_graph(
        width,
        dims,
        8,
        state_dtype,
        run_config={"BV": 32, "num_warps": 1, "num_stages": 3},
        projection_mode=projection_mode,
        activation_dtype=activation_dtype,
        gate_shift=gate_shift,
    )


@pytest.mark.parametrize("capacity,width", [(4, 1), (4, 3), (4, 4), (8, 1), (8, 3), (8, 4), (8, 5)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_short_replay_history_matches_independent_reference(capacity, width, dtype):
    test_replay_acceptance_flush_and_graph(width, (128, 128), capacity, dtype)


@pytest.mark.parametrize("capacity,width", [(4, 1), (4, 3), (4, 4), (8, 1), (8, 4), (8, 5), (16, 4), (16, 16), (32, 4)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_precomputed_state_matches_independent_reference(capacity, width, dtype):
    test_replay_acceptance_flush_and_graph(
        width,
        (128, 128),
        capacity,
        dtype,
        run_config={"BV": 32, "num_warps": 1},
        projection_mode="precompute",
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_production_gdn_geometry_matches_independent_reference(dtype):
    test_replay_acceptance_flush_and_graph(
        4,
        (128, 128),
        8,
        dtype,
        run_config={"BV": 32, "num_warps": 1, "num_stages": 3},
        projection_mode="precompute",
        head_geometry=(4, 12),
    )


@pytest.mark.parametrize("warps", [1, 4])
@pytest.mark.parametrize("decay", [-3, -10])
@pytest.mark.parametrize(
    "state_dtype,dtype",
    [(torch.float32, torch.float32), (torch.bfloat16, torch.bfloat16), (torch.float32, torch.bfloat16)],
)
@pytest.mark.parametrize("bv", [8, 32])
@pytest.mark.parametrize("width", [1, 4])
def test_precomputed_state_preserves_projection_precision(warps, decay, state_dtype, dtype, width, bv):
    torch.manual_seed(35)
    state = (torch.randn((1, 3, 4, 128, 128), device="cuda") * 0.1).to(state_dtype)
    config = {"BV": bv, "num_warps": warps}
    caches = [
        ReplaySSMCache(
            state.clone(),
            16 if width == 1 else 8,
            width,
            dtype,
            num_key_heads=2,
            projection_mode=mode,
            run_config=config,
        )
        for mode in ("inline", "precompute")
    ]
    reqs = torch.tensor([1, 0], dtype=torch.int32, device="cuda")
    tokens = 2 if width == 1 else 7
    cu = torch.tensor([0, 1, 2] if width == 1 else [0, 4, 7], dtype=torch.int32, device="cuda")
    q = torch.randn((1, tokens, 2, 128), device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn((1, tokens, 4, 128), device="cuda", dtype=dtype)
    a = torch.randn((tokens, 4), device="cuda", dtype=dtype) + decay
    b = torch.randn_like(a)
    alog = torch.zeros(4, device="cuda")
    accepted = torch.zeros(3, dtype=torch.int32, device="cuda")
    for step in range(64):
        for tensor in (q, k, v, b):
            tensor.normal_()
        accepted[0] = step % max(1, width - 1)
        accepted[1] = step % width
        outputs = []
        for cache in caches:
            cache.prepare_decode(reqs, cu)
            outputs.append(cache.forward(0, q, k, v, a, b, alog, alog, reqs, cu))
            cache.accept_updates(reqs, accepted)
        if dtype == torch.float32:
            torch.testing.assert_close(outputs[0], outputs[1], atol=2e-6, rtol=2e-5)
        else:
            torch.testing.assert_close(outputs[0], outputs[1], atol=5e-4, rtol=0.01)
            error = torch.linalg.vector_norm(outputs[0].float() - outputs[1].float())
            scale = torch.linalg.vector_norm(outputs[0].float())
            assert error < scale * (torch.finfo(dtype).eps / 2)
    for cache in caches:
        cache.merge_accepted_updates(reqs)
    torch.testing.assert_close(caches[0].state, caches[1].state, atol=0, rtol=0)


def test_bf16_decode_fold_uses_rounded_checkpoint():
    # Automatic fold and explicit materialization must start from the same BF16
    # checkpoint, including the token computed inside the automatic-fold kernel.
    state = torch.full((1, 2, 1, 32, 64), 0.123, device="cuda", dtype=torch.bfloat16)
    caches = [ReplaySSMCache(state.clone(), 16, 1, torch.float32) for _ in range(2)]
    reqs = torch.tensor([0], device="cuda", dtype=torch.int32)
    for cache in caches:
        cache.raw_keys.zero_()
        cache.raw_values.zero_()
        cache.gates.fill_(-0.01)
        cache.betas.zero_()
        cache.cursors[0] = 16
    caches[1].merge_accepted_updates(reqs)
    q = torch.ones((1, 1, 32), device="cuda")
    k = torch.zeros_like(q)
    v = torch.zeros((1, 1, 64), device="cuda")
    a = torch.full((1, 1), -3.0, device="cuda")
    b = torch.zeros_like(a)
    alog = torch.zeros(1, device="cuda")
    bias = torch.zeros_like(alog)
    outputs = []
    for cache in caches:
        cache.prepare_decode(reqs)
        cache.accept_updates(reqs)
        outputs.append(cache.forward(0, q, k, v, a, b, alog, bias, reqs))
    torch.testing.assert_close(outputs[0], outputs[1], atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(caches[0].state, caches[1].state, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("projection_mode", ["inline", "precompute"])
@pytest.mark.parametrize("capture_metadata", [False, True])
def test_dynamic_verify_lengths_fill_history_before_folding(dtype, projection_mode, capture_metadata):
    torch.manual_seed(31)
    state = (torch.randn(2, 4, 4, 128, 128, device="cuda") * 0.01).to(dtype)
    cache = ReplaySSMCache(
        state, 8, 4, dtype, num_key_heads=2, projection_mode=projection_mode, run_config={"BV": 32, "num_warps": 1}
    )
    expected = state.float().clone()
    pending, phases = [0, 0], [0, 0]
    reqs = torch.full((3,), 3, dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, 4, 8, 8], dtype=torch.int32, device="cuda")
    accepted = torch.zeros(4, dtype=torch.int32, device="cuda")
    q = torch.randn(1, 8, 2, 128, dtype=dtype, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn(1, 8, 4, 128, dtype=dtype, device="cuda")
    a = torch.full((8, 4), -3.0, dtype=dtype, device="cuda")
    b = torch.randn_like(a)
    alog = torch.zeros(4, device="cuda")

    def forward():
        return [cache.forward(layer, q, k, v, a, b, alog, alog, reqs, cu) for layer in range(2)]

    def step():
        positions = cache.prepare_decode(reqs, cu)
        out = forward()
        cache.accept_updates(reqs, accepted)
        return positions, out

    step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    positions = torch.zeros_like(reqs)
    with torch.cuda.graph(graph):
        if capture_metadata:
            positions, outputs = step()
        else:
            outputs = forward()
    lengths = [(3, 4), (2, 1), (3, 2), (1, 4), (4, 3), (0, 2), (2, 0), (4, 4)]
    for iteration in range(24):
        ids = [0, 1] if iteration < 3 or iteration % 2 == 0 else [1, 0]
        widths = lengths[iteration % len(lengths)]
        counts = (
            list(widths) if iteration < 3 else [min(width, 1 + (iteration + i) % 4) for i, width in enumerate(widths)]
        )
        reqs.copy_(reqs.new_tensor(ids + [3]))
        cu.copy_(cu.new_tensor([0, widths[0], sum(widths), 8]))
        accepted.fill_(-1)
        for req, count in zip(ids, counts):
            accepted[req] = count - 1
        for tensor in (q, k, v, b):
            tensor.normal_()
        before = state.clone()
        old_cursors = [phases[req] + pending[req] for req in ids] + [0]
        for req, width in zip(ids, widths):
            if pending[req] + width > 8:
                expected[:, req] = expected[:, req].to(dtype).float()
                pending[req] = 0
                phases[req] ^= 16
        if not capture_metadata:
            positions.copy_(cache.prepare_decode(reqs, cu))
        graph.replay()
        if not capture_metadata:
            cache.accept_updates(reqs, accepted)
        assert positions.tolist() == old_cursors
        for seq, (req, width, count) in enumerate(zip(ids, widths, counts)):
            for layer in range(2):
                current = expected[layer, req].clone()
                for j in range(width):
                    t = (0 if seq == 0 else widths[0]) + j
                    out, current = reference(q[0, t], k[0, t], v[0, t], a[t], b[t], alog, alog, current)
                    torch.testing.assert_close(outputs[layer][0, t].float(), out, atol=0.006, rtol=0.02)
                    if j + 1 == count:
                        expected[layer, req].copy_(current)
            pending[req] += count
        assert cache.cursors.tolist() == [phases[i] + pending[i] for i in range(2)] + [0, 0]
        for output in outputs:
            assert torch.count_nonzero(output[:, sum(widths) :]).item() == 0
        torch.testing.assert_close(state[:, 2:], before[:, 2:], atol=0, rtol=0)
        if iteration == 2:
            assert pending[0] == 8 and phases[0] == 0
            torch.testing.assert_close(state[:, 0], before[:, 0], atol=0, rtol=0)
    cache.merge_accepted_updates(reqs[:2])
    torch.testing.assert_close(
        state.float(),
        expected.to(dtype).float(),
        atol=2e-5 if dtype == torch.float32 else 0.002,
        rtol=2e-4 if dtype == torch.float32 else torch.finfo(dtype).eps,
    )


def test_snapshot_without_history_copies_active_state():
    state = torch.randn((2, 3, 2, 32, 64), device="cuda", dtype=torch.bfloat16)
    cache = ReplaySSMCache(state, 4, 3)
    snapshot = cache.snapshot_accepted_state(1)
    torch.testing.assert_close(snapshot, state[:, 1], rtol=0, atol=0)
    assert cache.cursors[1].item() == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_compact_preserves_accepted_prefix(dtype, run_config=None, kda=False):
    from lightllm.common.basemodel.triton_kernel.linear_att.replayssm_compact import CompactSSMCache

    torch.manual_seed(17)
    hv, kd, vd, width = 4, 128 if kda else 64, 128 if kda else 64, 3
    state = torch.randn(2, 4, hv, kd, vd, device="cuda", dtype=dtype) * 0.01
    cache = CompactSSMCache(state, width, torch.bfloat16, num_key_heads=hv, kda=kda, run_config=run_config)
    reqs = torch.tensor([2, 0, 3], device="cuda", dtype=torch.int32)
    cu = torch.tensor([0, 3, 5, 5], device="cuda", dtype=torch.int32)
    accepted = torch.tensor([1, 0, 0, 0], device="cuda", dtype=torch.int32)
    shape = (1, 5, hv, kd)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k, v = torch.randn_like(q), torch.randn_like(q)
    a = torch.randn((5, hv * kd if kda else hv), device="cuda", dtype=torch.bfloat16)
    b = torch.randn(5, hv, device="cuda", dtype=torch.bfloat16)
    alog = torch.randn(hv, device="cuda") * 0.1
    bias = torch.randn(hv * kd if kda else hv, device="cuda") * 0.1
    for _ in range(15):
        before = state.clone()
        outputs = [cache.forward(layer, q, k, v, a, b, alog, bias, reqs, cu) for layer in range(2)]
        assert torch.equal(state, before), "verify must not mutate the checkpoint"
        cache.accept_updates(reqs, accepted)
        for layer in range(2):
            for req, start, length in [(2, 0, 3), (0, 3, 2)]:
                cur = before[layer, req].float()
                for j in range(length):
                    t = start + j
                    out, cur = reference(q[0, t], k[0, t], v[0, t], a[t], b[t], alog, bias, cur, kda=kda)
                    torch.testing.assert_close(outputs[layer][0, t].float(), out, atol=0.004, rtol=0.02)
                    cur = cur.to(dtype).float()
                    if j == int(accepted[req]):
                        torch.testing.assert_close(
                            state[layer, req].float(), cur, atol=0.008 if dtype == torch.bfloat16 else 2e-5, rtol=0.02
                        )
        torch.testing.assert_close(state[:, 1], before[:, 1], rtol=0, atol=0)
        torch.testing.assert_close(state[:, 3], before[:, 3], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("width", [3, 4])
def test_compact_matches_native_mtp(dtype, width, run_config=None):
    from lightllm.common.basemodel.triton_kernel.linear_att.replayssm_compact import CompactSSMCache
    from lightllm.common.basemodel.triton_kernel.linear_att.mtp_fused_recurrent import (
        mtp_fused_recurrent_gated_delta_rule,
    )

    torch.manual_seed(77)
    batch, h, hv, kd, vd = 3, 2, 4, 128, 128
    q = torch.randn(1, batch * width, h, kd, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, batch * width, hv, vd, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(batch * width, hv, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(batch * width, hv, device="cuda", dtype=torch.bfloat16)
    alog = torch.randn(hv, device="cuda") * 0.1
    bias = torch.randn(hv, device="cuda") * 0.1
    state = torch.randn(1, batch + 1, hv, kd, vd, device="cuda", dtype=dtype) * 0.01
    reference_state = state[0].repeat_interleave(width, 0)
    reqs = torch.arange(batch, device="cuda", dtype=torch.int32)
    cu = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * width
    idx = torch.arange(batch * width, device="cuda", dtype=torch.int32).view(batch, width)
    counts = torch.ones(batch, device="cuda", dtype=torch.int32)
    accepted = torch.tensor([0, width // 2, width - 1, 0], device="cuda", dtype=torch.int32)
    cache = CompactSSMCache(state, width, torch.bfloat16, num_key_heads=h, run_config=run_config)
    for _ in range(32):
        q.normal_()
        k.normal_()
        v.normal_()
        a.normal_().sub_(3)
        b.normal_()
        out = cache.forward(0, q, k, v, a, b, alog, bias, reqs, cu)
        ref, _ = mtp_fused_recurrent_gated_delta_rule(
            q,
            k,
            v,
            reference_state,
            cu,
            idx,
            idx,
            counts,
            alog,
            bias,
            a,
            b,
            run_config={"num_stages": 1, **(run_config or {"BV": 8, "num_warps": 1})},
        )
        cache.accept_updates(reqs, accepted)
        # BF16 casts can straddle a rounding midpoint across compiler/layouts.
        rtol = 2 * torch.finfo(dtype).eps if dtype == torch.bfloat16 else 0
        torch.testing.assert_close(out, ref, rtol=rtol, atol=2e-5 if dtype == torch.bfloat16 else 0)
        torch.testing.assert_close(
            state[0, :batch],
            reference_state[reqs * width + accepted[:batch]],
            rtol=rtol,
            atol=5e-4 if dtype == torch.bfloat16 else 0,
        )
        counts.copy_(accepted[:batch] + 1)
        accepted[:batch].add_(1).remainder_(width)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("run_config", get_configs(), ids=lambda c: f"bv{c['BV']}-w{c['num_warps']}-s{c['num_stages']}")
@pytest.mark.parametrize("mode", ["gdn", "replay"])
def test_autotune_candidates_match_recurrence(dtype, run_config, mode):
    if mode == "gdn":
        test_compact_matches_native_mtp(dtype, 4, run_config)
    else:
        test_replay_acceptance_flush_and_graph(3, (32, 64), 8, dtype, run_config)


@pytest.mark.parametrize("save_big", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_checkpoint_restores_pending_history_and_accepted_conv(save_big, monkeypatch, dtype):
    from types import SimpleNamespace
    from lightllm.common.req_manager.linear_att import ReqManagerForMamba
    from lightllm.common.state_cache_manager import LinearAttCacheManager, LayerCache
    from lightllm.common.kv_cache_mem_manager.qwen3next_mem_manager import Qwen3NextLinearAttPageHelper

    torch.manual_seed(42)
    state_rtol = 2e-5 if dtype == torch.float32 else torch.finfo(dtype).eps
    # Real save/restore and PD helpers, with just the unrelated allocator omitted.
    manager = object.__new__(ReqManagerForMamba)
    config = SimpleNamespace(
        linear_layer_num=2,
        conv_state_dtype=torch.bfloat16,
        ssm_state_dtype=dtype,
        get_conv_state_shape=lambda: (192, 3),
        get_ssm_state_shape=lambda: (2, 32, 64),
        global_linear_k_heads=1,
        global_linear_v_heads=2,
        head_linear_k_dim=32,
        head_linear_v_dim=64,
        num_linear_k_heads=1,
        num_linear_v_heads=2,
        tp_world_size=1,
        conv_kernel_size=4,
    )
    manager.linear_config = config
    manager.mtp_step = 2
    manager.ssm_slots_per_req = 1
    manager.req_to_mtp_state_index = torch.zeros(4, device="cuda", dtype=torch.int32)
    manager.req_to_mtp_state_index[1] = 1
    manager.req_to_conv_state = LayerCache(4, torch.bfloat16, (192, 5), 2, "cuda")
    manager.req_to_conv_state.buffer.copy_(torch.randn_like(manager.req_to_conv_state.buffer))
    manager.req_to_ssm_state = LayerCache(4, dtype, (2, 32, 64), 2, "cuda")
    manager.req_to_ssm_state.buffer.zero_()
    manager.ssm_update_cache = ReplaySSMCache(manager.req_to_ssm_state.buffer, 16, 3)
    cache = manager.ssm_update_cache
    cache.raw_keys.normal_()
    cache.raw_values.normal_()
    cache.gates.uniform_(-0.1, -0.01)
    cache.betas.uniform_(0.1, 0.9)
    cache.cursors[1] = 2
    active_before = manager.req_to_ssm_state.buffer[:, 1].clone()
    expected = manager.req_to_ssm_state.buffer[:, 1].float().clone()
    for i in range(2):
        key = cache.raw_keys[:, 1, :, i].float()
        key /= (key.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        expected *= cache.gates[:, 1, :, i].exp()[..., None, None]
        delta = cache.betas[:, 1, :, i, None] * (
            cache.raw_values[:, 1, :, i].float() - torch.einsum("lhkv,lhk->lhv", expected, key)
        )
        expected += key[..., None] * delta[..., None, :]
    expected = expected.to(dtype)
    conv_expected = manager.req_to_conv_state.buffer[:, 1, :, 1:4].clone()
    cpu_cache = LinearAttCacheManager(2, config)
    manager.mem_manager = SimpleNamespace(big_page_buffers=cpu_cache)
    reqs = torch.tensor([1], device="cuda", dtype=torch.int32)
    materialized = cache.snapshot_accepted_state(1)
    torch.testing.assert_close(materialized, expected, rtol=state_rtol, atol=2e-5)
    if save_big:
        manager.save_big_page_states(reqs, [1], [0])
    else:
        manager.save_state(1, 0, cpu_cache)
    torch.cuda.synchronize()
    conv_saved, state_saved = cpu_cache.get_state_cache(0)
    # Transfer/restore are exact; only the independent recurrence reference has
    # a tolerance for FP32 reduction differences crossing BF16 rounding points.
    torch.testing.assert_close(state_saved, materialized.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(conv_saved, conv_expected.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(manager.req_to_ssm_state.buffer[:, 1], active_before, rtol=0, atol=0)
    assert cache.cursors[1].item() == 2
    cache.cursors[2] = 13  # stale history from a previous owner must disappear
    manager.restore_state(SimpleNamespace(req_idx=2), cpu_cache, 0)
    assert cache.cursors[2].item() == 0
    assert manager.req_to_mtp_state_index[2].item() == 0
    torch.testing.assert_close(manager.req_to_ssm_state.buffer[:, 2], materialized, rtol=0, atol=0)
    torch.testing.assert_close(manager.req_to_conv_state.buffer[:, 2, :, :3], conv_expected, rtol=0, atol=0)

    import lightllm.common.kv_cache_mem_manager.qwen3next_mem_manager as memory_module

    monkeypatch.setattr(memory_module, "get_env_start_args", lambda: SimpleNamespace(mtp_step=2))
    mem = SimpleNamespace(
        linear_config=config,
        req_to_conv_state=manager.req_to_conv_state,
        req_to_ssm_state=manager.req_to_ssm_state,
        ssm_update_cache=cache,
        ssm_slots_per_req=1,
        req_to_mtp_state_index=manager.req_to_mtp_state_index,
    )
    helper = Qwen3NextLinearAttPageHelper(mem)
    # A second pending suffix exercises PD export, independently of CPU save.
    cache.raw_keys[:, 1, :, 0].normal_()
    cache.raw_values[:, 1, :, 0].normal_()
    cache.gates[:, 1, :, 0].uniform_(-0.1, -0.01)
    cache.betas[:, 1, :, 0].uniform_(0.1, 0.9)
    cache.cursors[1] = 1
    key = cache.raw_keys[:, 1, :, 0].float()
    key /= (key.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    expected = active_before.float().clone()
    expected *= cache.gates[:, 1, :, 0].exp()[..., None, None]
    delta = cache.betas[:, 1, :, 0, None] * (
        cache.raw_values[:, 1, :, 0].float() - torch.einsum("lhkv,lhk->lhv", expected, key)
    )
    expected += key[..., None] * delta[..., None, :]
    conv_page = torch.empty(helper.conv_shape, device="cuda", dtype=torch.bfloat16)
    expected = expected.to(dtype)
    ssm_page = torch.empty(helper.ssm_shape, device="cuda", dtype=dtype)
    helper._write_one_rank(mem, 0, 1, conv_page, ssm_page)
    torch.testing.assert_close(ssm_page, expected, rtol=state_rtol, atol=2e-5)
    torch.testing.assert_close(manager.req_to_ssm_state.buffer[:, 1], active_before, rtol=0, atol=0)
    assert cache.cursors[1].item() == 1
    cache.cursors[2] = 7
    manager.req_to_mtp_state_index[2] = 2
    helper._read_one_rank(mem, 0, 2, conv_page, ssm_page)
    torch.testing.assert_close(manager.req_to_ssm_state.buffer[:, 2], ssm_page, rtol=0, atol=0)
    torch.testing.assert_close(manager.req_to_conv_state.buffer[:, 2, :, :3], conv_expected, rtol=0, atol=0)
    assert cache.cursors[2].item() == manager.req_to_mtp_state_index[2].item() == 0
    manager.init_hybrid_attention_state(SimpleNamespace(req_idx=2))
    assert torch.count_nonzero(manager.req_to_ssm_state.buffer[:, 2]).item() == 0
    assert torch.count_nonzero(manager.req_to_conv_state.buffer[:, 2]).item() == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("capacity,width", [(4, 1), (4, 3), (8, 4), (16, 5), (32, 4), (64, 4)])
@pytest.mark.parametrize("projection_mode", ["inline", "precompute"])
@pytest.mark.parametrize("lower_bound,gate_shift", [(-5.0, 0), (-2.5, -6)])
@pytest.mark.parametrize("activation_dtype", [torch.float32, torch.bfloat16])
def test_kda_replay_reference_fold_and_graph(
    dtype, capacity, width, projection_mode, lower_bound, gate_shift, activation_dtype
):
    test_replay_acceptance_flush_and_graph(
        width,
        (128, 128),
        capacity,
        dtype,
        run_config={"BV": 32, "num_warps": 2, "num_stages": 1},
        projection_mode=projection_mode,
        gate_shift=gate_shift,
        activation_dtype=activation_dtype,
        head_geometry=(2, 2),
        kda=True,
        lower_bound=lower_bound,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_kda_compact_reference_and_partial_acceptance(dtype):
    test_compact_preserves_accepted_prefix(dtype, kda=True)


@pytest.mark.parametrize("config", get_configs())
@pytest.mark.parametrize("mode", ["compact", "inline", "precompute"])
def test_kda_tuning_candidates_match_reference(config, mode):
    if mode == "compact":
        test_compact_preserves_accepted_prefix(torch.bfloat16, run_config=config, kda=True)
    else:
        test_replay_acceptance_flush_and_graph(
            3,
            (128, 128),
            4,
            torch.bfloat16,
            run_config=config,
            projection_mode=mode,
            head_geometry=(2, 2),
            kda=True,
            gate_shift=-3,
        )
