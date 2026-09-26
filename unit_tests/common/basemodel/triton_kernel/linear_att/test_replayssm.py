import pytest
import torch

from lightllm.common.basemodel.triton_kernel.linear_att.replayssm import ReplaySSMCache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def reference(q, k, v, a, b, a_log, bias, state):
    q, k, v = q.float(), k.float(), v.float()
    q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt() * (q.shape[-1] ** -0.5)
    k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    q = q.repeat_interleave(v.shape[-2] // q.shape[-2], -2)
    k = k.repeat_interleave(v.shape[-2] // k.shape[-2], -2)
    g = -a_log.exp() * torch.nn.functional.softplus(a.float() + bias)
    state = state * g.exp()[..., None, None]
    d = b.float().sigmoid()[..., None] * (v - torch.einsum("...kv,...k->...v", state, k))
    state = state + k[..., None] * d[..., None, :]
    return torch.einsum("...kv,...k->...v", state, q), state


@pytest.mark.parametrize("capacity", [16, 32, 64])
@pytest.mark.parametrize("width", [1, 3, 5, 16])
@pytest.mark.parametrize("dims", [(32, 64), (128, 128)])
def test_replay_acceptance_flush_and_graph(width, dims, capacity):
    torch.manual_seed(21)
    kdim, vdim = dims
    h, hv, slots = 2, 4, 5
    state = torch.randn(2, slots, hv, kdim, vdim, device="cuda") * 0.01
    cache = ReplaySSMCache(state, capacity, width)
    expected = state.clone()
    reqs = torch.tensor([2, 0, slots - 1], device="cuda", dtype=torch.int32)
    cu = torch.tensor([0, width, 2 * width - 1, 2 * width - 1], device="cuda", dtype=torch.int32)
    if width == 1:
        cu = torch.tensor([0, 1, 2, 2], device="cuda", dtype=torch.int32)
    tokens = 3 * width
    q = torch.randn(1, tokens, h, kdim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, tokens, hv, vdim, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(tokens, hv, device="cuda", dtype=torch.bfloat16) - 3
    b = torch.randn_like(a)
    alog = torch.randn(hv, device="cuda") * 0.1
    bias = torch.randn(hv, device="cuda") * 0.1
    accepted = torch.zeros(slots, dtype=torch.int32, device="cuda")

    def step():
        pos = cache.positions(reqs)
        out = [cache.forward(layer, q, k, v, a, b, alog, bias, reqs, pos, cu) for layer in range(2)]
        cache.commit(reqs, accepted)
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
        graph.replay()
        for layer in range(2):
            for seq, req in enumerate([2, 0]):
                start = seq * width
                length = width if seq == 0 else max(width - 1, 1)
                cur = expected[layer, req].clone()
                for j in range(length):
                    t = start + j
                    out, cur = reference(q[0, t], k[0, t], v[0, t], a[t], b[t], alog, bias, cur)
                    torch.testing.assert_close(outputs[layer][0, t].float(), out, atol=0.006, rtol=0.02)
                    if j + 1 == counts[seq]:
                        expected[layer, req].copy_(cur)
        if iteration == 2 * capacity + 1:
            cache.materialize(reqs[:2])
            torch.testing.assert_close(state[:, :4], expected[:, :4], atol=2e-5, rtol=2e-4)
    cache.materialize(reqs[:2])
    torch.testing.assert_close(state, expected, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kind", ["gdn", "kda"])
def test_compact_preserves_accepted_prefix(dtype, kind):
    from lightllm.common.basemodel.triton_kernel.linear_att.replayssm_compact import CompactSSMCache

    torch.manual_seed(17)
    hv, kd, vd, width = 4, 64, 64, 3
    state = torch.randn(2, 4, hv, kd, vd, device="cuda", dtype=dtype) * 0.01
    cache = CompactSSMCache(state, width, torch.bfloat16, kind)
    reqs = torch.tensor([2, 0, 3], device="cuda", dtype=torch.int32)
    cu = torch.tensor([0, 3, 5, 5], device="cuda", dtype=torch.int32)
    accepted = torch.tensor([1, 0, 0, 0], device="cuda", dtype=torch.int32)
    shape = (1, 5, hv, kd)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k, v = torch.randn_like(q), torch.randn_like(q)
    a = torch.randn((5, hv * kd if kind == "kda" else hv), device="cuda", dtype=torch.bfloat16)
    b = torch.randn(5, hv, device="cuda", dtype=torch.bfloat16)
    alog = torch.randn(hv, device="cuda") * 0.1
    bias = torch.randn(hv * kd if kind == "kda" else hv, device="cuda") * 0.1
    for _ in range(15):
        before = state.clone()
        outputs = [cache.forward(layer, q, k, v, a, b, alog, bias, reqs, None, cu) for layer in range(2)]
        assert torch.equal(state, before), "verify must not mutate the checkpoint"
        cache.commit(reqs, accepted)
        for layer in range(2):
            for req, start, length in [(2, 0, 3), (0, 3, 2)]:
                cur = before[layer, req].float()
                for j in range(length):
                    t = start + j
                    if kind == "gdn":
                        out, cur = reference(q[0, t], k[0, t], v[0, t], a[t], b[t], alog, bias, cur)
                    else:
                        qq, kk = q[0, t].float(), k[0, t].float()
                        qq *= torch.rsqrt(qq.square().sum(-1, keepdim=True) + 1e-6) * kd**-0.5
                        kk *= torch.rsqrt(kk.square().sum(-1, keepdim=True) + 1e-6)
                        gate = -5 * torch.sigmoid(alog.exp()[:, None] * (a[t].float().view(hv, kd) + bias.view(hv, kd)))
                        cur *= gate.exp()[..., None]
                        d = (v[0, t].float() - torch.einsum("hkv,hk->hv", cur, kk)) * b[t].float().sigmoid()[:, None]
                        cur += kk[..., None] * d[:, None, :]
                        out = torch.einsum("hkv,hk->hv", cur, qq)
                    torch.testing.assert_close(outputs[layer][0, t].float(), out, atol=0.004, rtol=0.02)
                    cur = cur.to(dtype).float()
                    if j == int(accepted[req]):
                        torch.testing.assert_close(
                            state[layer, req].float(), cur, atol=0.008 if dtype == torch.bfloat16 else 2e-5, rtol=0.02
                        )
        torch.testing.assert_close(state[:, 1], before[:, 1], rtol=0, atol=0)
        torch.testing.assert_close(state[:, 3], before[:, 3], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kind", ["gdn", "kda"])
@pytest.mark.parametrize("width", [3, 4])
def test_compact_matches_native_mtp(dtype, kind, width):
    from lightllm.common.basemodel.triton_kernel.linear_att.replayssm_compact import CompactSSMCache
    from lightllm.common.basemodel.triton_kernel.linear_att.mtp_fused_recurrent import (
        mtp_fused_recurrent_gated_delta_rule,
    )

    torch.manual_seed(77)
    batch, h, hv, kd, vd = 3, (4 if kind == "kda" else 2), 4, 128, 128
    q = torch.randn(1, batch * width, h, kd, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, batch * width, hv, vd, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(batch * width, hv * kd if kind == "kda" else hv, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(batch * width, hv, device="cuda", dtype=torch.bfloat16)
    alog = torch.randn(hv, device="cuda") * 0.1
    bias = torch.randn(hv * kd if kind == "kda" else hv, device="cuda") * 0.1
    state = torch.randn(1, batch + 1, hv, kd, vd, device="cuda", dtype=dtype) * 0.01
    reference_state = state[0].repeat_interleave(width, 0)
    reqs = torch.arange(batch, device="cuda", dtype=torch.int32)
    cu = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * width
    idx = torch.arange(batch * width, device="cuda", dtype=torch.int32).view(batch, width)
    counts = torch.ones(batch, device="cuda", dtype=torch.int32)
    accepted = torch.tensor([0, width // 2, width - 1, 0], device="cuda", dtype=torch.int32)
    cache = CompactSSMCache(state, width, torch.bfloat16, kind)
    for _ in range(32):
        q.normal_()
        k.normal_()
        v.normal_()
        a.normal_().sub_(3)
        b.normal_()
        out = cache.forward(0, q, k, v, a, b, alog, bias, reqs, None, cu)
        if kind == "kda":
            native = pytest.importorskip("lightllm.common.basemodel.triton_kernel.linear_att.fla.ops.kda_decode")
            ref, _ = native.fused_recurrent_kda(
                q,
                k,
                v,
                a.unsqueeze(0),
                b.unsqueeze(0),
                alog,
                bias,
                reference_state,
                idx,
                cu_seqlens=cu,
                num_accepted_tokens=counts,
            )
        else:
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
                run_config={"BV": 8, "num_warps": 1, "num_stages": 1},
            )
        cache.commit(reqs, accepted)
        if kind == "gdn":
            assert torch.equal(out, ref)
            atol, rtol = 0, 0
        else:
            # KDA's separate gate storage changes compiler contraction. Across
            # repeated commits this can cross a BF16 rounding boundary.
            torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-2)
            atol, rtol = (1e-7, 2e-5) if dtype == torch.float32 else (1e-5, 1e-2)
        torch.testing.assert_close(
            state[0, :batch], reference_state[reqs * width + accepted[:batch]], rtol=rtol, atol=atol
        )
        counts.copy_(accepted[:batch] + 1)
        accepted[:batch].add_(1).remainder_(width)


@pytest.mark.parametrize("save_big", [False, True])
def test_checkpoint_restores_pending_history_and_accepted_conv(save_big, monkeypatch):
    from types import SimpleNamespace
    from lightllm.common.req_manager.linear_att import ReqManagerForMamba
    from lightllm.common.state_cache_manager import LinearAttCacheManager, LayerCache
    from lightllm.common.kv_cache_mem_manager.qwen3next_mem_manager import Qwen3NextLinearAttPageHelper

    # Real save/restore and PD helpers, with just the unrelated allocator omitted.
    manager = object.__new__(ReqManagerForMamba)
    config = SimpleNamespace(
        linear_layer_num=2,
        conv_state_dtype=torch.bfloat16,
        ssm_state_dtype=torch.float32,
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
    manager.req_to_ssm_state = LayerCache(4, torch.float32, (2, 32, 64), 2, "cuda")
    manager.req_to_ssm_state.buffer.zero_()
    manager.replay_cache = ReplaySSMCache(manager.req_to_ssm_state.buffer, 16, 3)
    cache = manager.replay_cache
    cache.raw_keys.normal_()
    cache.raw_values.normal_()
    cache.gates.uniform_(-0.1, -0.01)
    cache.betas.uniform_(0.1, 0.9)
    cache.cursors[1] = 2
    expected = manager.req_to_ssm_state.buffer[:, 1].clone()
    for i in range(2):
        key = cache.raw_keys[:, 1, :, i].float()
        key /= (key.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        expected *= cache.gates[:, 1, :, i].exp()[..., None, None]
        delta = cache.betas[:, 1, :, i, None] * (
            cache.raw_values[:, 1, :, i].float() - torch.einsum("lhkv,lhk->lhv", expected, key)
        )
        expected += key[..., None] * delta[..., None, :]
    conv_expected = manager.req_to_conv_state.buffer[:, 1, :, 1:4].clone()
    cpu_cache = LinearAttCacheManager(2, config)
    manager.mem_manager = SimpleNamespace(big_page_buffers=cpu_cache)
    reqs = torch.tensor([1], device="cuda", dtype=torch.int32)
    if save_big:
        manager.save_big_page_states(reqs, [1], [0])
    else:
        manager.save_state(1, 0, cpu_cache)
    torch.cuda.synchronize()
    conv_saved, state_saved = cpu_cache.get_state_cache(0)
    torch.testing.assert_close(state_saved, expected.cpu(), rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(conv_saved, conv_expected.cpu(), rtol=0, atol=0)
    cache.cursors[2] = 13  # stale history from a previous owner must disappear
    manager.restore_state(SimpleNamespace(req_idx=2), cpu_cache, 0)
    assert cache.cursors[2].item() == 0
    assert manager.req_to_mtp_state_index[2].item() == 0
    torch.testing.assert_close(manager.req_to_ssm_state.buffer[:, 2], expected, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(manager.req_to_conv_state.buffer[:, 2, :, :3], conv_expected, rtol=0, atol=0)

    import lightllm.common.kv_cache_mem_manager.qwen3next_mem_manager as memory_module

    monkeypatch.setattr(memory_module, "get_env_start_args", lambda: SimpleNamespace(mtp_step=2))
    mem = SimpleNamespace(
        linear_config=config,
        req_to_conv_state=manager.req_to_conv_state,
        req_to_ssm_state=manager.req_to_ssm_state,
        replay_cache=cache,
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
    expected *= cache.gates[:, 1, :, 0].exp()[..., None, None]
    delta = cache.betas[:, 1, :, 0, None] * (
        cache.raw_values[:, 1, :, 0].float() - torch.einsum("lhkv,lhk->lhv", expected, key)
    )
    expected += key[..., None] * delta[..., None, :]
    conv_page = torch.empty(helper.conv_shape, device="cuda", dtype=torch.bfloat16)
    ssm_page = torch.empty(helper.ssm_shape, device="cuda")
    helper._write_one_rank(mem, 0, 1, conv_page, ssm_page)
    cache.cursors[2] = 7
    manager.req_to_mtp_state_index[2] = 2
    helper._read_one_rank(mem, 0, 2, conv_page, ssm_page)
    torch.testing.assert_close(manager.req_to_ssm_state.buffer[:, 2], expected, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(manager.req_to_conv_state.buffer[:, 2, :, :3], conv_expected, rtol=0, atol=0)
    assert cache.cursors[2].item() == manager.req_to_mtp_state_index[2].item() == 0
    manager.init_hybrid_attention_state(SimpleNamespace(req_idx=2))
    assert torch.count_nonzero(manager.req_to_ssm_state.buffer[:, 2]).item() == 0
    assert torch.count_nonzero(manager.req_to_conv_state.buffer[:, 2]).item() == 0
