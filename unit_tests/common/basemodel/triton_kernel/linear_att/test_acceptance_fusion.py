import pytest
import torch
import triton

from lightllm.common.basemodel.triton_kernel.mtp_utils import linear_att_mtp_state_index_update as update


def ReplaySSMExecutor(*args, **kwargs):
    kda = kwargs.get("kda", False)
    if kda:
        from lightllm.common.attention_state.bounded_att.kda.replay_kda_runtime import ReplayKDARuntime

        return ReplayKDARuntime(*args, **kwargs)
    from lightllm.common.attention_state.bounded_att.gdn.replay_gdn_runtime import ReplayGDNRuntime

    return ReplayGDNRuntime(*args, **kwargs)


from types import SimpleNamespace
from lightllm.common.attention_state.commit import StateAcceptance
from lightllm.common.attention_state.bounded_att.gdn.replay_gdn_runtime import ReplayGDNRuntime
from lightllm.common.basemodel.triton_kernel.linear_att.ssm_acceptance import commit_ssm_prefix
from lightllm.common.basemodel.triton_kernel.linear_att.ssm_autotune import replay_layer


def make(n, width):
    hold = n + 3
    ids = list(reversed(range(n))) + [hold]
    lengths = [1 + i % width for i in range(n)] + [1]
    starts = []
    reqs = []
    mtps = []
    for r, l in zip(ids, lengths):
        starts.append(len(reqs))
        reqs.extend([r] * l)
        mtps.extend(range(l))
    tensor = lambda x: torch.tensor(x, device="cuda", dtype=torch.int32)
    args = [tensor(starts), tensor(reqs), tensor(mtps), tensor([1] * len(reqs)), width]
    ssm_backend = ReplayGDNRuntime.__new__(ReplayGDNRuntime)
    ssm_backend.cursors = tensor([7] * (hold + 1))
    ssm_backend.hold_req_id = hold
    ssm_backend.capacity = max(4, triton.next_power_of_2(width))
    ssm_backend.verify_width = width
    manager = SimpleNamespace()
    manager.mtp_state_index = tensor([-9] * (hold + 1))
    manager._ssm_executor = ssm_backend
    return manager, args, (ids, lengths, starts)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("replay", [False, True])
def test_request_counts_share_compiled_kernel(monkeypatch, replay):
    from lightllm.common.basemodel.triton_kernel.linear_att.ssm_acceptance import _commit_prefix, commit_ssm_prefix

    reqs = torch.arange(512, dtype=torch.int32, device="cuda")
    lengths = reqs % 4 + 1
    indices = torch.zeros(513, dtype=torch.int32, device="cuda")
    cursors = torch.zeros_like(indices) if replay else None
    hashes = set()
    run = _commit_prefix.run

    def record(*args, **kwargs):
        kernel = run(*args, **kwargs)
        hashes.add(kernel.hash)
        return kernel

    monkeypatch.setattr(_commit_prefix, "run", record)
    for count in (1, 2, 7, 16, 23, 24, 25, 31, 32, 33, 255, 256, 257, 511):
        indices.fill_(-1)
        if replay:
            cursors.zero_()
        commit_ssm_prefix(indices, reqs[:count], lengths[:count], cursors)
        assert torch.equal(indices[:count], lengths[:count] - 1)
        assert (indices[count:] == -1).all()
        if replay:
            assert torch.equal(cursors[:count], lengths[:count])
            assert not cursors[count:].any()
    assert len(hashes) == 1


def accept_unfused(m, args):
    update(m.mtp_state_index, *args)
    m._ssm_executor.accept_updates(args[1][args[0].long()], m.mtp_state_index)


def accept_fused(m, args):
    reqs = args[1][args[0].long()]
    ends = torch.cat((args[0][1:], args[0].new_full((1,), args[3].numel())))
    cumulative = torch.cat((args[3].new_zeros(1), args[3].cumsum(0)))
    lengths = cumulative[ends.long()] - cumulative[args[0].long()]
    m._ssm_executor.commit(StateAcceptance(reqs, lengths), m.mtp_state_index, "main")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_acceptance_fusion_matches_original_and_cpu_under_graph():
    checks = 0
    for width in [1, 2, 4, 8, 16]:
        for n in [1, 7, 33, 128]:
            m, args, (ids, lengths, starts) = make(n, width)
            ref, _, _ = make(n, width)
            # Compile both paths before graph capture, then reset before replay.
            accept_unfused(ref, args)
            accept_fused(m, args)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                accept_fused(m, args)
            expected_index = [-9] * len(m.mtp_state_index)
            expected_cursor = [7] * len(expected_index)
            m.mtp_state_index.fill_(-9)
            ref.mtp_state_index.fill_(-9)
            m._ssm_executor.cursors.fill_(7)
            ref._ssm_executor.cursors.fill_(7)
            for step in range(24):
                mask = []
                for i, (r, l) in enumerate(zip(ids, lengths)):
                    count = (step + i) % (l + 1)
                    mask.extend([1] * count + [0] * (l - count))
                    expected_index[r] = count - 1
                    if r != m._ssm_executor.hold_req_id:
                        expected_cursor[r] += count if width > 1 else 1
                args[3].copy_(torch.tensor(mask, device="cuda", dtype=torch.int32))
                accept_unfused(ref, args)
                if step % 2:
                    graph.replay()
                else:
                    accept_fused(m, args)
                assert torch.equal(m.mtp_state_index, ref.mtp_state_index)
                assert torch.equal(m._ssm_executor.cursors, ref._ssm_executor.cursors)
                assert m.mtp_state_index.tolist() == expected_index
                assert m._ssm_executor.cursors.tolist() == expected_cursor
                checks += 1

            class _IndexOnly:
                def commit(self, acceptance, indices, state_name=None):
                    commit_ssm_prefix(indices, acceptance.req_indices, acceptance.accepted_lengths)

            m._ssm_executor = _IndexOnly()
            accept_fused(m, args)
            assert m.mtp_state_index.tolist() == expected_index

    print("ACCEPT_EQUIVALENCE_CHECKS=" + str(checks), flush=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("compute_mode", ["inline", "precompute"])
def test_fused_acceptance_preserves_folds_snapshots_and_slot_reuse(dtype, compute_mode):
    torch.manual_seed(93)
    initial = torch.randn(2, 8, 4, 32, 32, device="cuda").to(dtype) * 0.01
    ssm_backends = [
        ReplaySSMExecutor(
            initial.clone(),
            8,
            4,
            torch.bfloat16,
            num_key_heads=2,
            compute_mode=compute_mode,
            run_config={"BV": 32, "num_warps": 1, "num_stages": 1},
        )
        for _ in range(2)
    ]
    managers = []
    for ssm_backend in ssm_backends:
        manager = SimpleNamespace()
        manager._ssm_executor = ssm_backend
        manager.mtp_state_index = torch.zeros(8, dtype=torch.int32, device="cuda")
        managers.append(manager)
    reqs = torch.full((4,), 7, dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, 3, 6, 9, 12], dtype=torch.int32, device="cuda")
    token_reqs = torch.full((12,), 7, dtype=torch.int32, device="cuda")
    mtp_indexes = torch.arange(12, dtype=torch.int32, device="cuda") % 3
    accepted_mask = torch.ones(12, dtype=torch.int32, device="cuda")
    q = torch.randn(1, 12, 2, 32, dtype=torch.bfloat16, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn(1, 12, 4, 32, dtype=torch.bfloat16, device="cuda")
    a = torch.full((12, 4), -3.0, dtype=torch.bfloat16, device="cuda")
    b = torch.randn_like(a)
    alog = torch.zeros(4, device="cuda")
    args = [cu[:-1], token_reqs, mtp_indexes, accepted_mask, 4]

    def cycle(i):
        ssm_backend = ssm_backends[i]
        ssm_backend.prepare_decode(reqs, cu)
        outputs = [replay_layer(ssm_backend, layer, q, k, v, a, b, alog, alog, reqs, cu)[0] for layer in range(2)]
        (accept_unfused if i == 0 else accept_fused)(managers[i], args)
        return outputs

    graphs, outputs = [], []
    for i in range(2):
        cycle(i)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = cycle(i)
        graphs.append(graph)
        outputs.append(output)
    for iteration in range(40):
        ids = [1, 3, 6] if iteration % 2 else [6, 1, 3]
        lengths = [4, 4, 3] if iteration % 3 == 0 else [3, 3, 3]
        lengths.append(12 - sum(lengths))
        ids.append(7)
        starts, tokens, indexes, masks = [0], [], [], []
        for i, (req, length) in enumerate(zip(ids, lengths)):
            count = (iteration + i) % (length + 1) if req != 7 else 0
            tokens.extend([req] * length)
            indexes.extend(range(length))
            masks.extend([1] * count + [0] * (length - count))
            starts.append(len(tokens))
        for tensor, values in [
            (reqs, ids),
            (cu, starts),
            (token_reqs, tokens),
            (mtp_indexes, indexes),
            (accepted_mask, masks),
        ]:
            tensor.copy_(tensor.new_tensor(values))
        for tensor in (q, k, v, b):
            tensor.normal_()
        for graph in graphs:
            graph.replay()
        for left, right in zip(outputs[0], outputs[1]):
            assert torch.equal(left, right)
        assert torch.equal(ssm_backends[0].state, ssm_backends[1].state)
        assert torch.equal(ssm_backends[0].cursors, ssm_backends[1].cursors)
        assert torch.equal(managers[0].mtp_state_index, managers[1].mtp_state_index)
        if iteration % 7 == 0:
            assert torch.equal(ssm_backends[0].get_accepted_state(3), ssm_backends[1].get_accepted_state(3))
        if iteration % 11 == 10:
            for ssm_backend in ssm_backends:
                ssm_backend.merge_accepted_updates(reqs[:3])
            assert torch.equal(ssm_backends[0].state, ssm_backends[1].state)
        if iteration == 19:
            for manager in managers:
                manager._ssm_executor.clear_history(1)
                manager._ssm_executor.state[:, 1].zero_()
                manager.mtp_state_index[1] = 0
    for ssm_backend in ssm_backends:
        ssm_backend.merge_accepted_updates(reqs[:3])
    assert torch.equal(ssm_backends[0].state, ssm_backends[1].state)
    assert torch.equal(ssm_backends[0].state[:, [0, 2, 4, 5, 7]], initial[:, [0, 2, 4, 5, 7]])
