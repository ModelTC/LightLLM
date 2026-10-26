import pytest
import torch
import triton

from lightllm.common.basemodel.triton_kernel.mtp_utils import linear_att_mtp_state_index_update as update
from lightllm.common.basemodel.triton_kernel.linear_att.replayssm import ReplaySSMBackend
from lightllm.common.linear_state import LinearStateManager
from lightllm.common.req_manager.linear_att import ReqManagerForMamba


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
    ssm_backend = ReplaySSMBackend.__new__(ReplaySSMBackend)
    ssm_backend.cursors = tensor([7] * (hold + 1))
    ssm_backend.hold_req_id = hold
    ssm_backend.capacity = max(4, triton.next_power_of_2(width))
    ssm_backend.verify_width = width
    manager = ReqManagerForMamba.__new__(ReqManagerForMamba)
    manager.linear_state_manager = object.__new__(LinearStateManager)
    manager.linear_state_manager.accepted_state_index = tensor([-9] * (hold + 1))
    manager.linear_state_manager.ssm_backend = ssm_backend
    return manager, args, (ids, lengths, starts)


def accept_unfused(m, args):
    update(m.linear_state_manager.accepted_state_index, *args)
    m.linear_state_manager.ssm_backend.accept_updates(
        args[1][args[0].long()], m.linear_state_manager.accepted_state_index
    )


def accept_fused(m, args):
    m.update_mtp_state(*args)


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
            expected_index = [-9] * len(m.linear_state_manager.accepted_state_index)
            expected_cursor = [7] * len(expected_index)
            m.linear_state_manager.accepted_state_index.fill_(-9)
            ref.linear_state_manager.accepted_state_index.fill_(-9)
            m.linear_state_manager.ssm_backend.cursors.fill_(7)
            ref.linear_state_manager.ssm_backend.cursors.fill_(7)
            for step in range(24):
                mask = []
                for i, (r, l) in enumerate(zip(ids, lengths)):
                    count = (step + i) % (l + 1)
                    mask.extend([1] * count + [0] * (l - count))
                    expected_index[r] = count - 1
                    if r != m.linear_state_manager.ssm_backend.hold_req_id:
                        expected_cursor[r] += count if width > 1 else 1
                args[3].copy_(torch.tensor(mask, device="cuda", dtype=torch.int32))
                accept_unfused(ref, args)
                if step % 2:
                    graph.replay()
                else:
                    accept_fused(m, args)
                assert torch.equal(
                    m.linear_state_manager.accepted_state_index, ref.linear_state_manager.accepted_state_index
                )
                assert torch.equal(
                    m.linear_state_manager.ssm_backend.cursors, ref.linear_state_manager.ssm_backend.cursors
                )
                assert m.linear_state_manager.accepted_state_index.tolist() == expected_index
                assert m.linear_state_manager.ssm_backend.cursors.tolist() == expected_cursor
                checks += 1
            # Native optional-pointer path and manager dispatch leave metadata-only backend absent.
            m.linear_state_manager.ssm_backend = None
            accept_fused(m, args)
            assert m.linear_state_manager.accepted_state_index.tolist() == expected_index

    # Non-Replay backends must still receive the accepted request IDs and indexes.
    class CompactProbe:
        def accept_updates(self, reqs, index):
            self.seen = (reqs.clone(), index.clone())

    m, args, (ids, _, _) = make(7, 4)
    probe = CompactProbe()
    m.linear_state_manager.ssm_backend = probe
    accept_fused(m, args)
    assert probe.seen[0].tolist() == ids
    assert torch.equal(probe.seen[1], m.linear_state_manager.accepted_state_index)
    print("ACCEPT_EQUIVALENCE_CHECKS=" + str(checks), flush=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("compute_mode", ["inline", "precompute"])
def test_fused_acceptance_preserves_folds_snapshots_and_slot_reuse(dtype, compute_mode):
    torch.manual_seed(93)
    initial = torch.randn(2, 8, 4, 32, 32, device="cuda").to(dtype) * 0.01
    ssm_backends = [
        ReplaySSMBackend(
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
        manager = ReqManagerForMamba.__new__(ReqManagerForMamba)
        manager.linear_state_manager = object.__new__(LinearStateManager)
        manager.linear_state_manager.ssm_backend = ssm_backend
        manager.linear_state_manager.accepted_state_index = torch.zeros(8, dtype=torch.int32, device="cuda")
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
        outputs = [ssm_backend.forward(layer, q, k, v, a, b, alog, alog, reqs, cu)[0] for layer in range(2)]
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
        assert torch.equal(
            managers[0].linear_state_manager.accepted_state_index, managers[1].linear_state_manager.accepted_state_index
        )
        if iteration % 7 == 0:
            assert torch.equal(ssm_backends[0].get_accepted_state(3), ssm_backends[1].get_accepted_state(3))
        if iteration % 11 == 10:
            for ssm_backend in ssm_backends:
                ssm_backend.merge_accepted_updates(reqs[:3])
            assert torch.equal(ssm_backends[0].state, ssm_backends[1].state)
        if iteration == 19:
            for manager in managers:
                manager.linear_state_manager.ssm_backend.clear_history(1)
                manager.linear_state_manager.ssm_backend.state[:, 1].zero_()
                manager.linear_state_manager.accepted_state_index[1] = 0
    for ssm_backend in ssm_backends:
        ssm_backend.merge_accepted_updates(reqs[:3])
    assert torch.equal(ssm_backends[0].state, ssm_backends[1].state)
    assert torch.equal(ssm_backends[0].state[:, [0, 2, 4, 5, 7]], initial[:, [0, 2, 4, 5, 7]])
