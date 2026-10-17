import pytest
import torch

from lightllm.common.basemodel.triton_kernel.linear_att.replayssm import ReplaySSMCache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("capacity", [8, 16, 64])
def test_derived_fold_commits_only_accepted_history(capacity):
    torch.manual_seed(731)
    state = torch.randn(2, 5, 4, 128, 128, device="cuda", dtype=torch.bfloat16) * 0.02
    initial = state.clone()
    cache = ReplaySSMCache(state, capacity, 5, num_key_heads=2, projection_mode="parallel", fold_method="derived")
    cache.keys.fill_(float("nan"))
    cache.deltas.fill_(float("nan"))
    cache.gates.fill_(float("nan"))
    cache.raw_keys.fill_(float("nan"))
    cache.raw_values.fill_(float("nan"))
    cache.betas.fill_(float("nan"))
    # Fold request 2 from the second ring half; request 0 has an empty span.
    count = capacity - 2
    cache.cursors[2] = 2 * capacity + count
    cache.cursors[0] = count
    cache.keys[:, 2, :, capacity : capacity + count].normal_(0, 0.1)
    cache.deltas[:, 2, :, capacity : capacity + count].normal_(0, 0.1)
    cache.gates[:, 2, :, capacity : capacity + count].fill_(-0.1)
    expected = initial[:, 2].float()
    for j in range(count):
        keys = cache.keys[:, 2, :, capacity + j].repeat_interleave(2, dim=1)
        delta = cache.deltas[:, 2, :, capacity + j]
        decay = cache.gates[:, 2, :, capacity + j].exp()
        expected = expected * decay[:, :, None, None] + keys[:, :, :, None] * delta[:, :, None, :]
    reqs = torch.tensor([2, 0, 4], dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, 5, 5, 10], dtype=torch.int32, device="cuda")
    cache.prepare_decode(reqs, cu)
    assert torch.isfinite(state).all()
    torch.testing.assert_close(state[:, 2], expected.bfloat16(), atol=2e-4, rtol=0.008)
    torch.testing.assert_close(state[:, [0, 1, 3, 4]], initial[:, [0, 1, 3, 4]], atol=0, rtol=0)
    assert cache.cursors[2] == 0
    assert cache.cursors[0] == count
    assert cache.cursors[4] == 0


@pytest.mark.parametrize("fold", ["raw", "derived"])
def test_parallel_rejects_unsupported_shapes(fold):
    state = torch.zeros(1, 3, 4, 32, 64, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="128x128"):
        ReplaySSMCache(state, 16, 5, num_key_heads=2, projection_mode="parallel", fold_method=fold)


def test_derived_fold_requires_parallel_solver():
    state = torch.zeros(1, 3, 4, 128, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(AssertionError):
        ReplaySSMCache(state, 16, 5, num_key_heads=2, fold_method="derived")


@pytest.mark.parametrize("width", [2, 3, 5, 8, 16])
def test_parallel_raw_against_sequential_reference(width):
    from unit_tests.common.basemodel.triton_kernel.linear_att.test_replayssm import (
        test_replay_acceptance_flush_and_graph as check_replay,
    )

    check_replay(
        width=width,
        dims=(128, 128),
        capacity=16,
        dtype=torch.bfloat16,
        projection_mode="parallel",
        run_config={"BV": 128, "num_warps": 2, "num_stages": 1},
    )


@pytest.mark.parametrize("capacity,width", [(8, 3), (16, 5), (64, 16)])
@pytest.mark.parametrize("capture", [False, True], ids=["eager", "graph"])
def test_parallel_derived_continuous_state(capacity, width, capture):
    from unit_tests.common.basemodel.triton_kernel.linear_att.test_replayssm import reference

    torch.manual_seed(731)
    state = torch.randn(2, 4, 4, 128, 128, device="cuda", dtype=torch.bfloat16) * 0.02
    initial = state.clone()
    cache = ReplaySSMCache(state, capacity, width, num_key_heads=2, projection_mode="parallel", fold_method="derived")
    for history in (cache.keys, cache.deltas, cache.gates, cache.raw_keys, cache.raw_values, cache.betas):
        history.fill_(float("nan"))
    reqs = torch.tensor([2, 0, 3], device="cuda", dtype=torch.int32)
    lengths = [width, width - 1]
    cu = torch.tensor([0, width, 2 * width - 1, 2 * width], device="cuda", dtype=torch.int32)
    q = torch.randn(1, 2 * width, 2, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, 2 * width, 4, 128, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(2 * width, 4, device="cuda", dtype=torch.bfloat16) - 3
    b = torch.randn_like(a)
    alog = torch.randn(4, device="cuda") * 0.1
    bias = torch.randn_like(alog) * 0.1
    accepted = torch.zeros(4, device="cuda", dtype=torch.int32)

    def step():
        cache.prepare_decode(reqs, cu)
        outputs = [cache.forward(layer, q, k, v, a, b, alog, bias, reqs, cu) for layer in range(2)]
        cache.accept_updates(reqs, accepted)
        return outputs

    # Compile before capture, then reset the live state and capture only HOLD rows.
    step()
    state.copy_(initial)
    cache.cursors.zero_()
    if capture:
        reqs.fill_(cache.hold)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = step()
        reqs.copy_(torch.tensor([2, 0, 3], device="cuda", dtype=torch.int32))

    expected = initial.float()
    pending = [0, 0]
    folds = [0, 0]
    for iteration in range(3 * capacity + 5):
        for tensor in (q, k, v, b):
            tensor.normal_()
        a.normal_(-3, 1)
        counts = [iteration % width + 1, (iteration + 1) % (width - 1) + 1]
        for seq, req in enumerate((2, 0)):
            accepted[req] = counts[seq] - 1
            if pending[seq] + lengths[seq] > capacity:
                expected[:, req] = expected[:, req].bfloat16().float()
                pending[seq] = 0
                folds[seq] += 1
            pending[seq] += counts[seq]
        if capture:
            graph.replay()
        else:
            outputs = step()
        for layer in range(2):
            for seq, req in enumerate((2, 0)):
                current = expected[layer, req].clone()
                for j in range(lengths[seq]):
                    token = seq * width + j
                    output, current = reference(
                        q[0, token], k[0, token], v[0, token], a[token], b[token], alog, bias, current
                    )
                    torch.testing.assert_close(outputs[layer][0, token].float(), output, atol=0.006, rtol=0.02)
                    if j + 1 == counts[seq]:
                        expected[layer, req].copy_(current)
            assert torch.count_nonzero(outputs[layer][0, -1]) == 0

        if iteration == capacity:
            saved_state, saved_cursors = state.clone(), cache.cursors.clone()
            snapshot = cache.snapshot_accepted_state(2)
            torch.testing.assert_close(snapshot.float(), expected[:, 2], atol=0.002, rtol=0.02)
            torch.testing.assert_close(state, saved_state, atol=0, rtol=0)
            torch.testing.assert_close(cache.cursors, saved_cursors, atol=0, rtol=0)
            # Restore the canonical checkpoint while another request keeps its history.
            cache.clear_history(2)
            state[:, 2].copy_(snapshot)
            expected[:, 2].copy_(snapshot.float())
            pending[0] = 0

    assert min(folds) >= 2  # Exercise both ring halves, including after restore.
    cache.merge_accepted_updates(reqs[:2])
    torch.testing.assert_close(state.float(), expected.bfloat16().float(), atol=0.002, rtol=0.02)
    torch.testing.assert_close(state[:, [1, 3]], initial[:, [1, 3]], atol=0, rtol=0)
    assert torch.count_nonzero(cache.cursors) == 0
