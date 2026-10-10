import pytest
import torch

from lightllm.common.basemodel.triton_kernel.linear_att.replayssm import ReplaySSMCache
from lightllm.common.basemodel.triton_kernel.linear_att.mtp_state_params import (
    build_dynamic_mtp_linear_att_state_params,
)
from lightllm.common.req_manager.linear_att import ReqManagerForMamba

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("capture_metadata", [False, True])
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("kda", [False, True], ids=["gdn", "kda"])
def test_request_manager_accept_with_replay_graph(dtype, capture_metadata, varlen, kda):
    torch.manual_seed(827)
    config = {"BV": 32, "num_warps": 1, "num_stages": 1}
    initial = torch.randn(3, 5, 4, 128, 128, device="cuda", dtype=dtype) * 0.01
    caches = [
        cls(
            initial.clone(),
            8,
            4,
            torch.bfloat16,
            num_key_heads=2,
            kda=kda,
            projection_mode="precompute",
            run_config=config,
        )
        for cls in [ReplaySSMCache, ReplaySSMCache]
    ]
    manager = object.__new__(ReqManagerForMamba)
    manager.ssm_update_cache = caches[1]
    manager.req_to_mtp_state_index = torch.zeros(5, device="cuda", dtype=torch.int32)
    rows = torch.full((12,), 4, device="cuda", dtype=torch.int32)
    mtp = torch.zeros_like(rows)
    flags = torch.zeros_like(rows)
    starts = torch.zeros(3, device="cuda", dtype=torch.int32)
    accepted = torch.full((5,), -1, device="cuda", dtype=torch.int32)
    q = torch.randn(1, 12, 2, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, 12, 4, 128, device="cuda", dtype=torch.bfloat16)
    a = torch.full((12, 4 * 128 if kda else 4), -3.0, device="cuda", dtype=torch.bfloat16)
    beta = torch.randn(a.shape[0], 4, device="cuda", dtype=a.dtype)
    alog = torch.zeros(4, device="cuda")
    bias = torch.zeros(4 * 128 if kda else 4, device="cuda")

    def inputs(iteration):
        ids = [0, 1, 3] if iteration % 2 == 0 else [3, 0, 1]
        lengths = [1 + (iteration + j) % 4 for j in range(3)] if varlen else [4] * 3
        counts = [1 + (iteration // 2 + j) % length for j, length in enumerate(lengths)]
        offsets, req_rows, indexes, mask = [], [], [], []
        table = [-1] * 5
        for req, length, count in zip(ids, lengths, counts):
            offsets.append(len(req_rows))
            req_rows.extend([req] * length)
            indexes.extend(range(length))
            mask.extend([int(j < count) for j in range(length)])
            table[req] = count - 1
        logical = len(req_rows)
        req_rows.extend([4] * (12 - logical))
        indexes.extend([0] * (12 - logical))
        mask.extend([0] * (12 - logical))
        rows.copy_(rows.new_tensor(req_rows))
        mtp.copy_(mtp.new_tensor(indexes))
        flags.copy_(flags.new_tensor(mask))
        starts.copy_(starts.new_tensor(offsets))
        accepted.copy_(accepted.new_tensor(table))
        return ids, lengths, counts, logical

    def metadata(cache):
        cu, reqs, _ = build_dynamic_mtp_linear_att_state_params(rows, mtp, manager.req_to_mtp_state_index, 4)
        return cu, reqs, cache.prepare_decode(reqs, cu)

    def forward_accept(index, cu, reqs):
        cache = caches[index]
        out = [cache.forward(layer, q, k, v, a, beta, alog, bias, reqs, cu) for layer in range(3)]
        if index == 0:
            cache.accept_updates(reqs, accepted)
        else:
            manager.update_mtp_state(starts, rows, mtp, flags, 4)
        return out

    inputs(0)
    graphs, captured, outputs = [], [], []
    for index, cache in enumerate(caches):
        cu, reqs, pos = metadata(cache)
        forward_accept(index, cu, reqs)
        cache.state.copy_(initial)
        cache.cursors.zero_()
        manager.req_to_mtp_state_index.zero_()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            if capture_metadata:
                cu, reqs, pos = metadata(cache)
            out = forward_accept(index, cu, reqs)
        graphs.append(g)
        captured.append((cu, reqs, pos))
        outputs.append(out)
    for cache in caches:
        cache.state.copy_(initial)
        cache.cursors.zero_()
    manager.req_to_mtp_state_index.zero_()
    folds = 0
    for iteration in range(32):
        ids, lengths, counts, logical = inputs(iteration)
        for tensor in [q, k, v, beta]:
            tensor.normal_()
        for index, cache in enumerate(caches):
            if not capture_metadata:
                for dst, src in zip(captured[index], metadata(cache)):
                    dst.copy_(src)
            graphs[index].replay()
        torch.cuda.synchronize()
        for expected, actual in zip(captured[0], captured[1]):
            assert torch.equal(actual, expected)
        assert torch.equal(caches[0].cursors, caches[1].cursors)
        positions = captured[0][2].cpu().tolist()
        folds += sum(positions[j] % 16 + length > 8 for j, length in enumerate(lengths))
        for expected, actual in zip(outputs[0], outputs[1]):
            torch.testing.assert_close(
                actual[:, :logical].float(), expected[:, :logical].float(), atol=0.006, rtol=0.02
            )
        torch.testing.assert_close(
            caches[1].state.float(), caches[0].state.float(), atol=0.002, rtol=torch.finfo(dtype).eps
        )
        table = manager.req_to_mtp_state_index.cpu().tolist()
        for req, count in zip(ids, counts):
            assert table[req] == count - 1
        assert table[2] == table[4] == 0
        assert caches[1].cursors[4].item() == 0
    assert folds > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("capture_metadata", [False, True])
@pytest.mark.parametrize("stagger", [False, True])
@pytest.mark.parametrize("kda", [False, True], ids=["gdn", "kda"])
def test_decode_accept_order_preserves_forward_snapshot(dtype, capture_metadata, stagger, kda):
    torch.manual_seed(731)
    initial = torch.randn(3, 3, 4, 128, 128, device="cuda", dtype=dtype) * 0.01
    caches = [
        ReplaySSMCache(
            initial.clone(),
            8,
            1,
            dtype,
            num_key_heads=2,
            kda=kda,
            run_config={"BV": 32, "num_warps": 1, "num_stages": 1},
        )
        for _ in range(2)
    ]
    for cache in caches:
        cache.fold_programs = 3
    reqs = torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32)
    q = torch.randn(3, 2, 128, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn(3, 4, 128, device="cuda", dtype=dtype)
    a = torch.full((3, 4 * 128 if kda else 4), -3.0, device="cuda", dtype=dtype)
    beta = torch.randn(a.shape[0], 4, device="cuda", dtype=a.dtype)
    alog = torch.zeros(4, device="cuda")
    bias = torch.zeros(4 * 128 if kda else 4, device="cuda")

    def forward(cache):
        return [cache.forward(layer, q, k, v, a, beta, alog, bias, reqs) for layer in range(3)]

    def warm(cache):
        cache.prepare_decode(reqs)
        out = forward(cache)
        cache.accept_updates(reqs)
        return out

    graphs, positions, outputs = [], [], []
    for early, cache in enumerate(caches):
        warm(cache)
        cache.state.copy_(initial)
        cache.cursors.zero_()
        if stagger:
            reqs.copy_(reqs.new_tensor([0, 2, 2]))
            for _ in range(3):
                warm(cache)
            reqs.copy_(reqs.new_tensor([0, 1, 2]))
        pos = torch.zeros_like(reqs)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            if capture_metadata:
                pos = cache.prepare_decode(reqs)
                if early:
                    cache.accept_updates(reqs)
            out = forward(cache)
            if capture_metadata and not early:
                cache.accept_updates(reqs)
        graphs.append(graph)
        positions.append(pos)
        outputs.append(out)
    saw_fold = saw_nonfold = False
    for iteration in range(24):
        reqs.copy_(reqs.new_tensor([0, 1, 2] if iteration % 2 == 0 else [1, 0, 2]))
        for tensor in [q, k, v, beta]:
            tensor.normal_()
        old = caches[0].cursors[reqs.long()].clone()
        for early, (cache, graph, pos) in enumerate(zip(caches, graphs, positions)):
            if not capture_metadata:
                pos.copy_(cache.prepare_decode(reqs))
                if early:
                    cache.accept_updates(reqs)
            graph.replay()
            if not capture_metadata and not early:
                cache.accept_updates(reqs)
        torch.cuda.synchronize()
        assert torch.equal(positions[0], old)
        assert torch.equal(positions[0], positions[1])
        assert torch.equal(caches[0].cursors, caches[1].cursors)
        for expected, actual in zip(outputs[0], outputs[1]):
            torch.testing.assert_close(actual.float(), expected.float(), atol=0.006, rtol=0.02)
        torch.testing.assert_close(
            caches[1].state.float(), caches[0].state.float(), atol=0.002, rtol=torch.finfo(dtype).eps
        )
        folded = any(n % 16 + 1 > 8 for n in old[:2].cpu().tolist())
        saw_fold |= folded
        saw_nonfold |= not folded
    assert saw_fold and saw_nonfold


def test_cross_layer_fold_with_state_offset_above_int32():
    if torch.cuda.mem_get_info()[0] < 6 * 1024 ** 3:
        pytest.skip("large-stride regression requires 6 GiB of free device memory")
    torch.manual_seed(791)
    shape = (48, 3, 12, 128, 128)
    strides = (50_528_256, 196608, 16384, 128, 1)
    assert strides[0] < 2 ** 31 < 47 * strides[0]
    dense = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.01
    sparse = torch.empty_strided(shape, strides, device="cuda", dtype=dense.dtype)
    sparse.copy_(dense)
    config = {"BV": 32, "num_warps": 1, "num_stages": 1}
    caches = [
        ReplaySSMCache(state, 8, 4, torch.bfloat16, num_key_heads=4, run_config=config) for state in [dense, sparse]
    ]
    q = torch.randn(1, 8, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, 8, 12, 128, device="cuda", dtype=torch.bfloat16)
    a = torch.full((8, 12), -3.0, device="cuda", dtype=torch.bfloat16)
    beta = torch.randn_like(a)
    alog = torch.zeros(12, device="cuda")
    bias = torch.zeros_like(alog)
    reqs = torch.tensor([0, 2], device="cuda", dtype=torch.int32)
    cu = reqs.new_tensor([0, 4, 8])

    def step(cache, accepted):
        pos = cache.prepare_decode(reqs, cu)
        out = [cache.forward(layer, q, k, v, a, beta, alog, bias, reqs, cu) for layer in range(48)]
        cache.accept_updates(reqs, accepted)
        return pos, out

    for count in [2, 3]:
        accepted = reqs.new_tensor([count - 1, -1, -1])
        for cache in caches:
            step(cache, accepted)
    reqs.copy_(reqs.new_tensor([0, 1]))
    expected_pos, expected_out = step(caches[0], reqs.new_tensor([0, 0, -1]))
    actual_pos, actual_out = step(caches[1], reqs.new_tensor([0, 0, -1]))
    torch.cuda.synchronize()
    assert torch.equal(actual_pos, expected_pos)
    assert torch.equal(caches[0].cursors, caches[1].cursors)
    for layer in range(48):
        torch.testing.assert_close(actual_out[layer].float(), expected_out[layer].float(), atol=0.006, rtol=0.02)
        torch.testing.assert_close(
            sparse[layer].float(), dense[layer].float(), atol=0.002, rtol=torch.finfo(torch.bfloat16).eps
        )


@pytest.mark.parametrize("batch", [1, 8, 12])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kda", [False, True], ids=["gdn", "kda"])
def test_small_batch_matches_larger_graph_padding(batch, dtype, kda):
    torch.manual_seed(1087 + batch)
    initial = torch.randn(2, 14, 4, 128, 128, device="cuda", dtype=dtype) * 0.01
    config = {"BV": 32, "num_warps": 1, "num_stages": 1}
    caches = [
        ReplaySSMCache(initial.clone(), 8, 4, num_key_heads=2, kda=kda, projection_mode="precompute", run_config=config)
        for _ in range(2)
    ]
    reqs = torch.arange(batch, device="cuda", dtype=torch.int32)
    padded = torch.cat((reqs, reqs.new_full((13 - batch,), 13)))
    cu = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * 4
    padded_cu = torch.cat((cu, cu.new_full((13 - batch,), batch * 4)))
    q = torch.randn(1, batch * 4, 2, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, batch * 4, 4, 128, device="cuda", dtype=torch.bfloat16)
    a = torch.full((batch * 4, 4 * 128 if kda else 4), -3.0, device="cuda", dtype=torch.bfloat16)
    beta = torch.randn(a.shape[0], 4, device="cuda", dtype=a.dtype)
    alog = torch.zeros(4, device="cuda")
    bias = torch.zeros(4 * 128 if kda else 4, device="cuda")
    accepted = torch.zeros(14, device="cuda", dtype=torch.int32)
    graphs, outputs = [], []
    for cache, ids, offsets in zip(caches, [reqs, padded], [cu, padded_cu]):

        def step():
            cache.prepare_decode(ids, offsets)
            out = [cache.forward(layer, q, k, v, a, beta, alog, bias, ids, offsets) for layer in range(2)]
            cache.accept_updates(ids, accepted)
            return out

        step()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = step()
        graphs.append(graph)
        outputs.append(out)
        cache.state.copy_(initial)
        cache.cursors.zero_()
    for iteration in range(32):
        accepted[:batch] = (torch.arange(batch, device="cuda") + iteration) % 4
        for value in [q, k, v, beta]:
            value.normal_()
        for graph in graphs:
            graph.replay()
        for small, padded_out in zip(outputs[0], outputs[1]):
            torch.testing.assert_close(small.float(), padded_out.float(), atol=0.006, rtol=0.02)
        assert torch.equal(caches[0].cursors, caches[1].cursors)
        torch.testing.assert_close(
            caches[0].state.float(), caches[1].state.float(), atol=0.002, rtol=torch.finfo(dtype).eps
        )
