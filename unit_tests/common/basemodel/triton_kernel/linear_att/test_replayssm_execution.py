import pytest
import torch

from unit_tests.common.basemodel.triton_kernel.linear_att.replay_history import (
    accept_replay_window,
    attach_replay_history as ReplaySSMExecutor,
    replay_layer,
)
from lightllm.common.basemodel.triton_kernel.linear_att.mtp_state_params import (
    build_dynamic_mtp_linear_att_state_params,
)
from lightllm.common.basemodel.triton_kernel.linear_att.replayssm_cursor import advance_cursor
from lightllm.common.basemodel.triton_kernel.linear_att.ssm_acceptance import commit_ssm_prefix
from types import SimpleNamespace

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("kda", [False, True], ids=["gdn", "kda"])
def test_decode_compilation_uses_shape_buckets(monkeypatch, kda):
    from lightllm.common.basemodel.triton_kernel.linear_att.replayssm_cursor import _advance_cursor
    from lightllm.common.basemodel.triton_kernel.linear_att.replayssm_decode import _decode_replay

    kernels = {"_decode_replay": _decode_replay, "_advance_cursor": _advance_cursor}
    hashes = {name: set() for name in kernels}
    launches = []
    for name, kernel in kernels.items():

        def record(*args, run=kernel.run, name=name, **kwargs):
            compiled = run(*args, **kwargs)
            hashes[name].add(compiled.hash)
            launches.append(name)
            return compiled

        monkeypatch.setattr(kernel, "run", record)

    torch.manual_seed(137)
    initial = torch.randn(1, 513, 1, 32, 32, device="cuda") * 0.01
    eager, captured = [
        ReplaySSMExecutor(
            initial.clone(), 8, 1, torch.float32, kda=kda, run_config={"BV": 32, "num_warps": 1, "num_stages": 3}
        )
        for _ in range(2)
    ]
    reqs = torch.arange(512, device="cuda", dtype=torch.int32)
    q = torch.randn(512, 1, 32, device="cuda") * 0.2
    k, v = torch.randn_like(q), torch.randn_like(q)
    a = torch.full((512, 32 if kda else 1), -3.0, device="cuda")
    beta = torch.randn(512, 1, device="cuda")
    alog, bias = torch.zeros(1, device="cuda"), torch.zeros(a.shape[-1], device="cuda")

    def cycle(backend, ids):
        count = ids.numel()
        cu = torch.arange(count + 1, dtype=torch.int32, device=ids.device)
        out = replay_layer(backend, 0, q[:count], k[:count], v[:count], a[:count], beta[:count], alog, bias, ids, cu)
        accept_replay_window(backend, ids, cu_seqlens=cu)
        return out

    graphs = {}
    for bucket in (32, 512):
        ids = reqs[:bucket].clone()
        cycle(captured, ids)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = cycle(captured, ids)
        graphs[bucket] = graph, ids, out
    captured.ssm_state.copy_(initial)
    captured.cursors.zero_()
    counts = (1, 2, 7, 16, 23, 24, 25, 31, 32, 33, 255, 256, 257, 511)
    for count in counts:
        graph, ids, output = graphs[32 if count <= 32 else 512]
        ids.fill_(captured.hold_req_id)
        ids[:count].copy_(reqs[:count])
        for _ in range(8):
            live = reqs[:count]
            if (eager.cursors[live.long()] + 1 > eager.capacity).any():
                eager.fold_accepted_prefix(live)
                captured.fold_accepted_prefix(ids[:count])
            expected = cycle(eager, reqs[:count])
            calls = len(launches)
            graph.replay()
            assert len(launches) == calls
            torch.testing.assert_close(output[:count], expected, atol=3e-5, rtol=3e-5)
        assert torch.equal(captured.cursors, eager.cursors)
        torch.testing.assert_close(captured.ssm_state, eager.ssm_state, atol=3e-5, rtol=3e-5)
    assert len(hashes["_decode_replay"]) == 1
    assert len(hashes["_advance_cursor"]) == 1
    assert captured.cursors[-1] == 0
    assert torch.equal(captured.ssm_state[:, -1], initial[:, -1])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("capture_metadata", [False, True])
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("kda", [False, True], ids=["gdn", "kda"])
def test_request_manager_accept_with_replay_graph(dtype, capture_metadata, varlen, kda):
    torch.manual_seed(827)
    config = {"BV": 32, "num_warps": 1, "num_stages": 1}
    initial = torch.randn(3, 5, 4, 128, 128, device="cuda", dtype=dtype) * 0.01
    ssm_backends = [
        cls(
            initial.clone(),
            8,
            4,
            torch.bfloat16,
            num_key_heads=2,
            kda=kda,
            compute_mode="precompute",
            run_config=config,
        )
        for cls in [ReplaySSMExecutor, ReplaySSMExecutor]
    ]
    manager = SimpleNamespace()
    manager._ssm_executor = ssm_backends[1]
    manager.mtp_state_index = torch.zeros(5, device="cuda", dtype=torch.int32)
    rows = torch.full((12,), 4, device="cuda", dtype=torch.int32)
    mtp = torch.zeros_like(rows)
    flags = torch.zeros_like(rows)
    starts = torch.zeros(3, device="cuda", dtype=torch.int32)
    accepted = torch.full((5,), -1, device="cuda", dtype=torch.int32)
    q = torch.randn(12, 2, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(12, 4, 128, device="cuda", dtype=torch.bfloat16)
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

    def metadata(ssm_backend):
        cu, reqs, _counts = build_dynamic_mtp_linear_att_state_params(rows, mtp, manager.mtp_state_index, 4)
        return cu, reqs, ssm_backend.cursors.index_select(0, reqs.long())

    def forward_accept(index, cu, reqs):
        ssm_backend = ssm_backends[index]
        out = [replay_layer(ssm_backend, layer, q, k, v, a, beta, alog, bias, reqs, cu) for layer in range(3)]
        if index == 0:
            accept_replay_window(ssm_backend, reqs, accepted, cu)
        else:
            count = starts.numel()
            commit_reqs = rows[starts.long()]
            window = (cu[1 : count + 1] - cu[:count]).contiguous()
            advance_cursor(ssm_backend, commit_reqs, cu[: count + 1])
            commit_ssm_prefix(
                manager.mtp_state_index,
                commit_reqs,
                accepted[commit_reqs.long()] + 1,
                window,
                ssm_backend.cursors,
            )
        return out

    inputs(0)
    graphs, captured, outputs = [], [], []
    for index, ssm_backend in enumerate(ssm_backends):
        cu, reqs, pos = metadata(ssm_backend)
        forward_accept(index, cu, reqs)
        ssm_backend.ssm_state.copy_(initial)
        ssm_backend.cursors.zero_()
        manager.mtp_state_index.zero_()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            if capture_metadata:
                cu, reqs, pos = metadata(ssm_backend)
            out = forward_accept(index, cu, reqs)
        graphs.append(g)
        captured.append((cu, reqs, pos))
        outputs.append(out)
    for ssm_backend in ssm_backends:
        ssm_backend.ssm_state.copy_(initial)
        ssm_backend.cursors.zero_()
    manager.mtp_state_index.zero_()
    folds = 0
    for iteration in range(32):
        ids, lengths, counts, logical = inputs(iteration)
        for tensor in [q, k, v, beta]:
            tensor.normal_()
        overflow = [
            req for req, length in zip(ids, lengths) if ssm_backends[0].cursors[req] + length > ssm_backends[0].capacity
        ]
        if overflow:
            overflow_reqs = torch.tensor(overflow, dtype=torch.int32, device="cuda")
            for ssm_backend in ssm_backends:
                ssm_backend.fold_accepted_prefix(overflow_reqs)
            folds += len(overflow)
        for index, ssm_backend in enumerate(ssm_backends):
            if not capture_metadata:
                for dst, src in zip(captured[index], metadata(ssm_backend)):
                    dst.copy_(src)
            graphs[index].replay()
        torch.cuda.synchronize()
        for expected, actual in zip(captured[0], captured[1]):
            assert torch.equal(actual, expected)
        assert torch.equal(ssm_backends[0].cursors, ssm_backends[1].cursors)
        for expected, actual in zip(outputs[0], outputs[1]):
            torch.testing.assert_close(
                actual[:, :logical].float(), expected[:, :logical].float(), atol=0.006, rtol=0.02
            )
        torch.testing.assert_close(
            ssm_backends[1].ssm_state.float(),
            ssm_backends[0].ssm_state.float(),
            atol=0.002,
            rtol=torch.finfo(dtype).eps,
        )
        table = manager.mtp_state_index.cpu().tolist()
        for req, count in zip(ids, counts):
            assert table[req] == count - 1
        assert table[2] == table[4] == 0
        assert ssm_backends[1].cursors[4].item() == 0
    assert folds > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("capture_metadata", [False, True])
@pytest.mark.parametrize("stagger", [False, True])
@pytest.mark.parametrize("kda", [False, True], ids=["gdn", "kda"])
def test_decode_accept_order_preserves_forward_snapshot(dtype, capture_metadata, stagger, kda):
    torch.manual_seed(731)
    initial = torch.randn(3, 3, 4, 128, 128, device="cuda", dtype=dtype) * 0.01
    ssm_backends = [
        ReplaySSMExecutor(
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
    for ssm_backend in ssm_backends:
        ssm_backend.fold_programs = 3
    reqs = torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32)
    q = torch.randn(3, 2, 128, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn(3, 4, 128, device="cuda", dtype=dtype)
    a = torch.full((3, 4 * 128 if kda else 4), -3.0, device="cuda", dtype=dtype)
    beta = torch.randn(a.shape[0], 4, device="cuda", dtype=a.dtype)
    alog = torch.zeros(4, device="cuda")
    bias = torch.zeros(4 * 128 if kda else 4, device="cuda")

    cu = torch.arange(reqs.numel() + 1, dtype=torch.int32, device=reqs.device)

    def forward(ssm_backend):
        return [replay_layer(ssm_backend, layer, q, k, v, a, beta, alog, bias, reqs, cu) for layer in range(3)]

    def warm(ssm_backend):
        out = forward(ssm_backend)
        accept_replay_window(ssm_backend, reqs)
        return out

    graphs, positions, outputs = [], [], []
    for ssm_backend in ssm_backends:
        warm(ssm_backend)
        ssm_backend.ssm_state.copy_(initial)
        ssm_backend.cursors.zero_()
        if stagger:
            reqs.copy_(reqs.new_tensor([0, 2, 2]))
            for _ in range(3):
                warm(ssm_backend)
            reqs.copy_(reqs.new_tensor([0, 1, 2]))
        pos = torch.zeros_like(reqs)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            if capture_metadata:
                pos = ssm_backend.cursors.index_select(0, reqs.long())
            out = forward(ssm_backend)
            if capture_metadata:
                accept_replay_window(ssm_backend, reqs)
        graphs.append(graph)
        positions.append(pos)
        outputs.append(out)
    for iteration in range(24):
        reqs.copy_(reqs.new_tensor([0, 1, 2] if iteration % 2 == 0 else [1, 0, 2]))
        for tensor in [q, k, v, beta]:
            tensor.normal_()
        overflow = [int(req) for req in reqs.tolist() if ssm_backends[0].cursors[req] + 1 > ssm_backends[0].capacity]
        if overflow:
            overflow_reqs = torch.tensor(sorted(set(overflow)), dtype=torch.int32, device="cuda")
            for ssm_backend in ssm_backends:
                ssm_backend.fold_accepted_prefix(overflow_reqs)
        old = ssm_backends[0].cursors[reqs.long()].clone()
        for ssm_backend, graph, pos in zip(ssm_backends, graphs, positions):
            if not capture_metadata:
                pos.copy_(ssm_backend.cursors.index_select(0, reqs.long()))
            graph.replay()
            if not capture_metadata:
                accept_replay_window(ssm_backend, reqs)
        torch.cuda.synchronize()
        assert torch.equal(positions[0], old)
        assert torch.equal(positions[0], positions[1])
        assert torch.equal(ssm_backends[0].cursors, ssm_backends[1].cursors)
        for expected, actual in zip(outputs[0], outputs[1]):
            torch.testing.assert_close(actual.float(), expected.float(), atol=0.006, rtol=0.02)
        torch.testing.assert_close(
            ssm_backends[1].ssm_state.float(),
            ssm_backends[0].ssm_state.float(),
            atol=0.002,
            rtol=torch.finfo(dtype).eps,
        )


@pytest.mark.parametrize("kda", [False, True], ids=["gdn", "kda"])
def test_cross_layer_fold_with_state_offset_above_int32(kda):
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
    ssm_backends = [
        ReplaySSMExecutor(state, 8, 4, torch.bfloat16, num_key_heads=4, kda=kda, run_config=config)
        for state in [dense, sparse]
    ]
    q = torch.randn(8, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(8, 12, 128, device="cuda", dtype=torch.bfloat16)
    a = torch.full((8, 12 * 128 if kda else 12), -3.0, device="cuda", dtype=torch.bfloat16)
    beta = torch.randn(8, 12, device="cuda", dtype=torch.bfloat16)
    alog = torch.zeros(12, device="cuda")
    bias = torch.zeros(12 * 128 if kda else 12, device="cuda")
    reqs = torch.tensor([0, 2], device="cuda", dtype=torch.int32)
    cu = reqs.new_tensor([0, 4, 8])

    def step(ssm_backend, accepted):
        overflow = [int(req) for req in reqs.tolist() if ssm_backend.cursors[req] + 4 > ssm_backend.capacity]
        if overflow:
            ssm_backend.fold_accepted_prefix(reqs.new_tensor(overflow))
        pos = ssm_backend.cursors.index_select(0, reqs.long())
        out = [replay_layer(ssm_backend, layer, q, k, v, a, beta, alog, bias, reqs, cu) for layer in range(48)]
        accept_replay_window(ssm_backend, reqs, accepted, cu)
        return pos, out

    for count in [2, 3]:
        accepted = reqs.new_tensor([count - 1, -1, -1])
        for ssm_backend in ssm_backends:
            step(ssm_backend, accepted)
    reqs.copy_(reqs.new_tensor([0, 1]))
    expected_pos, expected_out = step(ssm_backends[0], reqs.new_tensor([0, 0, -1]))
    actual_pos, actual_out = step(ssm_backends[1], reqs.new_tensor([0, 0, -1]))
    torch.cuda.synchronize()
    assert torch.equal(actual_pos, expected_pos)
    assert torch.equal(ssm_backends[0].cursors, ssm_backends[1].cursors)
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
    ssm_backends = [
        ReplaySSMExecutor(initial.clone(), 8, 4, num_key_heads=2, kda=kda, compute_mode="precompute", run_config=config)
        for _ in range(2)
    ]
    reqs = torch.arange(batch, device="cuda", dtype=torch.int32)
    padded = torch.cat((reqs, reqs.new_full((13 - batch,), 13)))
    cu = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * 4
    padded_cu = torch.cat((cu, cu.new_full((13 - batch,), batch * 4)))
    q = torch.randn(batch * 4, 2, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(batch * 4, 4, 128, device="cuda", dtype=torch.bfloat16)
    a = torch.full((batch * 4, 4 * 128 if kda else 4), -3.0, device="cuda", dtype=torch.bfloat16)
    beta = torch.randn(a.shape[0], 4, device="cuda", dtype=a.dtype)
    alog = torch.zeros(4, device="cuda")
    bias = torch.zeros(4 * 128 if kda else 4, device="cuda")
    accepted = torch.zeros(14, device="cuda", dtype=torch.int32)
    graphs, outputs = [], []
    for ssm_backend, ids, offsets in zip(ssm_backends, [reqs, padded], [cu, padded_cu]):

        def step():
            out = [replay_layer(ssm_backend, layer, q, k, v, a, beta, alog, bias, ids, offsets) for layer in range(2)]
            accept_replay_window(ssm_backend, ids, accepted, offsets)
            return out

        step()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = step()
        graphs.append(graph)
        outputs.append(out)
        ssm_backend.ssm_state.copy_(initial)
        ssm_backend.cursors.zero_()
    for iteration in range(32):
        accepted[:batch] = (torch.arange(batch, device="cuda") + iteration) % 4
        for value in [q, k, v, beta]:
            value.normal_()
        overflow = [idx for idx in range(batch) if ssm_backends[0].cursors[idx] + 4 > ssm_backends[0].capacity]
        if overflow:
            overflow_reqs = torch.tensor(overflow, dtype=torch.int32, device="cuda")
            for ssm_backend in ssm_backends:
                ssm_backend.fold_accepted_prefix(overflow_reqs)
        for graph in graphs:
            graph.replay()
        for small, padded_out in zip(outputs[0], outputs[1]):
            torch.testing.assert_close(small.float(), padded_out.float(), atol=0.006, rtol=0.02)
        assert torch.equal(ssm_backends[0].cursors, ssm_backends[1].cursors)
        torch.testing.assert_close(
            ssm_backends[0].ssm_state.float(),
            ssm_backends[1].ssm_state.float(),
            atol=0.002,
            rtol=torch.finfo(dtype).eps,
        )


@pytest.mark.parametrize("kda", [False, True], ids=["gdn", "kda"])
def test_full_prefix_does_not_address_past_capacity(kda):
    torch.manual_seed(4)
    capacity = 4
    backend = ReplaySSMExecutor(
        torch.randn(1, 3, 2, 32, 32, device="cuda") * 0.01,
        capacity,
        1,
        torch.bfloat16,
        kda=kda,
        run_config={"BV": 32, "num_warps": 1, "num_stages": 1},
    )
    reqs = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    cu = torch.arange(reqs.numel() + 1, dtype=torch.int32, device=reqs.device)
    q = torch.randn(2, 1, 32, device="cuda") * 0.2
    k, v = torch.randn_like(q), torch.randn_like(q)
    a = torch.full((2, 32 if kda else 1), -3.0, device="cuda")
    beta = torch.randn(2, 1, device="cuda")
    alog, bias = torch.zeros(1, device="cuda"), torch.zeros(a.shape[-1], device="cuda")
    backend.ssm_state.zero_()
    backend.cursors.zero_()
    for name in ("keys", "deltas", "gates", "raw_keys", "raw_values", "betas"):
        getattr(backend, name).zero_()
    backend.cursors[reqs.long()] = capacity + 2
    replay_layer(backend, 0, q, k, v, a, beta, alog, bias, reqs, cu)
    backend.fold_accepted_prefix(reqs)
    torch.cuda.synchronize()
    assert torch.equal(backend.cursors[reqs.long()], torch.zeros(2, device="cuda", dtype=torch.int32))
    assert torch.isfinite(backend.ssm_state).all()
    for _ in range(12):
        replay_layer(backend, 0, q, k, v, a, beta, alog, bias, reqs, cu)
        accept_replay_window(backend, reqs)
        overflow = reqs[backend.cursors[reqs.long()] + backend.verify_width > backend.capacity]
        if overflow.numel():
            backend.fold_accepted_prefix(overflow)
    torch.cuda.synchronize()
    assert int(backend.cursors[reqs.long()].max()) <= capacity
    assert torch.isfinite(backend.ssm_state).all()
