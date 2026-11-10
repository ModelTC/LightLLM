import pytest
import torch
import triton
from triton.runtime.jit import JITFunction

from lightllm.models.glm5_next.triton_kernel.index_quant import (
    hadamard_transform_quant_fp8,
)
from lightllm.models.glm5_next.triton_kernel.kpool import (
    _compress_pools,
    _gather_pools,
    _get_pool_ranges,
    compress_pools,
    gather_pools,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _check_pool_ranges(batch, max_pools, single_query):
    counts = [1] * batch if single_query else [0 if i % 7 == 3 else i % 3 + 1 for i in range(batch)]
    cu = [0]
    visible, owners = [], []
    for owner, count in enumerate(counts):
        visible.extend(range(4, count + 4))
        owners.extend([owner] * count)
        cu.append(len(visible))
    lengths = torch.tensor(visible, device="cuda", dtype=torch.int32)
    cu_q_lens = torch.tensor(cu, device="cuda", dtype=torch.int32)
    outputs = [torch.empty_like(lengths) for _ in range(3)]
    tokens = lengths.numel()
    blocks = triton.cdiv(tokens, 256) if single_query else tokens // 256 + batch
    compiled = _get_pool_ranges[(blocks,)](
        lengths, cu_q_lens, *outputs, batch, tokens, max_pools, single_query, 256, num_warps=4
    )
    starts = torch.tensor(owners, device="cuda", dtype=torch.int32) * max_pools
    expected = (starts, starts + lengths // 4, lengths // 4)
    for actual, reference in zip(outputs, expected):
        assert torch.equal(actual, reference)
    return compiled.hash


@pytest.mark.parametrize("single_query", [False, True])
def test_pool_ranges_reuse_compilation_across_runtime_sizes(single_query):
    # The indexer rounds capacity up to 128 pools. Changing that capacity keeps
    # its default alignment specialization while batch/token counts are dynamic.
    hashes = {
        _check_pool_ranges(batch, max_pools, single_query)
        for batch, max_pools in [(1, 128), (3, 256), (16, 384), (17, 640), (127, 768), (128, 1024), (129, 1152)]
    }
    assert len(hashes) == 1


@pytest.mark.parametrize("single_query", [False, True])
@pytest.mark.parametrize("batch,max_pools", [(1, 1), (3, 3), (16, 16), (129, 1025)])
def test_pool_ranges_nonstandard_capacities_preserve_correctness(batch, max_pools, single_query):
    _check_pool_ranges(batch, max_pools, single_query)


def test_pool_compression_reuses_compilation_within_static_query_layouts(monkeypatch):
    torch.manual_seed(1525)
    tail = torch.zeros((130, 4, 256), device="cuda", dtype=torch.bfloat16)
    ape = torch.zeros((4, 128), device="cuda")
    hashes = {}
    layout = None
    misses = []
    original_run = _compress_pools.run
    original_compile = JITFunction._do_compile

    def observe_compile(self, *args, **kwargs):
        if self is _compress_pools:
            misses.append(layout)
        return original_compile(self, *args, **kwargs)

    def observe(*args, **kwargs):
        compiled = original_run(*args, **kwargs)
        hashes.setdefault(layout, set()).add(compiled.hash)
        return compiled

    monkeypatch.setattr(_compress_pools, "run", observe)
    monkeypatch.setattr(JITFunction, "_do_compile", observe_compile)
    for batch, single_query in [(1, False), (3, True), (33, False), (1, True), (17, True), (1, True)]:
        values = torch.randint(-16, 17, (batch, 4, 256), device="cuda").bfloat16()
        values[:, :, 128:] = 0
        tail[:batch].zero_()
        packed = torch.zeros((batch * 4, 1, 720), device="cuda", dtype=torch.uint8)
        requests = torch.arange(batch, device="cuda", dtype=torch.int32)
        ragged = torch.arange(batch * 4, device="cuda", dtype=torch.int32)
        seq_lens = torch.full((batch,), 4, device="cuda", dtype=torch.int32)
        if single_query:
            tail[:batch, :3] = values[:, :3]
            raw = values[:, 3].contiguous()
            lengths = seq_lens
            mem_index = requests * 4 + 3
            cu = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
        else:
            raw = values.reshape(batch * 4, 256)
            lengths = torch.arange(1, 5, device="cuda", dtype=torch.int32).repeat(batch)
            mem_index = ragged
            cu = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * 4
        # Query mode and address strides stay constexpr for the faster kernel.
        layout = (single_query, raw.stride(0))
        seen_layout = layout in hashes
        misses_before = len(misses)
        compress_pools(raw, tail, packed, ape, lengths, mem_index, requests, cu, seq_lens, 1 if single_query else 4)
        if seen_layout:
            assert len(misses) == misses_before
        keys, scales = gather_pools(packed, ragged.view(batch, 4), requests, seq_lens, 1)
        expected = values[:, :, :128].float().mean(1).bfloat16()
        expected_key, expected_scale = hadamard_transform_quant_fp8(expected, scale=128 ** -0.5)
        assert torch.equal(keys.view(torch.uint8), expected_key.view(torch.uint8))
        assert torch.equal(scales, expected_scale.flatten())
        assert torch.equal(tail[:batch], values)
    assert len(hashes) == 3
    assert all(len(variants) == 1 for variants in hashes.values())


def test_gather_pools_reuses_compilation_across_pool_counts():
    torch.manual_seed(1525)
    packed = torch.zeros((16, 1, 720), device="cuda", dtype=torch.uint8)
    source_keys = torch.randint(0, 256, (16, 128), device="cuda", dtype=torch.uint8)
    source_scales = torch.rand(16, device="cuda") + 0.1
    packed[:, 0, :128] = source_keys
    packed[:, 0, 128:132] = source_scales.view(torch.uint8).view(16, 4)
    locations = torch.randint(0, 16, (6, 129), device="cuda", dtype=torch.int32)
    table = torch.zeros((6, 129 * 4), device="cuda", dtype=torch.int32)
    table[:, 3::4] = locations
    requests = torch.tensor([2, 0, 5], device="cuda", dtype=torch.int32)
    hashes = set()
    for pools in [1, 3, 16, 33, 64, 129]:
        lengths = torch.tensor([pools * 4, min(3, pools) * 4, 2], device="cuda", dtype=torch.int32)
        keys = torch.empty((3, pools, 128), device="cuda", dtype=torch.float8_e4m3fn)
        scales = torch.empty((3, pools), device="cuda")
        compiled = _gather_pools[(triton.cdiv(pools, 32), 3)](
            packed,
            table,
            requests,
            lengths,
            keys,
            scales,
            packed.stride(0),
            table.stride(0),
            pools,
            POOLS_PER_BLOCK=32,
            num_warps=4,
        )
        hashes.add(compiled.hash)
        for row, (request, count) in enumerate([(2, pools), (0, min(3, pools)), (5, 0)]):
            indices = locations[request, :count].long()
            assert torch.equal(keys.view(torch.uint8)[row, :count], source_keys[indices])
            assert torch.equal(scales[row, :count], source_scales[indices])
            assert not keys.view(torch.uint8)[row, count:].any()
            assert (scales[row, count:] == 1).all()
    assert len(hashes) == 1


def test_pool_compression_reuses_compilation_across_request_counts(monkeypatch):
    torch.manual_seed(1525)
    tail = torch.zeros((130, 4, 256), device="cuda", dtype=torch.bfloat16)
    # Exact uniform weights isolate indexing from softmax/FP8 rounding differences.
    ape = torch.zeros((4, 128), device="cuda")
    hashes = set()
    original_run = _compress_pools.run

    def observe(*args, **kwargs):
        compiled = original_run(*args, **kwargs)
        hashes.add(compiled.hash)
        return compiled

    monkeypatch.setattr(_compress_pools, "run", observe)
    for batch in [1, 3, 16, 17, 127, 128, 129]:
        raw = torch.randint(-16, 17, (batch * 4, 256), device="cuda").bfloat16()
        raw[:, 128:] = 0
        packed = torch.zeros((batch * 4, 1, 720), device="cuda", dtype=torch.uint8)
        requests = torch.arange(batch, device="cuda", dtype=torch.int32)
        lengths = torch.arange(1, 5, device="cuda", dtype=torch.int32).repeat(batch)
        ragged = torch.arange(batch * 4, device="cuda", dtype=torch.int32)
        cu = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * 4
        seq_lens = torch.full((batch,), 4, device="cuda", dtype=torch.int32)
        compress_pools(
            raw,
            tail,
            packed,
            ape,
            lengths,
            ragged,
            requests,
            cu,
            seq_lens,
            max_q_seq_len=4,
        )
        keys, scales = gather_pools(packed, ragged.view(batch, 4), requests, seq_lens, 1)
        values = raw.view(batch, 4, 256)
        expected = (values[:, :, :128].float() * (values[:, :, 128:].float() + ape).softmax(1)).sum(1).bfloat16()
        expected_key, expected_scale = hadamard_transform_quant_fp8(expected, scale=128 ** -0.5)
        assert torch.equal(keys.view(torch.uint8), expected_key.view(torch.uint8))
        assert torch.equal(scales, expected_scale.flatten())
        assert torch.equal(tail[:batch], values)
    assert len(hashes) == 1
