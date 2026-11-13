"""Check the FlashInfer SM90 path used for unpadded GLM NoPE MLA."""

import inspect

import pytest
import torch

flashinfer = pytest.importorskip("flashinfer")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="FlashInfer NoPE FA3 MLA requires Hopper",
)


@pytest.mark.parametrize(
    "num_tokens,num_heads,kv_dtype",
    [
        (1, 16, torch.bfloat16),
        (8, 16, torch.bfloat16),
        (128, 16, torch.bfloat16),
        (8, 64, torch.bfloat16),
        (8, 16, torch.float8_e4m3fn),
    ],
)
def test_nope_sparse_mla_eager_and_graph_replay(num_tokens, num_heads, kv_dtype):
    wrapper_cls = flashinfer.mla.BatchMLAPagedAttentionWrapper
    assert "ckv_scale_arr" in inspect.signature(wrapper_cls.run).parameters, "FlashInfer >= 0.6.18 is required"

    torch.manual_seed(17)
    dim, num_slots, width = 512, 4096, 2176
    sm_scale = 0.0625
    q = torch.randn(num_tokens, num_heads, dim, dtype=torch.bfloat16, device="cuda")
    q_pe = q[..., :0]
    # BF16 KV shares its token row with the 144-byte index region. The
    # attention kernel must respect the resulting non-contiguous token stride.
    tail = 72 if kv_dtype == torch.bfloat16 else 0
    packed = torch.full((num_slots, 1, dim + tail), 7.0, dtype=kv_dtype, device="cuda")
    ckv = packed[..., :dim]
    raw_kv = torch.randn(num_slots, 1, dim, dtype=torch.bfloat16, device="cuda")
    scale = 0.125 if kv_dtype == torch.float8_e4m3fn else 1.0
    ckv.copy_((raw_kv.float() / scale).to(kv_dtype))
    kpe = packed[..., dim:dim]
    reference_kv = (ckv.float() * scale).to(torch.bfloat16).float()
    scale_kwargs = {"ckv_scale": scale, "kpe_scale": 1.0} if scale != 1.0 else {}

    qo_indptr = torch.arange(num_tokens + 1, dtype=torch.int32)
    kv_indptr = qo_indptr * width
    slots = torch.zeros(num_tokens, width, dtype=torch.int32, device="cuda")
    wrapper = wrapper_cls(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        use_cuda_graph=True,
        qo_indptr=qo_indptr.to("cuda"),
        kv_indptr=kv_indptr.to("cuda"),
        kv_indices=slots.flatten(),
        kv_len_arr=torch.empty(num_tokens, dtype=torch.int32, device="cuda"),
        backend="fa3",
    )

    def plan(step):
        choices = [1, 3, 127, 128, 511, 2048, 2049, 2051]
        lengths = [choices[(row + step) % len(choices)] for row in range(num_tokens)]
        slots.zero_()
        for row, length in enumerate(lengths):
            slots[row, :length] = (torch.arange(length, device="cuda") * 5 + row * 31 + step * 7) % num_slots
        # Lengths are compiled into the host-side plan. Replan before graph
        # replay when they change, while keeping all captured buffers stable.
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            slots.flatten(),
            torch.tensor(lengths, dtype=torch.int32),
            num_heads,
            dim,
            0,
            1,
            False,
            sm_scale,
            torch.bfloat16,
            kv_dtype,
        )
        return lengths

    def check(output, lengths):
        expected = []
        for row, length in enumerate(lengths):
            keys = reference_kv[slots[row, :length].long(), 0]
            scores = q[row].float() @ keys.T * sm_scale
            expected.append(scores.softmax(-1) @ keys)
        torch.testing.assert_close(output.float(), torch.stack(expected), atol=0.012, rtol=0.015)

    lengths = plan(0)
    output = torch.empty_like(q)
    wrapper.run(q, q_pe, ckv, kpe, out=output, **scale_kwargs)
    check(output, lengths)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(q, q_pe, ckv, kpe, out=output, **scale_kwargs)
    graph.replay()
    check(output, lengths)

    lengths = plan(1)
    q.normal_()
    graph.replay()
    check(output, lengths)
