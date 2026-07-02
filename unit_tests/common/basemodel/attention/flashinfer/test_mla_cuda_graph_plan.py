import pytest
import torch

flashinfer = pytest.importorskip("flashinfer")

from lightllm.common.basemodel.triton_kernel.flashinfer_mla_plan import (  # noqa: E402
    fill_mla_decode_plan_for_cuda_graph,
)


def _make_wrapper(
    q_indptr,
    kv_indptr_buf,
    kv_indptr,
    kv_indices,
    kv_lens_buf,
    num_heads,
    head_dim_ckv,
    head_dim_kpe,
    sm_scale,
    dtype,
    init_short,
):
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda")
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
        workspace,
        use_cuda_graph=True,
        qo_indptr=q_indptr,
        kv_indices=kv_indices,
        kv_indptr=kv_indptr_buf,
        kv_len_arr=kv_lens_buf,
    )
    if init_short:
        batch_size = kv_lens_buf.numel()
        init_lens = torch.full((batch_size,), 2, dtype=torch.int32, device="cuda")
        init_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device="cuda") * 2
        wrapper.plan(
            q_indptr,
            init_indptr,
            kv_indices,
            init_lens,
            num_heads,
            head_dim_ckv,
            head_dim_kpe,
            1,
            False,
            sm_scale,
            dtype,
            dtype,
        )
        kv_indptr_buf.copy_(kv_indptr)
        kv_lens_buf.copy_(torch.diff(kv_indptr))
    else:
        wrapper.plan(
            q_indptr,
            kv_indptr,
            kv_indices,
            kv_lens_buf,
            num_heads,
            head_dim_ckv,
            head_dim_kpe,
            1,
            False,
            sm_scale,
            dtype,
            dtype,
        )
    return wrapper


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("lengths,num_heads", [([4, 64], 32), ([1000, 32768], 32), ([1000, 131072], 128)])
def test_mla_cuda_graph_triton_plan_matches_flashinfer_plan(lengths, num_heads):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    head_dim_ckv = 512
    head_dim_kpe = 64
    batch_size = len(lengths)
    total_kv_len = sum(lengths)
    sm_scale = (head_dim_ckv + head_dim_kpe) ** -0.5

    q_nope = torch.randn((batch_size, num_heads, head_dim_ckv), dtype=dtype, device="cuda")
    q_pe = torch.randn((batch_size, num_heads, head_dim_kpe), dtype=dtype, device="cuda")
    ckv = torch.randn((total_kv_len, 1, head_dim_ckv), dtype=dtype, device="cuda")
    kpe = torch.randn((total_kv_len, 1, head_dim_kpe), dtype=dtype, device="cuda")
    kv_lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    kv_indptr = torch.empty(batch_size + 1, dtype=torch.int32, device="cuda")
    kv_indptr[0] = 0
    kv_indptr[1:] = torch.cumsum(kv_lens, dim=0)
    kv_indices = torch.arange(total_kv_len, dtype=torch.int32, device="cuda")
    q_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device="cuda")

    ref_wrapper = _make_wrapper(
        q_indptr,
        kv_indptr.clone(),
        kv_indptr,
        kv_indices,
        kv_lens.clone(),
        num_heads,
        head_dim_ckv,
        head_dim_kpe,
        sm_scale,
        dtype,
        init_short=False,
    )
    graph_wrapper = _make_wrapper(
        q_indptr,
        kv_indptr.clone(),
        kv_indptr,
        kv_indices,
        kv_lens.clone(),
        num_heads,
        head_dim_ckv,
        head_dim_kpe,
        sm_scale,
        dtype,
        init_short=True,
    )
    fill_mla_decode_plan_for_cuda_graph(
        graph_wrapper,
        graph_wrapper._kv_indptr_buf,
        batch_size,
        num_heads,
        max(lengths),
    )

    ref_out = torch.empty((batch_size, num_heads, head_dim_ckv), dtype=dtype, device="cuda")
    graph_out = torch.empty_like(ref_out)
    ref_wrapper.run(q_nope, q_pe, ckv, kpe, out=ref_out, return_lse=False)
    graph_wrapper.run(q_nope, q_pe, ckv, kpe, out=graph_out, return_lse=False)

    assert torch.allclose(ref_out, graph_out, atol=1e-2, rtol=1e-2)
