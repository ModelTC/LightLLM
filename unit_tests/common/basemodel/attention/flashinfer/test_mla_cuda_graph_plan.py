from types import SimpleNamespace

import pytest
import torch

flashinfer = pytest.importorskip("flashinfer")

from lightllm.common.basemodel.attention.flashinfer.mla import (  # noqa: E402
    MlaFlashInferDecodeAttState,
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
    page_size,
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
        init_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device="cuda") * (
            (2 + page_size - 1) // page_size
        )
        wrapper.plan(
            q_indptr,
            init_indptr,
            kv_indices,
            init_lens,
            num_heads,
            head_dim_ckv,
            head_dim_kpe,
            page_size,
            False,
            sm_scale,
            dtype,
            dtype,
        )
    else:
        wrapper.plan(
            q_indptr,
            kv_indptr,
            kv_indices,
            kv_lens_buf,
            num_heads,
            head_dim_ckv,
            head_dim_kpe,
            page_size,
            False,
            sm_scale,
            dtype,
            dtype,
        )
    return wrapper


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("page_size", [1, 3, 4, 16, 64, 128])
@pytest.mark.parametrize("lengths,num_heads", [([5, 64, 65], 32), ([1000, 32768], 32), ([1000, 131073], 128)])
def test_mla_cuda_graph_triton_plan_matches_flashinfer_plan(lengths, num_heads, page_size):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    head_dim_ckv = 512
    head_dim_kpe = 64
    batch_size = len(lengths)
    total_pages = sum((length + page_size - 1) // page_size for length in lengths)
    sm_scale = (head_dim_ckv + head_dim_kpe) ** -0.5

    q_nope = torch.randn((batch_size, num_heads, head_dim_ckv), dtype=dtype, device="cuda")
    q_pe = torch.randn((batch_size, num_heads, head_dim_kpe), dtype=dtype, device="cuda")
    ckv = torch.randn((total_pages, page_size, head_dim_ckv), dtype=dtype, device="cuda")
    kpe = torch.randn((total_pages, page_size, head_dim_kpe), dtype=dtype, device="cuda")
    kv_lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    kv_indptr = torch.empty(batch_size + 1, dtype=torch.int32, device="cuda")
    kv_indptr[0] = 0
    kv_indptr[1:] = torch.cumsum((kv_lens + page_size - 1) // page_size, dim=0)
    kv_indices = torch.randperm(total_pages, dtype=torch.int32, device="cuda")
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
        page_size,
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
        page_size,
        init_short=True,
    )
    backend = SimpleNamespace(tp_q_head_num=num_heads, infer_page_size=page_size)
    graph_state = MlaFlashInferDecodeAttState(
        backend=backend,
        infer_state=SimpleNamespace(batch_size=batch_size, b_seq_len=graph_wrapper._kv_len_arr_buf),
        kv_starts=graph_wrapper._kv_indptr_buf,
        kv_indices=graph_wrapper._kv_indices_buf,
        decode_wrapper=graph_wrapper,
    )
    ref_out = torch.empty((batch_size, num_heads, head_dim_ckv), dtype=dtype, device="cuda")
    graph_out = torch.empty_like(ref_out)
    graph_wrapper.run(q_nope, q_pe, ckv, kpe, out=graph_out, return_lse=False)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_wrapper.run(q_nope, q_pe, ckv, kpe, out=graph_out, return_lse=False)

    # 复用同一 graph，覆盖页表变化、末页空位及 split/exact 切换后的 plan 更新。
    for replay_lengths in (lengths, [1] * batch_size, lengths):
        replay_lens = torch.tensor(replay_lengths, dtype=torch.int32, device="cuda")
        replay_indptr = torch.zeros_like(kv_indptr)
        replay_indptr[1:] = torch.cumsum((replay_lens + page_size - 1) // page_size, dim=0)
        new_state = MlaFlashInferDecodeAttState(
            backend=backend,
            infer_state=SimpleNamespace(
                batch_size=batch_size, b_seq_len=replay_lens, max_kv_seq_len=max(replay_lengths)
            ),
            kv_starts=replay_indptr,
            kv_indices=kv_indices,
        )
        graph_state.copy_for_decode_cuda_graph(new_state)
        graph.replay()
        ref_wrapper.plan(
            q_indptr,
            replay_indptr,
            kv_indices,
            replay_lens,
            num_heads,
            head_dim_ckv,
            head_dim_kpe,
            page_size,
            False,
            sm_scale,
            dtype,
            dtype,
        )
        ref_wrapper.run(q_nope, q_pe, ckv, kpe, out=ref_out, return_lse=False)
        torch.testing.assert_close(ref_out, graph_out, atol=1e-2, rtol=1e-2)
