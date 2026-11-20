from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import triton

from lightllm.common.attention.backend import AttentionBackend, TensorAllocator
from . import GQAInputs

if TYPE_CHECKING:
    from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
    from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper


@dataclass(frozen=True)
class FlashInferPrefillBatch:
    """A planned NHD wrapper for packed, causal prefill.

    The plan uses scale 1/sqrt(head_dim) and no custom mask, sliding window,
    positional encoding or soft cap. Page index p addresses cache slots
    [p * page_size, (p + 1) * page_size). The wrapper borrows the caller's
    workspace and page metadata, which must stay alive and unshared until
    the last layer of the pass has run.
    """

    wrapper: "BatchPrefillWithPagedKVCacheWrapper"
    page_size: int


@dataclass(frozen=True)
class FlashInferDecodeBatch:
    """A planned NHD tensor-core wrapper for one query per request.

    Storage follows FlashInferPrefillBatch's contract. The page metadata
    buffers and planned configuration are kept so refresh can re-plan the
    wrapper in place for CUDA graph replay.
    """

    wrapper: "BatchDecodeWithPagedKVCacheWrapper"
    page_size: int
    kv_starts: torch.Tensor
    kv_indices: torch.Tensor
    kv_last_page_len: torch.Tensor
    q_heads: int
    kv_heads: int
    head_dim: int
    q_dtype: torch.dtype
    kv_dtype: torch.dtype


class FlashInferGQAPrefillBackend(AttentionBackend[FlashInferPrefillBatch, GQAInputs]):
    def plan(
        self,
        *,
        workspace: torch.Tensor,
        q_starts: torch.Tensor,
        kv_starts: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_last_page_len: torch.Tensor,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        page_size: int,
        q_dtype: torch.dtype,
        kv_dtype: torch.dtype,
    ) -> FlashInferPrefillBatch:
        """Build and plan a prefill wrapper over caller-owned workspace and page metadata.

        q_starts and kv_starts are int32 [batch + 1] prefix offsets of queries and pages;
        kv_indices lists each request's pages in order and kv_last_page_len gives the
        valid tokens in each request's final page.
        """
        import flashinfer

        wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            workspace,
            qo_indptr_buf=q_starts,
            paged_kv_indptr_buf=kv_starts,
            paged_kv_indices_buf=kv_indices,
            paged_kv_last_page_len_buf=kv_last_page_len,
        )
        wrapper.plan(
            q_starts,
            kv_starts,
            kv_indices,
            kv_last_page_len,
            q_heads,
            kv_heads,
            head_dim,
            page_size,
            causal=True,
            pos_encoding_mode="NONE",
            logits_soft_cap=0.0,
            q_data_type=q_dtype,
            kv_data_type=kv_dtype,
        )
        return FlashInferPrefillBatch(wrapper=wrapper, page_size=page_size)

    def forward(
        self,
        batch: FlashInferPrefillBatch,
        inputs: GQAInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Fill and return out with q's shape, dtype and device.

        Q/out are contiguous FP16/BF16 tensors on the same CUDA device as
        the cache and plan. Cache slots must form complete pages; head_dim
        and head strides are contiguous, while the slot stride may include
        padding (including K/V slices of shared storage). Q/K/V are read-only;
        out must not overlap them or the plan's metadata/workspace.
        Only sliding_window=(-1, -1), the full causal prefix, is supported.
        Scratch is already owned by the planned wrapper, so forward does not
        call alloc_func.
        """
        assert inputs.sliding_window == (-1, -1), "FlashInfer sliding_window is not supported"
        batch.wrapper.run(
            q=inputs.q,
            paged_kv_cache=(
                inputs.k_cache.view(-1, batch.page_size, inputs.k_cache.shape[1], inputs.k_cache.shape[2]),
                inputs.v_cache.view(-1, batch.page_size, inputs.v_cache.shape[1], inputs.v_cache.shape[2]),
            ),
            out=out,
        )
        return out


class FlashInferGQADecodeBackend(AttentionBackend[FlashInferDecodeBatch, GQAInputs]):
    def plan(
        self,
        *,
        workspace: torch.Tensor,
        kv_starts: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_last_page_len: torch.Tensor,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        page_size: int,
        q_dtype: torch.dtype,
        kv_dtype: torch.dtype,
    ) -> FlashInferDecodeBatch:
        """Build and plan a CUDA-graph-capable decode wrapper over caller-owned buffers.

        The page metadata follows FlashInferGQAPrefillBackend.plan, with one query per
        request. For graph capture these buffers must keep their addresses so that
        replay can rewrite them in place before refresh.
        """
        import flashinfer

        wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
            workspace,
            "NHD",
            use_cuda_graph=True,
            use_tensor_cores=True,
            paged_kv_indptr_buffer=kv_starts,
            paged_kv_indices_buffer=kv_indices,
            paged_kv_last_page_len_buffer=kv_last_page_len,
        )
        wrapper.plan(
            kv_starts,
            kv_indices,
            kv_last_page_len,
            q_heads,
            kv_heads,
            head_dim,
            page_size,
            q_data_type=q_dtype,
            kv_data_type=kv_dtype,
            non_blocking=True,
        )
        return FlashInferDecodeBatch(
            wrapper=wrapper,
            page_size=page_size,
            kv_starts=kv_starts,
            kv_indices=kv_indices,
            kv_last_page_len=kv_last_page_len,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
        )

    def refresh(self, batch: FlashInferDecodeBatch, *, max_kv_len: int) -> None:
        """Re-plan the captured wrapper after the caller rewrites its page metadata for replay.

        max_kv_len is the replay batch's longest request. The host-side schedule assumes
        every request has that many pages, so the plan stays valid for the actual lengths.
        """
        from flashinfer.decode import fast_decode_plan

        batch_size = batch.kv_last_page_len.shape[0]
        uniform_kv_indptr_cpu = torch.arange(batch_size + 1, dtype=torch.int32, device="cpu") * triton.cdiv(
            max_kv_len, batch.page_size
        )
        fast_decode_plan(
            batch.wrapper,
            indptr=batch.kv_starts,
            indices=batch.kv_indices,
            last_page_len=batch.kv_last_page_len,
            num_qo_heads=batch.q_heads,
            num_kv_heads=batch.kv_heads,
            head_dim=batch.head_dim,
            page_size=batch.page_size,
            q_data_type=batch.q_dtype,
            kv_data_type=batch.kv_dtype,
            non_blocking=True,
            global_override_indptr_cpu=uniform_kv_indptr_cpu,
        )

    def forward(
        self,
        batch: FlashInferDecodeBatch,
        inputs: GQAInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Fill and return out with q's shape, dtype and device.

        Tensor layout, mutation, allocation and sliding_window rules match
        the prefill backend, with one query per request in the planned batch
        order.
        """
        assert inputs.sliding_window == (-1, -1), "FlashInfer sliding_window is not supported"
        batch.wrapper.run(
            q=inputs.q,
            paged_kv_cache=(
                inputs.k_cache.view(-1, batch.page_size, inputs.k_cache.shape[1], inputs.k_cache.shape[2]),
                inputs.v_cache.view(-1, batch.page_size, inputs.v_cache.shape[1], inputs.v_cache.shape[2]),
            ),
            out=out,
        )
        return out
