from __future__ import annotations

from lightllm.common.attention.api.spec import CacheEncoding, AttentionFamily, AttentionPhase, AttentionFeature

import dataclasses
from typing import TYPE_CHECKING

import torch

from lightllm.common.attention.api import WorkspaceBudget
from lightllm.common.attention.api.backend import LayerAttention
from lightllm.common.attention.api.execution import WorkspaceRequirement
from lightllm.common.attention.api.inputs import DenseInputs
from lightllm.common.attention.backends._flashinfer import set_flashinfer_envs
from lightllm.common.attention.api.backend import AttentionContext, AttentionBackend
from lightllm.utils.dist_utils import get_dp_world_size

if TYPE_CHECKING:
    from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
    from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper


@dataclasses.dataclass(slots=True)
class FlashInferPrefillContext(AttentionContext):
    prefill_wrapper: BatchPrefillWithPagedKVCacheWrapper | None = None


@dataclasses.dataclass(slots=True)
class FlashInferDecodeContext(AttentionContext):
    kv_last_page_len_buffer: torch.Tensor = None
    kv_indices: torch.Tensor = None
    kv_starts: torch.Tensor = None
    decode_wrapper: BatchDecodeWithPagedKVCacheWrapper | None = None


class FlashInferDenseCommon:
    family = AttentionFamily.DENSE
    supports_expert_parallel = False

    provider = "flashinfer"

    dependency = "flashinfer"

    activation_dtypes = frozenset({torch.float16, torch.bfloat16})

    cache_encodings = frozenset({CacheEncoding.PLAIN})

    workspace_buffer_key = "flashinfer_fp"

    workspace_buffer_size = 512 * 1024 * 1024

    def __init__(self, env, spec):
        import triton

        set_flashinfer_envs()
        super().__init__(env=env, spec=spec)
        self.tp_q_head_num = spec.num_q_heads
        self.tp_kv_head_num = spec.cache.key_heads
        self.head_dim = spec.head_size
        self.max_seq_length = env.max_seq_len
        self.max_page_num = triton.cdiv(self.max_seq_length, self.kernel_page_size)
        self.q_data_type = spec.dtype
        self.kv_data_type = spec.cache.dtype

    def resource_requirements(self, capacity, *, capture):
        return WorkspaceBudget(
            (WorkspaceRequirement(self.workspace_buffer_key, self.workspace_buffer_size, torch.uint8),)
        )


class FlashInferDensePrefillBackend(
    FlashInferDenseCommon, AttentionBackend[FlashInferPrefillContext, DenseInputs, torch.Tensor]
):
    phase = AttentionPhase.PREFILL
    supported_features = frozenset(
        {AttentionFeature.MTP, AttentionFeature.DYNAMIC_VERIFY, AttentionFeature.NONCAUSAL, AttentionFeature.MTP_WINDOW}
    )

    def _prefill_paged(self, layer, context, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        o_tensor = self.alloc_tensor(q.shape, q.dtype, device=q.device)
        context.prefill_wrapper.run(
            q,
            (
                k.view(-1, self.kernel_page_size, k.shape[1], k.shape[2]),
                v.view(-1, self.kernel_page_size, v.shape[1], v.shape[2]),
            ),
            out=o_tensor,
        )
        return o_tensor

    def allocate_context(self, capacity, *, capture, workspace) -> FlashInferPrefillContext:
        return FlashInferPrefillContext(capacity, capture, workspace)

    def forward(
        self, layer: LayerAttention, context: FlashInferPrefillContext, inputs: DenseInputs, *, out=None
    ) -> torch.Tensor:
        q = inputs.query
        k = layer.resources.key
        v = layer.resources.value
        return self._prefill_paged(layer, context, q=q, k=k, v=v)

    def plan_host(self, context: FlashInferPrefillContext):
        from lightllm.common.basemodel.triton_kernel.gen_prefill_params import gen_cumsum_pad0_tensor
        from lightllm.common.basemodel.triton_kernel.repack_kv_index import repack_kv_index
        import flashinfer

        batch = context.batch
        q_starts = batch.b1_cu_q_seq_len.int()
        b_page_len = (batch.b_seq_len + self.kernel_page_size - 1) // self.kernel_page_size
        kv_starts, _ = gen_cumsum_pad0_tensor(b_page_len, b_page_len)
        last_page_len = batch.b_seq_len - (b_page_len - 1) * self.kernel_page_size
        indices = context.workspace.scratch(
            "flashinfer_indices", max(1, batch.batch_size * self.max_page_num), torch.int32
        )
        repack_kv_index(
            req_to_token_indexs=batch.req_to_token_indexs,
            b_req_idx=batch.b_req_idx,
            b_token_len=batch.b_seq_len,
            b_page_start_loc=kv_starts[:-1],
            max_token_len=batch.max_kv_seq_len,
            out_page_indices=indices,
            page_size=self.kernel_page_size,
        )
        context.prefill_wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            context.workspace.scratch(self.workspace_buffer_key, self.workspace_buffer_size),
            qo_indptr_buf=q_starts,
            paged_kv_indptr_buf=kv_starts,
            paged_kv_indices_buf=indices,
            paged_kv_last_page_len_buf=last_page_len,
        )
        context.prefill_wrapper.plan(
            q_starts,
            kv_starts,
            indices,
            last_page_len,
            self.tp_q_head_num,
            self.tp_kv_head_num,
            self.head_dim,
            self.kernel_page_size,
            causal=self.spec.causal,
            pos_encoding_mode="NONE",
            logits_soft_cap=0.0,
            q_data_type=self.q_data_type,
            kv_data_type=self.kv_data_type,
        )


class FlashInferDenseDecodeBackend(
    FlashInferDenseCommon, AttentionBackend[FlashInferDecodeContext, DenseInputs, torch.Tensor]
):
    phase = AttentionPhase.DECODE
    supported_features = frozenset({AttentionFeature.MTP, AttentionFeature.DYNAMIC_VERIFY, AttentionFeature.MTP_WINDOW})

    def _refresh_decode_plan(self, context, max_kv_len: int):
        import triton
        from flashinfer.decode import fast_decode_plan

        uniform_kv_indptr_cpu = torch.arange(
            context.batch.batch_size + 1, dtype=torch.int32, device="cpu"
        ) * triton.cdiv(max_kv_len, self.kernel_page_size)
        fast_decode_plan(
            context.decode_wrapper,
            indptr=context.kv_starts,
            indices=context.kv_indices,
            last_page_len=context.kv_last_page_len_buffer,
            num_qo_heads=self.tp_q_head_num,
            num_kv_heads=self.tp_kv_head_num,
            head_dim=self.head_dim,
            page_size=self.kernel_page_size,
            q_data_type=self.q_data_type,
            kv_data_type=self.kv_data_type,
            non_blocking=True,
            global_override_indptr_cpu=uniform_kv_indptr_cpu,
        )
        return

    def _run_decode(self, layer, context, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        o_tensor = self.alloc_tensor(q.shape, q.dtype)
        context.decode_wrapper.run(
            q,
            (
                k.view(-1, self.kernel_page_size, k.shape[1], k.shape[2]),
                v.view(-1, self.kernel_page_size, v.shape[1], v.shape[2]),
            ),
            out=o_tensor,
        )
        return o_tensor

    def allocate_context(self, capacity, *, capture, workspace) -> FlashInferDecodeContext:
        return FlashInferDecodeContext(capacity, capture, workspace)

    def forward(
        self, layer: LayerAttention, context: FlashInferDecodeContext, inputs: DenseInputs, *, out=None
    ) -> torch.Tensor:
        q = inputs.query
        k = layer.resources.key
        v = layer.resources.value
        return self._run_decode(layer, context, q=q, k=k, v=v)

    def plan_host(self, context: FlashInferDecodeContext):
        from lightllm.common.basemodel.triton_kernel.gen_prefill_params import gen_cumsum_pad0_tensor
        from lightllm.common.basemodel.triton_kernel.repack_kv_index import repack_kv_index
        import flashinfer

        batch = context.batch
        b_page_len = (batch.b_seq_len + self.kernel_page_size - 1) // self.kernel_page_size
        starts, _ = gen_cumsum_pad0_tensor(b_page_len, b_page_len)
        last_page_len = batch.b_seq_len - (b_page_len - 1) * self.kernel_page_size
        if context.kv_starts is None:
            context.kv_starts, context.kv_last_page_len_buffer = starts, last_page_len
            context.kv_indices = context.workspace.scratch(
                "flashinfer_indices", max(1, batch.batch_size * self.max_page_num), torch.int32
            )
        else:
            context.kv_starts.copy_(starts, non_blocking=True)
            context.kv_last_page_len_buffer.copy_(last_page_len, non_blocking=True)
        repack_kv_index(
            req_to_token_indexs=batch.req_to_token_indexs,
            b_req_idx=batch.b_req_idx,
            b_token_len=batch.b_seq_len,
            b_page_start_loc=context.kv_starts[:-1],
            max_token_len=context.current_max_kv_seq_len,
            out_page_indices=context.kv_indices,
            page_size=self.kernel_page_size,
        )
        if context.decode_wrapper is None:
            context.decode_wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
                context.workspace.scratch(self.workspace_buffer_key, self.workspace_buffer_size),
                "NHD",
                use_cuda_graph=context.capture,
                use_tensor_cores=True,
                paged_kv_indptr_buffer=context.kv_starts,
                paged_kv_indices_buffer=context.kv_indices,
                paged_kv_last_page_len_buffer=context.kv_last_page_len_buffer,
            )
            context.decode_wrapper.plan(
                context.kv_starts,
                context.kv_indices,
                context.kv_last_page_len_buffer,
                self.tp_q_head_num,
                self.tp_kv_head_num,
                self.head_dim,
                self.kernel_page_size,
                q_data_type=self.q_data_type,
                kv_data_type=self.kv_data_type,
                non_blocking=True,
            )
        else:
            self._refresh_decode_plan(context, context.current_max_kv_seq_len)
