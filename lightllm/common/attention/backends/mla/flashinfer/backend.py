from __future__ import annotations

from lightllm.common.attention.api.spec import CacheEncoding, AttentionFamily, AttentionPhase, AttentionFeature

import dataclasses
from typing import TYPE_CHECKING
from typing import Tuple

import torch

from lightllm.common.attention.api import WorkspaceBudget
from lightllm.common.attention.api.backend import LayerAttention
from lightllm.common.attention.api.execution import WorkspaceRequirement
from lightllm.common.attention.api.inputs import MLADecodeInputs, MLAPrefillInputs
from lightllm.common.attention.backends._flashinfer import set_flashinfer_envs
from lightllm.common.attention.api.backend import AttentionContext, AttentionBackend
from lightllm.utils.dist_utils import get_dp_world_size

if TYPE_CHECKING:
    from flashinfer.mla import BatchMLAPagedAttentionWrapper
    from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper


@dataclasses.dataclass(slots=True)
class MlaFlashInferPrefillContext(AttentionContext):
    prefill_wrapper: BatchPrefillWithRaggedKVCacheWrapper | None = None


@dataclasses.dataclass(slots=True)
class MlaFlashInferDecodeContext(AttentionContext):
    kv_indices: torch.Tensor = None
    kv_starts: torch.Tensor = None
    q_indptr: torch.Tensor = None
    decode_wrapper: BatchMLAPagedAttentionWrapper | None = None


class FlashInferMLACommon:
    supports_expert_parallel = False

    provider = "flashinfer"

    dependency = "flashinfer"

    activation_dtypes = frozenset({torch.float16, torch.bfloat16})

    cache_encodings = frozenset({CacheEncoding.PLAIN})

    family = AttentionFamily.MLA

    workspace_buffer_key = "flashinfer_mla"

    workspace_buffer_size = 256 * 1024 * 1024

    def __init__(self, env, spec):
        import triton

        set_flashinfer_envs()
        super().__init__(env=env, spec=spec)
        self.tp_q_head_num = spec.num_q_heads
        self.qk_nope_head_dim = spec.head_size - spec.config.rope_head_dim
        self.qk_rope_head_dim = spec.config.rope_head_dim
        self.kv_lora_rank = spec.value_head_size
        self.v_head_dim = spec.config.value_head_dim
        self.q_data_type = spec.dtype
        self.kv_data_type = spec.cache.dtype
        self.max_seq_length = env.max_seq_len
        self.max_page_num = triton.cdiv(self.max_seq_length, self.kernel_page_size)

    def resource_requirements(self, capacity, *, capture):
        return WorkspaceBudget(
            (WorkspaceRequirement(self.workspace_buffer_key, self.workspace_buffer_size, torch.uint8),)
        )


class FlashInferMLAPrefillBackend(
    FlashInferMLACommon, AttentionBackend[MlaFlashInferPrefillContext, MLAPrefillInputs, torch.Tensor]
):
    phase = AttentionPhase.PREFILL
    supported_features = frozenset(
        {AttentionFeature.MTP, AttentionFeature.DYNAMIC_VERIFY, AttentionFeature.NONCAUSAL, AttentionFeature.MTP_WINDOW}
    )

    def _run_prefill(
        self,
        layer,
        context,
        q: torch.Tensor,
        k: Tuple[torch.Tensor, torch.Tensor],
        v: torch.Tensor,
    ) -> torch.Tensor:
        k_nope, k_rope = k
        o_tensor = self.alloc_tensor((q.shape[0], q.shape[1], v.shape[-1]), dtype=q.dtype, device=q.device)
        q_head_num = q.shape[1]
        k = torch.cat([k_nope, torch.repeat_interleave(k_rope, q_head_num, dim=-2)], dim=-1)
        context.prefill_wrapper.run(q, k, v, out=o_tensor)
        return o_tensor

    def allocate_context(self, capacity, *, capture, workspace) -> MlaFlashInferPrefillContext:
        return MlaFlashInferPrefillContext(capacity, capture, workspace)

    def plan_host(self, context: MlaFlashInferPrefillContext):
        import flashinfer

        q_starts = context.batch.b1_cu_q_seq_len.int()
        kv_starts = context.batch.b1_cu_kv_seq_len.int()
        if context.prefill_wrapper is None:
            context.prefill_wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
                context.workspace.scratch(self.workspace_buffer_key, self.workspace_buffer_size), "NHD"
            )
        context.prefill_wrapper.plan(
            qo_indptr=q_starts,
            kv_indptr=kv_starts,
            num_qo_heads=self.tp_q_head_num,
            num_kv_heads=self.tp_q_head_num,
            head_dim_qk=self.qk_nope_head_dim + self.qk_rope_head_dim,
            head_dim_vo=self.v_head_dim,
            q_data_type=self.q_data_type,
            causal=self.spec.causal,
            sm_scale=self.spec.softmax_scale,
        )
        return

    def forward(
        self, layer: LayerAttention, context: MlaFlashInferPrefillContext, inputs: MLAPrefillInputs, *, out=None
    ) -> torch.Tensor:
        q = inputs.query
        k = (inputs.key_nope, inputs.key_rope)
        v = inputs.value
        return self._run_prefill(layer, context, q=q, k=k, v=v)


class FlashInferMLADecodeBackend(
    FlashInferMLACommon, AttentionBackend[MlaFlashInferDecodeContext, MLADecodeInputs, torch.Tensor]
):
    phase = AttentionPhase.DECODE
    supported_features = frozenset({AttentionFeature.MTP, AttentionFeature.DYNAMIC_VERIFY, AttentionFeature.MTP_WINDOW})

    def _refresh_decode_plan(self, context, kv_lens: torch.Tensor, max_kv_len: int):
        from lightllm.common.basemodel.triton_kernel.flashinfer_mla_plan import fill_mla_decode_plan_for_cuda_graph

        fill_mla_decode_plan_for_cuda_graph(
            decode_wrapper=context.decode_wrapper,
            kv_page_indptr=context.kv_starts,
            kv_lens=kv_lens,
            batch_size=context.batch.batch_size,
            num_heads=self.tp_q_head_num,
            max_kv_len=max_kv_len,
        )

    def _run_decode(
        self,
        layer,
        context,
        q: Tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
    ):
        qk_rope_head_dim = layer.resources.qk_rope_head_dim
        q_nope, q_rope = q
        o_tensor = self.alloc_tensor(q_nope.shape, dtype=q_nope.dtype, device=q_nope.device)
        context.decode_wrapper.run(
            q_nope,
            q_rope,
            k[:, :, :-qk_rope_head_dim].view(-1, self.kernel_page_size, 1, k.shape[-1] - qk_rope_head_dim),
            k[:, :, -qk_rope_head_dim:].view(-1, self.kernel_page_size, 1, qk_rope_head_dim),
            out=o_tensor,
            return_lse=False,
        )
        return o_tensor

    def allocate_context(self, capacity, *, capture, workspace) -> MlaFlashInferDecodeContext:
        return MlaFlashInferDecodeContext(capacity, capture, workspace)

    def forward(
        self, layer: LayerAttention, context: MlaFlashInferDecodeContext, inputs: MLADecodeInputs, *, out=None
    ) -> torch.Tensor:
        q = (inputs.query_nope, inputs.query_rope)
        k = layer.resources.kv
        return self._run_decode(layer, context, q=q, k=k)

    def plan_host(self, context: MlaFlashInferDecodeContext):
        from lightllm.common.basemodel.triton_kernel.gen_prefill_params import gen_cumsum_pad0_tensor
        from lightllm.common.basemodel.triton_kernel.repack_kv_index import repack_kv_index
        import flashinfer

        batch = context.batch
        b_page_len = (batch.b_seq_len + self.kernel_page_size - 1) // self.kernel_page_size
        starts, _ = gen_cumsum_pad0_tensor(b_page_len, b_page_len)
        if context.kv_starts is None:
            context.kv_starts = starts
            context.kv_indices = context.workspace.scratch(
                "flashinfer_indices", max(1, batch.batch_size * self.max_page_num), torch.int32
            )
            context.q_indptr = torch.arange(batch.batch_size + 1, dtype=torch.int32, device=batch.device)
        else:
            context.kv_starts.copy_(starts, non_blocking=True)
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
            context.decode_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
                context.workspace.scratch(self.workspace_buffer_key, self.workspace_buffer_size),
                use_cuda_graph=context.capture,
                qo_indptr=context.q_indptr,
                kv_indices=context.kv_indices,
                kv_indptr=context.kv_starts,
                kv_len_arr=batch.b_seq_len,
            )
            context.decode_wrapper.plan(
                context.q_indptr,
                context.kv_starts,
                context.kv_indices,
                batch.b_seq_len,
                self.tp_q_head_num,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                self.kernel_page_size,
                False,
                self.spec.softmax_scale,
                self.q_data_type,
                self.kv_data_type,
            )
        else:
            self._refresh_decode_plan(context, batch.b_seq_len, context.current_max_kv_seq_len)
