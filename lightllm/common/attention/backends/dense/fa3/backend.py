from lightllm.common.attention.api.spec import CacheEncoding, AttentionFamily, AttentionPhase, AttentionFeature
import dataclasses

import torch

from lightllm.common.attention.api.backend import LayerAttention
from lightllm.common.attention.api.inputs import DenseInputs
from lightllm.common.attention.api.backend import AttentionContext, AttentionBackend
from lightllm.utils.envs_utils import enable_env_vars


@dataclasses.dataclass(slots=True)
class Fa3PrefillContext(AttentionContext):
    cu_seqlens_q: torch.Tensor = None
    cu_seqlens_k: torch.Tensor = None
    page_table: torch.Tensor = None
    causal: bool = None


@dataclasses.dataclass(slots=True)
class Fa3DecodeContext(AttentionContext):
    cu_seqlens_q: torch.Tensor = None
    cu_seqlens_k: torch.Tensor = None
    page_table: torch.Tensor = None
    b_att_seq_len: torch.Tensor = None
    decode_max_q_seq_len: int = None
    decode_max_kv_seq_len: int = None
    causal: bool = None


class Fa3DenseCommon:
    family = AttentionFamily.DENSE
    supported_features = frozenset(
        {
            AttentionFeature.MTP,
            AttentionFeature.DYNAMIC_VERIFY,
            AttentionFeature.SINK,
            AttentionFeature.SLIDING_WINDOW,
            AttentionFeature.NONCAUSAL,
            AttentionFeature.MTP_WINDOW,
        }
    )

    requires_cpu_cache_sync = enable_env_vars("LIGHTLLM_FA3_CPU_CACHE_SYNC")

    provider = "fa3"

    dependency = "sgl_kernel.flash_attn"

    min_device_capability = (9, 0)

    activation_dtypes = frozenset({torch.float16, torch.bfloat16})

    max_head_size = 256

    cache_encodings = frozenset({CacheEncoding.PLAIN})


class Fa3DensePrefillBackend(Fa3DenseCommon, AttentionBackend[Fa3PrefillContext, DenseInputs, torch.Tensor]):
    phase = AttentionPhase.PREFILL

    def _prefill_paged(self, layer, context, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        from lightllm.utils.sgl_utils import flash_attn_with_kvcache

        if self.spec.sliding_window is not None:
            window_size = self.spec.sliding_window
        else:
            window_size = (-1, -1)
        if layer.resources.sink is not None:
            sink_weight: torch.Tensor = layer.resources.sink
        else:
            sink_weight = None
        k_descale, v_descale = (None, None)
        Lq = q.shape[-1]
        sm_scale = 1.0 / Lq ** 0.5
        o = flash_attn_with_kvcache(
            q=q,
            k_cache=k.view(-1, self.kernel_page_size, k.shape[1], k.shape[2]),
            v_cache=v.view(-1, self.kernel_page_size, v.shape[1], v.shape[2]),
            page_table=context.page_table,
            cache_seqlens=context.batch.b_seq_len,
            cu_seqlens_q=context.cu_seqlens_q,
            cu_seqlens_k_new=context.cu_seqlens_k,
            max_seqlen_q=context.batch.max_q_seq_len,
            softmax_scale=sm_scale,
            causal=context.causal,
            window_size=window_size,
            softcap=0.0,
            k_descale=k_descale,
            v_descale=v_descale,
            return_softmax_lse=False,
            sinks=sink_weight,
        )
        return o

    def allocate_context(self, capacity, *, capture, workspace) -> Fa3PrefillContext:
        return Fa3PrefillContext(capacity, capture, workspace)

    def build_device_metadata(self, context: Fa3PrefillContext):
        from lightllm.common.basemodel.triton_kernel.fa3_utils import page_table_copy
        import triton

        context.causal = self.spec.causal
        context.cu_seqlens_q = context.batch.b1_cu_q_seq_len.int()
        context.cu_seqlens_k = context.batch.b1_cu_kv_seq_len.int()
        context.page_table = torch.empty(
            (context.batch.batch_size, triton.cdiv(context.batch.max_kv_seq_len, self.kernel_page_size)),
            dtype=torch.int32,
            device=context.batch.device,
        )
        page_table_copy(
            page_table=context.page_table,
            req_to_token_indexs=context.batch.req_to_token_indexs,
            b_req_idx=context.batch.b_req_idx,
            page_size=self.kernel_page_size,
        )

    def forward(
        self, layer: LayerAttention, context: Fa3PrefillContext, inputs: DenseInputs, *, out=None
    ) -> torch.Tensor:
        q = inputs.query
        k = layer.resources.key
        v = layer.resources.value
        return self._prefill_paged(layer, context, q=q, k=k, v=v)


class Fa3DenseDecodeBackend(Fa3DenseCommon, AttentionBackend[Fa3DecodeContext, DenseInputs, torch.Tensor]):
    phase = AttentionPhase.DECODE

    def _decode_init_dynamic_verify_metadata(self, context, draft_step: int) -> torch.Tensor:
        from lightllm.common.basemodel.triton_kernel.fa3_utils import build_dynamic_spec_fa3_decode_params
        from lightllm.common.basemodel.triton_kernel.mtp_utils import build_mtp_shared_group_markers

        b_mark_mtp_shared_group = build_mtp_shared_group_markers(context.batch.b_req_idx, hold_req_id=self.hold_req_id)
        b_q_seq_len, b_kv_seq_len, b_att_req_idx, context.b_att_seq_len = build_dynamic_spec_fa3_decode_params(
            b_req_idx=context.batch.b_req_idx,
            b_seq_len=context.batch.b_seq_len,
            b_mark_mtp_shared_group=b_mark_mtp_shared_group,
            att_batch_size=context.batch.batch_size,
            hold_req_id=self.hold_req_id,
        )
        self._decode_init_mtp_cu_seqlens(context, b_q_seq_len, b_kv_seq_len)
        context.decode_max_q_seq_len = draft_step + 1
        return b_att_req_idx

    def _decode_init_fixed_mtp_metadata(self, context, draft_step: int) -> torch.Tensor:
        mtp_size = draft_step + 1
        assert context.batch.batch_size % mtp_size == 0, (
            "FA3 fixed-layout decode requires batch_size to be divisible by draft_step + 1, "
            f"got batch_size={context.batch.batch_size}, draft_step={draft_step}."
        )
        b_q_seq_len = torch.full(
            (context.batch.b_seq_len.shape[0] // mtp_size,),
            fill_value=mtp_size,
            dtype=torch.int32,
            device=context.batch.b_seq_len.device,
        )
        b_kv_seq_len = context.batch.b_seq_len[draft_step::mtp_size]
        b_att_req_idx = context.batch.b_req_idx[draft_step::mtp_size]
        context.b_att_seq_len = b_kv_seq_len.contiguous()
        self._decode_init_mtp_cu_seqlens(context, b_q_seq_len, b_kv_seq_len)
        context.decode_max_q_seq_len = mtp_size
        return b_att_req_idx

    def _decode_init_normal_metadata(self, context) -> torch.Tensor:
        context.cu_seqlens_q = context.batch.b1_cu_q_seq_len.int()
        context.cu_seqlens_k = context.batch.b1_cu_kv_seq_len.int()
        context.b_att_seq_len = context.batch.b_seq_len
        context.decode_max_q_seq_len = 1
        return context.batch.b_req_idx

    def _decode_init_mtp_cu_seqlens(self, context, b_q_seq_len: torch.Tensor, b_kv_seq_len: torch.Tensor):
        from lightllm.common.basemodel.triton_kernel.gen_prefill_params import gen_cumsum_pad0_tensor

        b1_cu_q_seq_len, b1_cu_kv_seq_len = gen_cumsum_pad0_tensor(b_q_seq_len, b_kv_seq_len)
        context.cu_seqlens_q = b1_cu_q_seq_len.int()
        context.cu_seqlens_k = b1_cu_kv_seq_len.int()

    def _decode_init_page_table(self, context, b_att_req_idx):
        from lightllm.common.basemodel.triton_kernel.fa3_utils import page_table_copy
        import triton

        context.decode_max_kv_seq_len = context.batch.max_kv_seq_len
        width = triton.cdiv(context.capacity.max_kv_seq_len, self.kernel_page_size)
        context.page_table = context.workspace.scratch("fa3_page_table", b_att_req_idx.numel() * width, torch.int32)[
            : b_att_req_idx.numel() * width
        ].view(b_att_req_idx.numel(), width)
        page_table_copy(
            page_table=context.page_table,
            req_to_token_indexs=context.batch.req_to_token_indexs,
            b_req_idx=b_att_req_idx,
            page_size=self.kernel_page_size,
        )

    def _run_decode(self, layer, context, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        from lightllm.utils.sgl_utils import flash_attn_with_kvcache_autotune

        if self.spec.sliding_window is not None:
            window_size = self.spec.sliding_window
        else:
            window_size = (-1, -1)
        if layer.resources.sink is not None:
            sink_weight: torch.Tensor = layer.resources.sink
        else:
            sink_weight = None
        k_descale, v_descale = (None, None)
        Lq = q.shape[-1]
        sm_scale = 1.0 / Lq ** 0.5
        o = flash_attn_with_kvcache_autotune(
            q=q,
            k_cache=k.view(-1, self.kernel_page_size, k.shape[1], k.shape[2]),
            v_cache=v.view(-1, self.kernel_page_size, v.shape[1], v.shape[2]),
            page_table=context.page_table,
            cache_seqlens=context.b_att_seq_len,
            cu_seqlens_q=context.cu_seqlens_q,
            cu_seqlens_k_new=None,
            max_seqlen_q=context.decode_max_q_seq_len,
            max_seqlen_k=context.decode_max_kv_seq_len,
            softmax_scale=sm_scale,
            causal=context.causal,
            window_size=window_size,
            softcap=0.0,
            k_descale=k_descale,
            v_descale=v_descale,
            return_softmax_lse=False,
            sinks=sink_weight,
        )
        return o

    def allocate_context(self, capacity, *, capture, workspace) -> Fa3DecodeContext:
        return Fa3DecodeContext(capacity, capture, workspace)

    def build_device_metadata(self, context: Fa3DecodeContext):
        context.causal = self.spec.causal
        draft_step = self.draft_step
        if self.spec.dynamic_verify:
            b_att_req_idx = self._decode_init_dynamic_verify_metadata(context, draft_step)
        elif draft_step > 0:
            b_att_req_idx = self._decode_init_fixed_mtp_metadata(context, draft_step)
        else:
            b_att_req_idx = self._decode_init_normal_metadata(context)
        self._decode_init_page_table(context, b_att_req_idx)

    def forward(
        self, layer: LayerAttention, context: Fa3DecodeContext, inputs: DenseInputs, *, out=None
    ) -> torch.Tensor:
        q = inputs.query
        k = layer.resources.key
        v = layer.resources.value
        return self._run_decode(layer, context, q=q, k=k, v=v)
