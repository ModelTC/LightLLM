import dataclasses

import torch
import triton

from lightllm.common.attention.api.backend import LayerAttentionBinding
from lightllm.common.attention.api.inputs import DenseInputs
from lightllm.common.attention.backends.base import BaseAttentionBackend, PreparedPlan
from lightllm.common.basemodel.triton_kernel.fa3_utils import build_dynamic_spec_fa3_decode_params, page_table_copy
from lightllm.common.basemodel.triton_kernel.gen_prefill_params import gen_cumsum_pad0_tensor
from lightllm.common.basemodel.triton_kernel.mtp_utils import build_mtp_shared_group_markers
from lightllm.utils.envs_utils import enable_env_vars
from lightllm.utils.sgl_utils import flash_attn_with_kvcache, flash_attn_with_kvcache_autotune


@dataclasses.dataclass
class Fa3PrefillPlan(PreparedPlan):
    cu_seqlens_q: torch.Tensor = None
    cu_seqlens_k: torch.Tensor = None
    page_table: torch.Tensor = None
    causal: bool = None


@dataclasses.dataclass
class Fa3DecodePlan(PreparedPlan):
    cu_seqlens_q: torch.Tensor = None
    cu_seqlens_k: torch.Tensor = None
    page_table: torch.Tensor = None
    b_att_seq_len: torch.Tensor = None
    decode_max_q_seq_len: int = None
    decode_max_kv_seq_len: int = None
    causal: bool = None


class Fa3AttBackend(BaseAttentionBackend):
    requires_cpu_cache_sync = enable_env_vars("LIGHTLLM_FA3_CPU_CACHE_SYNC")
    provider = "fa3"
    dependency = "sgl_kernel.flash_attn"
    min_device_capability = (9, 0)
    activation_dtypes = frozenset({torch.float16, torch.bfloat16})
    supports_attention_sink = True
    supports_sliding_window = True
    supports_noncausal_decode = True
    max_head_size = 256
    cache_formats = frozenset({"None"})
    """Common fixed page-table storage for FA3 attention backends."""

    def __init__(self, model):
        super().__init__(model=model)
        self._init_infer_page_size()

    def _init_infer_page_size(self):
        self.infer_page_size = self.page_size
        assert (
            self.page_size % self.infer_page_size == 0
        ), f"model page_size {self.page_size} must be divisible by infer_page_size {self.infer_page_size}"

    def _allocate_prefill_plan(self, capacity, capture, workspace):
        return Fa3PrefillPlan(capacity, capture, workspace)

    def _allocate_decode_plan(self, capacity, capture, workspace):
        return Fa3DecodePlan(capacity, capture, workspace)

    def _update_prefill(self, plan):
        plan.causal = self.uses_causal_attention()
        plan.cu_seqlens_q = plan.batch.b1_cu_q_seq_len.int()
        plan.cu_seqlens_k = plan.batch.b1_cu_kv_seq_len.int()
        plan.page_table = torch.empty(
            (plan.batch.batch_size, triton.cdiv(plan.batch.max_kv_seq_len, self.infer_page_size)),
            dtype=torch.int32,
            device=plan.batch.device,
        )
        page_table_copy(
            page_table=plan.page_table,
            req_to_token_indexs=plan.batch.req_to_token_indexs,
            b_req_idx=plan.batch.b_req_idx,
            page_size=self.infer_page_size,
        )

    def _forward_prefill(self, layer: LayerAttentionBinding, plan, inputs: DenseInputs, *, out=None) -> torch.Tensor:
        alloc_func = layer.alloc_tensor
        q = inputs.query
        k = layer.params.key
        v = layer.params.value
        return self._prefill_nomarl_prefill_att(layer, plan, q=q, k=k, v=v, inputs=inputs, alloc_func=alloc_func)

    def _prefill_nomarl_prefill_att(
        self, layer, plan, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, inputs, alloc_func=torch.empty
    ) -> torch.Tensor:
        if layer.params.sliding_window is not None:
            window_size = layer.params.sliding_window
        else:
            window_size = (-1, -1)
        if layer.params.sink is not None:
            sink_weight: torch.Tensor = layer.params.sink
        else:
            sink_weight = None
        k_descale, v_descale = (None, None)
        Lq = q.shape[-1]
        sm_scale = 1.0 / Lq ** 0.5
        o = flash_attn_with_kvcache(
            q=q,
            k_cache=k.view(-1, self.infer_page_size, k.shape[1], k.shape[2]),
            v_cache=v.view(-1, self.infer_page_size, v.shape[1], v.shape[2]),
            page_table=plan.page_table,
            cache_seqlens=plan.batch.b_seq_len,
            cu_seqlens_q=plan.cu_seqlens_q,
            cu_seqlens_k_new=plan.cu_seqlens_k,
            max_seqlen_q=plan.batch.max_q_seq_len,
            softmax_scale=sm_scale,
            causal=plan.causal,
            window_size=window_size,
            softcap=0.0,
            k_descale=k_descale,
            v_descale=v_descale,
            return_softmax_lse=False,
            sinks=sink_weight,
        )
        return o

    def _update_decode(self, plan):
        plan.causal = self.uses_causal_attention()
        draft_step = self.decode_draft_step
        if self.uses_dynamic_spec_verify_layout():
            b_att_req_idx = self._decode_init_dynamic_spec_verify_state(plan, draft_step)
        elif draft_step > 0:
            b_att_req_idx = self._decode_init_fixed_spec_decode_state(plan, draft_step)
        else:
            b_att_req_idx = self._decode_init_normal_decode_state(plan)
        self._decode_init_page_table(plan, b_att_req_idx)

    def _decode_init_dynamic_spec_verify_state(self, plan, draft_step: int) -> torch.Tensor:
        b_mark_mtp_shared_group = build_mtp_shared_group_markers(plan.batch.b_req_idx, hold_req_id=self.hold_req_id)
        b_q_seq_len, b_kv_seq_len, b_att_req_idx, plan.b_att_seq_len = build_dynamic_spec_fa3_decode_params(
            b_req_idx=plan.batch.b_req_idx,
            b_seq_len=plan.batch.b_seq_len,
            b_mark_mtp_shared_group=b_mark_mtp_shared_group,
            att_batch_size=plan.batch.batch_size,
            hold_req_id=self.hold_req_id,
        )
        self._decode_init_spec_decode_cu_seqlens(plan, b_q_seq_len, b_kv_seq_len)
        plan.decode_max_q_seq_len = draft_step + 1
        return b_att_req_idx

    def _decode_init_fixed_spec_decode_state(self, plan, draft_step: int) -> torch.Tensor:
        mtp_size = draft_step + 1
        assert plan.batch.batch_size % mtp_size == 0, (
            "FA3 fixed-layout decode requires batch_size to be divisible by draft_step + 1, "
            f"got batch_size={plan.batch.batch_size}, draft_step={draft_step}."
        )
        b_q_seq_len = torch.full(
            (plan.batch.b_seq_len.shape[0] // mtp_size,),
            fill_value=mtp_size,
            dtype=torch.int32,
            device=plan.batch.b_seq_len.device,
        )
        b_kv_seq_len = plan.batch.b_seq_len[draft_step::mtp_size]
        b_att_req_idx = plan.batch.b_req_idx[draft_step::mtp_size]
        plan.b_att_seq_len = b_kv_seq_len.contiguous()
        self._decode_init_spec_decode_cu_seqlens(plan, b_q_seq_len, b_kv_seq_len)
        plan.decode_max_q_seq_len = mtp_size
        return b_att_req_idx

    def _decode_init_normal_decode_state(self, plan) -> torch.Tensor:
        plan.cu_seqlens_q = plan.batch.b1_cu_q_seq_len.int()
        plan.cu_seqlens_k = plan.batch.b1_cu_kv_seq_len.int()
        plan.b_att_seq_len = plan.batch.b_seq_len
        plan.decode_max_q_seq_len = 1
        return plan.batch.b_req_idx

    def _decode_init_spec_decode_cu_seqlens(self, plan, b_q_seq_len: torch.Tensor, b_kv_seq_len: torch.Tensor):
        b1_cu_q_seq_len, b1_cu_kv_seq_len = gen_cumsum_pad0_tensor(b_q_seq_len, b_kv_seq_len)
        plan.cu_seqlens_q = b1_cu_q_seq_len.int()
        plan.cu_seqlens_k = b1_cu_kv_seq_len.int()

    def _decode_init_page_table(self, plan, b_att_req_idx):
        plan.decode_max_kv_seq_len = plan.batch.max_kv_seq_len
        width = triton.cdiv(plan.capacity.max_kv_seq_len, self.infer_page_size)
        plan.page_table = plan.workspace.get("fa3_page_table", b_att_req_idx.numel() * width, torch.int32)[
            : b_att_req_idx.numel() * width
        ].view(b_att_req_idx.numel(), width)
        page_table_copy(
            page_table=plan.page_table,
            req_to_token_indexs=plan.batch.req_to_token_indexs,
            b_req_idx=b_att_req_idx,
            page_size=self.infer_page_size,
        )

    def _forward_decode(self, layer: LayerAttentionBinding, plan, inputs: DenseInputs, *, out=None):
        alloc_func = layer.alloc_tensor
        q = inputs.query
        k = layer.params.key
        v = layer.params.value
        return self._decode_normal_decode_att(layer, plan, q=q, k=k, v=v, inputs=inputs, alloc_func=alloc_func)

    def _decode_normal_decode_att(
        self, layer, plan, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, inputs, alloc_func=torch.empty
    ):
        if layer.params.sliding_window is not None:
            window_size = layer.params.sliding_window
        else:
            window_size = (-1, -1)
        if layer.params.sink is not None:
            sink_weight: torch.Tensor = layer.params.sink
        else:
            sink_weight = None
        k_descale, v_descale = (None, None)
        Lq = q.shape[-1]
        sm_scale = 1.0 / Lq ** 0.5
        o = flash_attn_with_kvcache_autotune(
            q=q,
            k_cache=k.view(-1, self.infer_page_size, k.shape[1], k.shape[2]),
            v_cache=v.view(-1, self.infer_page_size, v.shape[1], v.shape[2]),
            page_table=plan.page_table,
            cache_seqlens=plan.b_att_seq_len,
            cu_seqlens_q=plan.cu_seqlens_q,
            cu_seqlens_k_new=None,
            max_seqlen_q=plan.decode_max_q_seq_len,
            max_seqlen_k=plan.decode_max_kv_seq_len,
            softmax_scale=sm_scale,
            causal=plan.causal,
            window_size=window_size,
            softcap=0.0,
            k_descale=k_descale,
            v_descale=v_descale,
            return_softmax_lse=False,
            sinks=sink_weight,
        )
        return o
