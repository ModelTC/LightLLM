import dataclasses
from typing import Tuple

import torch
import triton

from lightllm.common.attention.api.backend import LayerAttention
from lightllm.common.attention.api.inputs import MLADecodeInputs, MLAPrefillInputs
from lightllm.common.attention.api.spec import AttentionFamily
from lightllm.common.attention.backends.base import PreparedPlan
from lightllm.common.attention.backends.dense.fa3.backend import Fa3AttBackend
from lightllm.common.basemodel.triton_kernel.fa3_utils import build_dynamic_spec_fa3_decode_params, page_table_copy
from lightllm.common.basemodel.triton_kernel.gen_prefill_params import gen_cumsum_pad0_tensor
from lightllm.common.basemodel.triton_kernel.mtp_utils import build_mtp_shared_group_markers
from lightllm.utils.sgl_utils import flash_attn_varlen_func, flash_attn_with_kvcache


@dataclasses.dataclass
class MlaFa3PrefillPlan(PreparedPlan):
    cu_seqlens_q: torch.Tensor = None
    cu_seqlens_k: torch.Tensor = None
    causal: bool = None


@dataclasses.dataclass
class MlaFa3DecodePlan(PreparedPlan):
    cu_seqlens_q: torch.Tensor = None
    cu_seqlens_k: torch.Tensor = None
    page_table: torch.Tensor = None
    b_att_seq_len: torch.Tensor = None
    decode_max_q_seq_len: int = None
    causal: bool = None


class MlaFa3AttBackend(Fa3AttBackend):
    provider = "fa3"
    max_head_size = None
    cache_formats = frozenset({"None"})
    family = AttentionFamily.MLA

    def _allocate_prefill_plan(self, capacity, capture, workspace):
        return MlaFa3PrefillPlan(capacity, capture, workspace)

    def _allocate_decode_plan(self, capacity, capture, workspace):
        return MlaFa3DecodePlan(capacity, capture, workspace)

    def _update_prefill(self, plan):
        plan.causal = self.uses_causal_attention()
        plan.cu_seqlens_q = plan.batch.b1_cu_q_seq_len.int()
        plan.cu_seqlens_k = plan.batch.b1_cu_kv_seq_len.int()

    def _forward_prefill(self, layer: LayerAttention, plan, inputs: MLAPrefillInputs, *, out=None) -> torch.Tensor:
        alloc_func = layer.alloc_tensor
        q = inputs.query
        k = (inputs.key_nope, inputs.key_rope)
        v = inputs.value
        return self._prefill_mla_prefill_att(layer, plan, q=q, k=k, v=v, inputs=inputs, alloc_func=alloc_func)

    def _prefill_mla_prefill_att(
        self, layer, plan, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, inputs, alloc_func=torch.empty
    ) -> torch.Tensor:
        k_nope, k_rope = k
        q_head_num = q.shape[1]
        k = torch.cat([k_nope, torch.repeat_interleave(k_rope, q_head_num, dim=-2)], dim=-1)
        assert q.ndim == 3 and k.ndim == 3 and (v.ndim == 3)
        softmax_scale = layer.params.softmax_scale
        o_tensor = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=plan.cu_seqlens_q,
            cu_seqlens_k=plan.cu_seqlens_k,
            max_seqlen_q=plan.batch.max_q_seq_len,
            max_seqlen_k=plan.batch.max_kv_seq_len,
            softmax_scale=softmax_scale,
            causal=plan.causal,
            return_softmax_lse=False,
        )
        return o_tensor

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

    def _forward_decode(self, layer: LayerAttention, plan, inputs: MLADecodeInputs, *, out=None):
        alloc_func = layer.alloc_tensor
        q = (inputs.query_nope, inputs.query_rope)
        k = layer.params.kv
        v = None
        assert v is None
        return self._decode_mla_decode_att(layer, plan, q=q, k=k, v=v, inputs=inputs, alloc_func=alloc_func)

    def _decode_mla_decode_att(
        self,
        layer,
        plan,
        q: Tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        v: torch.Tensor,
        inputs=None,
        alloc_func=torch.empty,
    ):
        q_nope, q_rope = q
        kv = k
        qk_rope_head_dim = layer.params.qk_rope_head_dim
        kv_lora_rank = kv.shape[-1] - qk_rope_head_dim
        k_rope = kv[:, :, -qk_rope_head_dim:].view(-1, self.infer_page_size, 1, qk_rope_head_dim)
        kv_nope = kv[:, :, :-qk_rope_head_dim].view(-1, self.infer_page_size, 1, kv_lora_rank)
        k_descale, v_descale = (None, None)
        softmax_scale = layer.params.softmax_scale
        o_tensor = flash_attn_with_kvcache(
            q=q_rope,
            k_cache=k_rope,
            v_cache=kv_nope,
            qv=q_nope,
            page_table=plan.page_table,
            cache_seqlens=plan.b_att_seq_len,
            cu_seqlens_q=plan.cu_seqlens_q,
            cu_seqlens_k_new=plan.cu_seqlens_k,
            max_seqlen_q=plan.decode_max_q_seq_len,
            softmax_scale=softmax_scale,
            causal=plan.causal,
            window_size=(-1, -1),
            softcap=0.0,
            k_descale=k_descale,
            v_descale=v_descale,
            return_softmax_lse=False,
        )
        return o_tensor
