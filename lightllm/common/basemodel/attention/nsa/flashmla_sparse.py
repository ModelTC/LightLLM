# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/nsa_backend.py
# Uses sgl_kernel.flash_mla and sgl_kernel.flash_attn from the sglang kernel library.

import dataclasses
import torch
from typing import Tuple, TYPE_CHECKING

from ..base_att import BaseAttBackend, BasePrefillAttState, BaseDecodeAttState, AttControl
from lightllm.utils.dist_utils import get_current_device_id

if TYPE_CHECKING:
    from lightllm.common.basemodel.infer_struct import InferStateInfo


class NsaFlashMlaSparseAttBackend(BaseAttBackend):
    def __init__(self, model):
        super().__init__(model=model)

    def create_att_prefill_state(self, infer_state: "InferStateInfo") -> "NsaFlashMlaSparsePrefillAttState":
        return NsaFlashMlaSparsePrefillAttState(backend=self, infer_state=infer_state)

    def create_att_decode_state(self, infer_state: "InferStateInfo") -> "NsaFlashMlaSparseDecodeAttState":
        return NsaFlashMlaSparseDecodeAttState(backend=self, infer_state=infer_state)


@dataclasses.dataclass
class NsaFlashMlaSparsePrefillAttState(BasePrefillAttState):
    """Prefill attention state for NSA using flash_mla_sparse_fwd."""

    cu_seqlens_q: torch.Tensor = None
    cu_seqlens_k: torch.Tensor = None

    def init_state(self):
        self.cu_seqlens_q = self.infer_state.b1_cu_q_seq_len.int()
        self.cu_seqlens_k = self.infer_state.b1_cu_kv_seq_len.int()

    def prefill_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        assert att_control.nsa_prefill, "nsa_prefill must be True for NSA prefill attention"
        assert att_control.nsa_prefill_dict is not None, "nsa_prefill_dict is required"

        return self._nsa_prefill_att(q=q, kv=k, att_control=att_control)

    def _nsa_prefill_att(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        att_control: AttControl,
    ) -> torch.Tensor:
        from sgl_kernel.flash_mla import flash_mla_sparse_fwd

        nsa_dict = att_control.nsa_prefill_dict
        topk_indices = nsa_dict["topk_indices"]
        softmax_scale = nsa_dict["softmax_scale"]
        kv_lora_rank = nsa_dict["kv_lora_rank"]

        if topk_indices.ndim == 2:
            topk_indices = topk_indices.unsqueeze(1)

        mla_out, _, _ = flash_mla_sparse_fwd(
            q=q,
            kv=kv,
            indices=topk_indices,
            sm_scale=softmax_scale,
            d_v=kv_lora_rank,
        )
        return mla_out


@dataclasses.dataclass
class NsaFlashMlaSparseDecodeAttState(BaseDecodeAttState):

    cu_seqlens_q: torch.Tensor = None
    cu_seqlens_k: torch.Tensor = None

    def init_state(self):
        self.cu_seqlens_q = self.infer_state.b1_cu_q_seq_len.int()
        self.cu_seqlens_k = self.infer_state.b1_cu_kv_seq_len.int()

    def decode_att(
        self,
        q: Tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        assert att_control.nsa_decode, "nsa_decode must be True for NSA decode attention"
        assert att_control.nsa_decode_dict is not None, "nsa_decode_dict is required"

        return self._nsa_decode_att(q=q, kv=k, att_control=att_control)

    def _nsa_decode_att(
        self,
        q: Tuple[torch.Tensor, torch.Tensor],
        kv: torch.Tensor,
        att_control: AttControl,
    ) -> torch.Tensor:
        from sgl_kernel.flash_attn import flash_attn_with_kvcache

        nsa_dict = att_control.nsa_decode_dict
        topk_indices = nsa_dict["topk_indices"]
        nsa_cache_seqlens = nsa_dict["nsa_cache_seqlens"]
        nsa_cu_seqlens_k = nsa_dict["nsa_cu_seqlens_k"]
        softmax_scale = nsa_dict["softmax_scale"]
        kv_lora_rank = nsa_dict["kv_lora_rank"]
        qk_rope_head_dim = nsa_dict["qk_rope_head_dim"]

        q_nope, q_rope = q

        # Extract k_rope and kv_nope from the KV buffer
        k_rope = kv[:, :, -qk_rope_head_dim:].view(-1, 1, 1, qk_rope_head_dim)
        kv_nope = kv[:, :, :-qk_rope_head_dim].view(-1, 1, 1, kv_lora_rank)

        o_tensor = flash_attn_with_kvcache(
            q=q_rope,
            k_cache=k_rope,
            v_cache=kv_nope,
            qv=q_nope,
            page_table=topk_indices,
            cache_seqlens=nsa_cache_seqlens,
            cu_seqlens_q=self.cu_seqlens_q,
            cu_seqlens_k_new=nsa_cu_seqlens_k,
            max_seqlen_q=self.infer_state.max_q_seq_len,
            softmax_scale=softmax_scale,
            causal=True,
        )
        return o_tensor
