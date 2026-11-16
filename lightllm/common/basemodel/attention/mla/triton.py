import dataclasses
import weakref
import torch
from ..base_att import BaseAttBackend, BasePrefillAttState, BaseDecodeAttState
from ..params import MlaAttParams
from typing import Tuple


class MlaTritonAttBackend(BaseAttBackend):
    def create_att_prefill_state(self, infer_state) -> "MlaTritonPrefillAttState":
        return MlaTritonPrefillAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))

    def create_att_decode_state(self, infer_state) -> "MlaTritonDecodeAttState":
        return MlaTritonDecodeAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))


@dataclasses.dataclass
class MlaTritonPrefillAttState(BasePrefillAttState):
    params_type = MlaAttParams

    def init_state(self):
        pass

    def prefill_att(
        self,
        q: torch.Tensor,
        k: Tuple[torch.Tensor, torch.Tensor],
        v: torch.Tensor,
        params: MlaAttParams = None,
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        params = self._check_params(params)
        if q.shape[2] == 256:
            # glm model
            return self._contiguous_kv_prefill_att(q=q, k=k, v=v, params=params, alloc_func=alloc_func)
        else:
            return self._mla_prefill_att(q=q, k=k, v=v, params=params, alloc_func=alloc_func)

    def _mla_prefill_att(
        self,
        q: torch.Tensor,
        k: Tuple[torch.Tensor, torch.Tensor],
        v: torch.Tensor,
        params: MlaAttParams,
        alloc_func=torch.empty,
    ):
        from ...triton_kernel.mla_att.prefill_att import context_attention_fwd_with_v

        qk_rope_head_dim = 64
        q_nope, q_rope = q[:, :, :-qk_rope_head_dim], q[:, :, -qk_rope_head_dim:]
        o_tensor = alloc_func((q_nope.shape[0], q_nope.shape[1], v.shape[-1]), dtype=q_nope.dtype, device=q.device)
        k_nope, k_rope = k
        softmax_scale = params.softmax_scale
        context_attention_fwd_with_v(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            v,
            o_tensor,
            self.infer_state.b_q_start_loc,
            self.infer_state.b1_cu_kv_seq_len,
            self.infer_state.b_seq_len,
            self.infer_state.b_ready_cache_len,
            self.infer_state.max_q_seq_len,
            softmax_scale,
        )
        return o_tensor

    def _contiguous_kv_prefill_att(
        self,
        q: torch.Tensor,
        k: Tuple[torch.Tensor, torch.Tensor],
        v: torch.Tensor,
        params: MlaAttParams,
        alloc_func=torch.empty,
    ):
        from ...triton_kernel.att.prefill_att.context_flashattention_nopad import context_attention_fwd_contiguous_kv

        k_nope, k_rope = k
        q_head_num = q.shape[1]

        k_merged = torch.cat([k_nope.expand(-1, q_head_num, -1), k_rope.expand(-1, q_head_num, -1)], dim=-1)

        o_tensor = alloc_func(q.shape, dtype=q.dtype, device=q.device)

        context_attention_fwd_contiguous_kv(
            q,
            k_merged,
            v,
            o_tensor,
            self.infer_state.b_q_start_loc,
            self.infer_state.b1_cu_kv_seq_len,
            self.infer_state.b_seq_len,
            self.infer_state.max_q_seq_len,
            self.infer_state.b_ready_cache_len,
        )
        return o_tensor


@dataclasses.dataclass
class MlaTritonDecodeAttState(BaseDecodeAttState):
    params_type = MlaAttParams

    def init_state(self):
        pass

    def decode_att(
        self,
        q: Tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        v: torch.Tensor,
        params: MlaAttParams = None,
        alloc_func=torch.empty,
    ):
        params = self._check_params(params)
        assert v is None

        return self._mla_decode_att(
            q=q,
            k=k,
            v=v,
            params=params,
            alloc_func=alloc_func,
        )

    def _mla_decode_att(
        self,
        q: Tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        v: torch.Tensor,
        params: MlaAttParams,
        alloc_func=torch.empty,
    ):
        softmax_scale = params.softmax_scale

        from ...triton_kernel.mla_att.decode_att import gqa_token_decode_attention_flash_decoding

        qk_rope_head_dim = 64
        q_nope, q_rope = q
        kv = k

        out = gqa_token_decode_attention_flash_decoding(
            q_nope=q_nope,
            q_rope=q_rope,
            kv_nope=kv[:, :, :-qk_rope_head_dim],
            kv_rope=kv[:, :, -qk_rope_head_dim:],
            infer_state=self.infer_state,
            softmax_scale=softmax_scale,
            alloc_tensor_func=alloc_func,
        )
        return out
