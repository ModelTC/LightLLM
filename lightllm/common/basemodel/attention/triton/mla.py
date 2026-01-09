import dataclasses
import torch
from ..base_att import BaseAttBackend, BasePrefillAttState, BaseDecodeAttState, AttControl
from typing import Tuple


class MlaTritonAttBackend(BaseAttBackend):
    def create_att_prefill_state(self, infer_state) -> "MlaTritonPrefillAttState":
        return MlaTritonPrefillAttState(backend=self, infer_state=infer_state)

    def create_att_decode_state(self, infer_state) -> "MlaTritonDecodeAttState":
        return MlaTritonDecodeAttState(backend=self, infer_state=infer_state)


@dataclasses.dataclass
class MlaTritonPrefillAttState(BasePrefillAttState):
    def init_state(self):
        pass

    def prefill_att(
        self,
        q: torch.Tensor,
        k: Tuple[torch.Tensor, torch.Tensor],
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        assert (
            att_control.use_alibi is False
            and att_control.use_sliding_window is False
            and att_control.use_att_sink is False
        )
        return self._mla_prefill_att(q=q, k=k, v=v, att_control=att_control, alloc_func=alloc_func)

    def _mla_prefill_att(
        self,
        q: torch.Tensor,
        k: Tuple[torch.Tensor, torch.Tensor],
        v: torch.Tensor,
        att_control: AttControl,
        alloc_func=torch.empty,
    ):
        from ...triton_kernel.mla_att.prefill_att import context_attention_fwd_with_v

        qk_rope_head_dim = 64
        q_nope, q_rope = q[:, :, :-qk_rope_head_dim], q[:, :, -qk_rope_head_dim:]
        o_tensor = alloc_func(q_nope.shape, dtype=q_nope.dtype, device=q.device)
        k_nope, k_rope = k
        assert att_control.mla_prefill
        softmax_scale = att_control.mla_prefill_dict["softmax_scale"]
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


@dataclasses.dataclass
class MlaTritonDecodeAttState(BaseDecodeAttState):
    def init_state(self):
        pass

    def copy_for_decode_cuda_graph(self, new_state: "MlaTritonDecodeAttState"):
        super().copy_for_decode_cuda_graph(new_state)

    def decode_att(
        self,
        q: Tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ):
        assert (
            att_control.use_sliding_window is False
            and att_control.use_att_sink is False
            and att_control.use_alibi is False
        )
        assert v is None

        return self._mla_decode_att(
            q=q,
            k=k,
            v=v,
            att_control=att_control,
            alloc_func=alloc_func,
        )

    def _mla_decode_att(
        self,
        q: Tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl,
        alloc_func=torch.empty,
    ):
        assert att_control.mla_decode
        softmax_scale = att_control.mla_decode_dict["softmax_scale"]

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
