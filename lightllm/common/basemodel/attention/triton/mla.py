from ..mla import BoundMla, MlaPrefillInput, MlaDecodeInput
import dataclasses
import weakref
import torch
from ..base_att import BaseAttBackend, BaseAttState
from typing import Tuple


class MlaTritonAttBackend(BaseAttBackend):
    supports_mla = True

    def mla_support_error(self, spec, format, phase):
        reason = super().mla_support_error(spec, format, phase)
        if reason:
            return reason
        if phase == "decode" and spec.rope_dim != 64:
            return "Triton MLA decode requires RoPE dimension 64"
        return None

    def create_att_prefill_state(self, infer_state) -> "MlaTritonPrefillAttState":
        return MlaTritonPrefillAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))

    def create_att_decode_state(self, infer_state) -> "MlaTritonDecodeAttState":
        return MlaTritonDecodeAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))


@dataclasses.dataclass
class MlaTritonPrefillAttState(BaseAttState):
    graph_buffer_names = ()

    def init_state(self):
        pass

    def run(self, inputs: MlaPrefillInput, binding: BoundMla, alloc_func=torch.empty):
        if inputs.q.shape[2] == 256:
            return self._contiguous_kv_prefill_att(inputs, binding, alloc_func)
        return self._mla_prefill_att(inputs, binding, alloc_func)

    def _mla_prefill_att(self, inputs: MlaPrefillInput, binding: BoundMla, alloc_func=torch.empty):
        q, k, v = inputs.q, (inputs.k_nope, inputs.k_rope), inputs.v
        from ...triton_kernel.mla_att.prefill_att import context_attention_fwd_with_v

        qk_rope_head_dim = binding.rope_dim
        q_nope, q_rope = q[:, :, :-qk_rope_head_dim], q[:, :, -qk_rope_head_dim:]
        o_tensor = alloc_func((q_nope.shape[0], q_nope.shape[1], v.shape[-1]), dtype=q_nope.dtype, device=q.device)
        k_nope, k_rope = k
        softmax_scale = binding.softmax_scale
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

    def _contiguous_kv_prefill_att(self, inputs: MlaPrefillInput, binding: BoundMla, alloc_func=torch.empty):
        q, k, v = inputs.q, (inputs.k_nope, inputs.k_rope), inputs.v
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
class MlaTritonDecodeAttState(BaseAttState):
    graph_buffer_names = ()
    decode_att_block_seq: torch.Tensor = None
    mid_o_batch_start_index: torch.Tensor = None

    def init_state(self):
        self.prepare_for_capture()

    def prepare_for_capture(self):
        self.decode_att_block_seq = self.mid_o_batch_start_index = None

    def copy_for_decode_cuda_graph(self, new_state: "MlaTritonDecodeAttState"):
        super().copy_for_decode_cuda_graph(new_state)

    def run(self, inputs: MlaDecodeInput, binding: BoundMla, alloc_func=torch.empty):
        q, k = (inputs.q_nope, inputs.q_rope), binding.cache
        softmax_scale = binding.softmax_scale

        from ...triton_kernel.mla_att.decode_att import gqa_token_decode_attention_flash_decoding

        qk_rope_head_dim = binding.rope_dim
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
            metadata=self,
        )
        return out
