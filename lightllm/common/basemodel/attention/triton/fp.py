import dataclasses
import torch
from ..base_att import BaseAttBackend, BasePrefillAttState, BaseDecodeAttState, AttControl
from typing import Optional


class TritonAttBackend(BaseAttBackend):
    def create_att_prefill_state(self, infer_state) -> "TritonPrefillAttState":
        return TritonPrefillAttState(backend=self, infer_state=infer_state)

    def create_att_decode_state(self, infer_state) -> "TritonDecodeAttState":
        return TritonDecodeAttState(backend=self, infer_state=infer_state)


@dataclasses.dataclass
class TritonPrefillAttState(BasePrefillAttState):
    def init_state(self):
        pass

    def prefill_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        assert att_control.use_sliding_window is False and att_control.use_att_sink is False
        if att_control.use_alibi:
            assert att_control.tp_alibi is not None
            return self._alibi_prefill_att(q=q, k=k, v=v, att_control=att_control, alloc_func=alloc_func)
        else:
            return self._nomarl_prefill_att(q=q, k=k, v=v, alloc_func=alloc_func)

    def _alibi_prefill_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl,
        alloc_func=torch.empty,
    ):
        out = alloc_func(q.shape, q.dtype)

        from ...triton_kernel.alibi_att.context_flashattention_nopad import context_attention_fwd

        context_attention_fwd(
            q,
            k,
            v,
            out,
            self.infer_state.b_req_idx,
            att_control.tp_alibi,
            self.infer_state.b_q_start_loc,
            self.infer_state.b_seq_len,
            self.infer_state.b_ready_cache_len,
            self.infer_state.max_q_seq_len,
            self.infer_state.req_manager.req_to_token_indexs,
        )
        return out

    def _nomarl_prefill_att(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, alloc_func=torch.empty):
        from ...triton_kernel.att.prefill_att.context_flashattention_nopad import context_attention_fwd

        out = alloc_func(q.shape, q.dtype)
        context_attention_fwd(
            q,
            k,
            v,
            out,
            self.infer_state.b_req_idx,
            self.infer_state.b_q_start_loc,
            self.infer_state.b_seq_len,
            self.infer_state.b_ready_cache_len,
            self.infer_state.max_q_seq_len,
            self.infer_state.req_manager.req_to_token_indexs,
        )
        return out


@dataclasses.dataclass
class TritonDecodeAttState(BaseDecodeAttState):
    def init_state(self):
        pass

    def copy_for_decode_cuda_graph(self, new_state: "TritonDecodeAttState"):
        super().copy_for_decode_cuda_graph(new_state)

    def decode_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ):
        assert att_control.use_sliding_window is False and att_control.use_att_sink is False
        if att_control.use_alibi:
            assert att_control.tp_alibi is not None
            return self._alibi_decode_att(q=q, k=k, v=v, att_control=att_control, alloc_func=alloc_func)
        else:
            q_head_num = q.shape[1]
            k_head_num = k.shape[1]
            if q_head_num == k_head_num:
                return self._normal_decode_flash_decoding_att(q=q, k=k, v=v, alloc_func=alloc_func)
            elif q_head_num > k_head_num:
                return self._normal_decode_gqa_flash_decoding_att(q=q, k=k, v=v, alloc_func=alloc_func)
            else:
                raise NotImplementedError("error")

    def _alibi_decode_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl,
        alloc_func=torch.empty,
    ):
        from ...triton_kernel.alibi_att.token_flashattention_nopad import token_attention_fwd

        out = alloc_func(q.shape, q.dtype)
        token_attention_fwd(
            q,
            k,
            v,
            out,
            att_control.tp_alibi,
            self.infer_state.req_manager.req_to_token_indexs,
            self.infer_state.b_req_idx,
            self.infer_state.b_kv_start_loc,
            self.infer_state.b_seq_len,
            self.infer_state.max_kv_seq_len,
            self.infer_state.total_token_num,
            alloc_tensor_func=alloc_func,
        )
        return out

    def _normal_decode_flash_decoding_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alloc_func=torch.empty,
    ):
        from ...triton_kernel.att.decode_att.mha.flash_decoding.flash_decoding import (
            token_decode_attention_flash_decoding,
        )

        out = alloc_func(q.shape, q.dtype)

        token_decode_attention_flash_decoding(
            q=q,
            infer_state=self.infer_state,
            cache_k=k,
            cache_v=v,
            out=out,
            alloc_tensor_func=alloc_func,
        )
        return out

    def _normal_decode_gqa_flash_decoding_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alloc_func=torch.empty,
    ):
        from ...triton_kernel.att.decode_att.gqa.flash_decoding.gqa_flash_decoding import (
            gqa_token_decode_attention_flash_decoding,
        )

        out = alloc_func(q.shape, q.dtype)

        gqa_token_decode_attention_flash_decoding(
            q=q,
            infer_state=self.infer_state,
            cache_k=k,
            cache_v=v,
            out=out,
            alloc_tensor_func=alloc_func,
        )

        return out

    def _normal_decode_gqa_flash_decoding_att_vsm(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alloc_func=torch.empty,
    ):
        # TODO USE , 在特定场景下比 _normal_decode_gqa_flash_decoding_att 省显存
        from ...triton_kernel.att.decode_att.gqa.flash_decoding.gqa_flash_decoding_vsm import (
            gqa_token_decode_attention_flash_decoding_vsm,
        )

        out = alloc_func(q.shape, q.dtype)

        gqa_token_decode_attention_flash_decoding_vsm(
            q=q,
            k=k,
            v=v,
            infer_state=self.infer_state,
            out=out,
            alloc_tensor_func=alloc_func,
        )
        return out

    def _normal_decode_gqa_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_weight,
        alloc_func=torch.empty,
    ):
        # TODO USE , 在特定场景下比 _normal_decode_gqa_flash_decoding_att 省显存
        from ...triton_kernel.att.decode_att.gqa.gqa_decode_flashattention_nopad import gqa_decode_attention_fwd

        out = alloc_func(q.shape, q.dtype)

        gqa_decode_attention_fwd(
            q=q,
            k=k,
            v=v,
            out=out,
            req_to_tokens=self.infer_state.req_manager.req_to_token_indexs,
            b_req_idx=self.infer_state.b_req_idx,
            b_seq_len=self.infer_state.b_seq_len,
        )
        return out

    def _normal_decode_stage3_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alloc_func=torch.empty,
    ):
        total_token_num = self.infer_state.total_token_num
        batch_size = self.infer_state.batch_size
        q_head_num = q.shape[1]
        head_dim = q.shape[2]

        calcu_shape1 = (batch_size, q_head_num, head_dim)
        att_m_tensor = alloc_func((q_head_num, total_token_num), torch.float32)

        from ...triton_kernel.att.decode_att.mha.stage3_decode_att.token_attention_nopad_att1 import token_att_fwd

        token_att_fwd(
            q.view(calcu_shape1),
            k,
            att_m_tensor,
            Req_to_tokens=self.infer_state.req_manager.req_to_token_indexs,
            B_req_idx=self.infer_state.b_req_idx,
            B_Start_Loc=self.infer_state.b_kv_start_loc,
            B_Seqlen=self.infer_state.b_seq_len,
            max_len_in_batch=self.infer_state.max_kv_seq_len,
        )

        o_tensor = alloc_func(q.shape, q.dtype)
        from ...triton_kernel.att.decode_att.mha.stage3_decode_att.token_attention_softmax_and_reducev import (
            token_softmax_reducev_fwd,
        )

        token_softmax_reducev_fwd(
            att_m_tensor,
            v,
            o_tensor.view(calcu_shape1),
            req_to_tokens=self.infer_state.req_manager.req_to_token_indexs,
            b_req_idx=self.infer_state.b_req_idx,
            b_start_loc=self.infer_state.b_kv_start_loc,
            b_seq_len=self.infer_state.b_seq_len,
        )
        return o_tensor
