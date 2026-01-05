import dataclasses
import torch
from .base_att import BaseAttBackend, BasePrefillAttState, BaseDecodeAttState
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

    def copy_for_prefill_cuda_graph(self, new_state: "TritonPrefillAttState"):
        pass

    def prefill_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_weight,
        alloc_func=torch.empty,
        use_alibi=False,
    ) -> torch.Tensor:
        if use_alibi:
            return self._alibi_prefill_att(q=q, k=k, v=v, layer_weight=layer_weight, alloc_func=alloc_func)
        else:
            return self._nomarl_prefill_att(q=q, k=k, v=v, layer_weight=layer_weight, alloc_func=alloc_func)

    def _alibi_prefill_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_weight,
        alloc_func=torch.empty,
    ):
        out = alloc_func(q.shape, q.dtype)

        from ..triton_kernel.alibi_att.context_flashattention_nopad import context_attention_fwd

        context_attention_fwd(
            q,
            k,
            v,
            out,
            self.infer_state.b_req_idx,
            layer_weight.tp_alibi,
            self.infer_state.b_start_loc,
            self.infer_state.b_seq_len,
            self.infer_state.b_ready_cache_len,
            self.infer_state.max_len_in_batch,
            self.infer_state.req_manager.req_to_token_indexs,
        )
        return out

    def _nomarl_prefill_att(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_weight, alloc_func=torch.empty
    ):
        from ..triton_kernel.att.context_flashattention_nopad import context_attention_fwd

        out = alloc_func(q.shape, q.dtype)
        context_attention_fwd(
            q,
            k,
            v,
            out,
            self.infer_state.b_req_idx,
            self.infer_state.b_start_loc,
            self.infer_state.b_seq_len,
            self.infer_state.b_ready_cache_len,
            self.infer_state.max_len_in_batch,
            self.infer_state.req_manager.req_to_token_indexs,
        )
        return out


@dataclasses.dataclass
class TritonDecodeAttState(BaseDecodeAttState):
    def init_state(self):
        pass

    def copy_for_decode_cuda_graph(self, new_state: "TritonDecodeAttState"):
        pass

    def decode_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_weight,
        alloc_func=torch.empty,
        use_alibi=False,
    ):
        if use_alibi:
            return self._alibi_decode_att(q=q, k=k, v=v, layer_weight=layer_weight, alloc_func=alloc_func)
        else:
            q_head_num = q.shape[1]
            k_head_num = k.shape[1]
            if q_head_num == k_head_num:
                return self._normal_decode_flash_decoding_att(
                    q=q, k=k, v=v, layer_weight=layer_weight, alloc_func=alloc_func
                )
            elif q_head_num > k_head_num:
                return self._normal_decode_gqa_flash_decoding_att(
                    q=q, k=k, v=v, layer_weight=layer_weight, alloc_func=alloc_func
                )
            else:
                raise NotImplementedError("error")

    def _alibi_decode_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_weight,
        alloc_func=torch.empty,
    ):
        from ..triton_kernel.alibi_att.token_flashattention_nopad import token_attention_fwd

        out = alloc_func(q.shape, q.dtype)
        token_attention_fwd(
            q,
            k,
            v,
            out,
            layer_weight.tp_alibi,
            self.infer_state.req_manager.req_to_token_indexs,
            self.infer_state.b_req_idx,
            self.infer_state.b_start_loc,
            self.infer_state.b_seq_len,
            self.infer_state.max_len_in_batch,
            self.infer_state.total_token_num,
            alloc_tensor_func=alloc_func,
        )
        return out

    def _normal_decode_flash_decoding_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_weight,
        alloc_func=torch.empty,
    ):
        from ..triton_kernel.att.flash_decoding import token_decode_attention_flash_decoding

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
        layer_weight,
        alloc_func=torch.empty,
    ):
        from ..triton_kernel.att.gqa_flash_decoding import gqa_token_decode_attention_flash_decoding

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
        layer_weight,
        alloc_func=torch.empty,
    ):
        # TODO USE , 在特定场景下比 _normal_decode_gqa_flash_decoding_att 省显存
        from ..triton_kernel.att.gqa_flash_decoding_vsm import gqa_token_decode_attention_flash_decoding_vsm

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
        from ..triton_kernel.att.gqa_decode_flashattention_nopad import gqa_decode_attention_fwd

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
