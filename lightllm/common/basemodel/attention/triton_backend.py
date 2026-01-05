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
        k: torch.tensor,
        v: torch.tensor,
        layer_weight,
        out: Optional[torch.Tensor] = None,
        alloc_func=torch.empty,
        use_alibi=False,
    ) -> torch.Tensor:
        if use_alibi:
            return self._alibi_prefill_att(q=q, k=k, v=v, layer_weight=layer_weight, out=out, alloc_func=alloc_func)
        else:
            raise NotImplementedError("error")

    def _alibi_prefill_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_weight,
        out: Optional[torch.Tensor] = None,
        alloc_func=torch.empty,
    ):
        from lightllm.common.basemodel.infer_struct import InferStateInfo

        infer_state: InferStateInfo = self.infer_state
        out = alloc_func(q.shape, q.dtype) if out is None else out

        from ..triton_kernel.alibi_att.context_flashattention_nopad import context_attention_fwd

        context_attention_fwd(
            q,
            k,
            v,
            out,
            infer_state.b_req_idx,
            layer_weight.tp_alibi,
            infer_state.b_start_loc,
            infer_state.b_seq_len,
            infer_state.b_ready_cache_len,
            infer_state.max_len_in_batch,
            infer_state.req_manager.req_to_token_indexs,
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
        out: Optional[torch.Tensor] = None,
        alloc_func=torch.empty,
        use_alibi=False,
    ):
        if use_alibi:
            return self._alibi_decode_att(q=q, k=k, v=v, layer_weight=layer_weight, out=out, alloc_func=alloc_func)
        else:
            raise NotImplementedError("error")

    def _alibi_decode_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_weight,
        out: Optional[torch.Tensor] = None,
        alloc_func=torch.empty,
    ):
        from lightllm.common.basemodel.infer_struct import InferStateInfo

        infer_state: InferStateInfo = self.infer_state

        from ..triton_kernel.alibi_att.token_flashattention_nopad import token_attention_fwd

        out = alloc_func(q.shape, q.dtype) if out is None else out
        token_attention_fwd(
            q,
            k,
            v,
            out,
            layer_weight.tp_alibi,
            infer_state.req_manager.req_to_token_indexs,
            infer_state.b_req_idx,
            infer_state.b_start_loc,
            infer_state.b_seq_len,
            infer_state.max_len_in_batch,
            infer_state.total_token_num,
            alloc_tensor_func=alloc_func,
        )
        return out
