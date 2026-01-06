import dataclasses
import torch
from .base_att import BaseAttBackend, BasePrefillAttState, BaseDecodeAttState, AttControl
from typing import Optional, Tuple


class Int8kvTritonAttBackend(BaseAttBackend):
    def __init__(self, quant_group_size: int):
        self.quant_group_size: int = quant_group_size

    def create_att_prefill_state(self, infer_state) -> "Int8kvTritonPrefillAttState":
        return Int8kvTritonPrefillAttState(backend=self, infer_state=infer_state)

    def create_att_decode_state(self, infer_state) -> "Int8kvTritonDecodeAttState":
        return Int8kvTritonDecodeAttState(backend=self, infer_state=infer_state)


@dataclasses.dataclass
class Int8kvTritonPrefillAttState(BasePrefillAttState):

    # 用于反量化的时候使用，可以减少反量化占用的显存数量。按需使用。
    b_kv_start_loc: torch.Tensor = None

    def init_state(self):
        self.b_kv_start_loc = (
            torch.cumsum(self.infer_state.b_seq_len, dim=0, dtype=self.infer_state.b_seq_len.dtype)
            - self.infer_state.b_seq_len
        )

    def copy_for_prefill_cuda_graph(self, new_state: "Int8kvTritonPrefillAttState"):
        pass

    def prefill_att(
        self,
        q: torch.Tensor,
        k: Tuple[torch.Tensor, torch.Tensor],
        v: Tuple[torch.Tensor, torch.Tensor],
        layer_weight,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        assert att_control.use_alibi is False

        self.backend: Int8kvTritonAttBackend = self.backend  # for typing
        if self.backend.quant_group_size == 8:
            pass
        # context_attention_fwd_ppl_int8kv(
        #     q.view(-1, self.tp_q_head_num_, self.head_dim_),
        #     kv_dequant[:, 0 : self.tp_k_head_num_, :, :],
        #     kv_dequant[:, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :, :],
        #     o_tensor.view(-1, self.tp_q_head_num_, self.head_dim_),
        #     infer_state.b_start_loc,
        #     infer_state.b_seq_len,
        #     infer_state.max_len_in_batch,
        #     infer_state.b_ready_cache_len,
        # )

        return self._nomarl_prefill_att(q=q, k=k, v=v, layer_weight=layer_weight, alloc_func=alloc_func)

    def _groupsize8_quant_prefill_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        k_scale: torch.Tensor,
        v: torch.Tensor,
        v_scale: torch.Tensor,
        layer_weight,
        alloc_func=torch.empty,
    ):
        # o_tensor = alloc_func(q.shape, q.dtype, device=q.device)
        # batch_size = self.infer_state.b_seq_len.shape[0]

        assert k.untyped_storage().data_ptr() == v.untyped_storage().data_ptr()
        assert k_scale.untyped_storage().data_ptr() == v_scale.untyped_storage().data_ptr()

        total_token_num = self.infer_state.total_token_num
        k_dequant = alloc_func((total_token_num, k.shape[1], k.shape[2]), dtype=q.dtype, device=q.device)
        v_dequant = alloc_func((total_token_num, v.shape[1], v.shape[2]), dtype=q.dtype, device=q.device)

        max_kv_seq_len = self.infer_state.max_kv_seq_len

        from ..triton_kernel.kv_copy.ppl_quant_copy_kv import dequantize_int8kv

        dequantize_int8kv(
            k=k,
            k_scale=k_scale,
            v=v,
            v_scale=v_scale,
            req_to_token_indexs=self.infer_state.req_manager.req_to_token_indexs,
            b_seq_len=self.infer_state.b_seq_len,
            b_req_idx=self.infer_state.b_req_idx,
            b_kv_start_loc=self.b_kv_start_loc,
            k_out=k_dequant,
            v_out=v_dequant,
            max_len_in_batch=max_kv_seq_len,
            quant_group_size=self.backend.quant_group_size,
        )

        # context_attention_fwd_ppl_int8kv(
        #     q.view(-1, self.tp_q_head_num_, self.head_dim_),
        #     kv_dequant[:, 0 : self.tp_k_head_num_, :, :],
        #     kv_dequant[:, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :, :],
        #     o_tensor.view(-1, self.tp_q_head_num_, self.head_dim_),
        #     infer_state.b_start_loc,
        #     infer_state.b_seq_len,
        #     infer_state.max_len_in_batch,
        #     infer_state.b_ready_cache_len,
        # )


@dataclasses.dataclass
class Int8kvTritonDecodeAttState(BaseDecodeAttState):
    def init_state(self):
        pass

    def copy_for_decode_cuda_graph(self, new_state: "Int8kvTritonDecodeAttState"):
        pass

    def decode_att(
        self,
        q: torch.Tensor,
        k: Tuple[torch.Tensor, torch.Tensor],
        v: Tuple[torch.Tensor, torch.Tensor],
        layer_weight,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ):
        assert att_control.use_alibi is False
        q = q
        k, k_scale = k
        v, v_scale = v
        if k_scale.ndim == 3 and k_scale.shape[2] == 1:
            return self._per_head_quant_decode_stage3_att(
                q=q,
                k=k,
                k_scale=k_scale,
                v=v,
                v_scale=v,
                layer_weight=layer_weight,
                alloc_func=alloc_func,
            )
        else:
            raise NotImplementedError("not support decode att")

    def _per_head_quant_decode_stage3_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        k_scale: torch.Tensor,
        v: torch.Tensor,
        v_scale: torch.Tensor,
        layer_weight,
        alloc_func=torch.empty,
    ):
        raise NotImplementedError("error")
