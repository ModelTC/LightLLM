import dataclasses
import torch
from lightllm.utils.envs_utils import get_env_start_args
from ..base_att import BaseAttBackend, BasePrefillAttState, BaseDecodeAttState, AttControl
from typing import Optional, Tuple


class Int4kvTritonAttBackend(BaseAttBackend):
    def __init__(self, model):
        super().__init__(model)
        self.quant_group_size: int = get_env_start_args().llm_kv_quant_group_size

    def create_att_prefill_state(self, infer_state) -> "Int4kvTritonPrefillAttState":
        return Int4kvTritonPrefillAttState(backend=self, infer_state=infer_state)

    def create_att_decode_state(self, infer_state) -> "Int4kvTritonDecodeAttState":
        return Int4kvTritonDecodeAttState(backend=self, infer_state=infer_state)


@dataclasses.dataclass
class Int4kvTritonPrefillAttState(BasePrefillAttState):

    # 用于反量化的时候使用，可以减少反量化占用的显存数量。按需使用。
    b_kv_start_loc: torch.Tensor = None

    def init_state(self):
        self.b_kv_start_loc = (
            torch.cumsum(self.infer_state.b_seq_len, dim=0, dtype=self.infer_state.b_seq_len.dtype)
            - self.infer_state.b_seq_len
        )

    def prefill_att(
        self,
        q: torch.Tensor,
        k: Tuple[torch.Tensor, torch.Tensor],
        v: Tuple[torch.Tensor, torch.Tensor],
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        assert (
            att_control.use_alibi is False
            and att_control.use_sliding_window is False
            and att_control.use_att_sink is False
        )

        self.backend: Int4kvTritonAttBackend = self.backend  # for typing

        k, k_scale = k
        v, v_scale = v
        o = self._groupsize_quant_prefill_att(
            q=q,
            k=k,
            k_scale=k_scale,
            v=v,
            v_scale=v_scale,
            alloc_func=alloc_func,
        )
        return o

    def _groupsize_quant_prefill_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        k_scale: torch.Tensor,
        v: torch.Tensor,
        v_scale: torch.Tensor,
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        # o_tensor = alloc_func(q.shape, q.dtype, device=q.device)
        # batch_size = self.infer_state.b_seq_len.shape[0]

        assert k.untyped_storage().data_ptr() == v.untyped_storage().data_ptr()
        assert k_scale.untyped_storage().data_ptr() == v_scale.untyped_storage().data_ptr()

        total_token_num = self.infer_state.total_token_num
        head_dim = k.shape[2] * 2  # 2个4bit存储为一个int8, 所以维度需要翻倍，才是解量化后的精度
        k_dequant = alloc_func((total_token_num, k.shape[1], head_dim), dtype=q.dtype, device=q.device)
        v_dequant = alloc_func((total_token_num, v.shape[1], head_dim), dtype=q.dtype, device=q.device)
        o_tensor = alloc_func(q.shape, dtype=q.dtype, device=q.device)

        max_kv_seq_len = self.infer_state.max_kv_seq_len

        from ...triton_kernel.kv_copy.ppl_int4kv_copy_kv import dequantize_int4kv

        dequantize_int4kv(
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

        from ...triton_kernel.att.prefill_att.context_flashattention_nopad import context_attention_fwd_contiguous_kv

        context_attention_fwd_contiguous_kv(
            q=q,
            k=k_dequant,
            v=v_dequant,
            o=o_tensor,
            b_start_loc=self.infer_state.b_q_start_loc,
            b_kv_start_loc=self.b_kv_start_loc,
            b_seq_len=self.infer_state.b_seq_len,
            max_q_input_len=self.infer_state.max_q_seq_len,
            b_prompt_cache_len=self.infer_state.b_ready_cache_len,
        )
        return o_tensor


@dataclasses.dataclass
class Int4kvTritonDecodeAttState(BaseDecodeAttState):
    def init_state(self):
        pass

    def copy_for_decode_cuda_graph(self, new_state: "Int4kvTritonDecodeAttState"):
        super().copy_for_decode_cuda_graph(new_state)

    def decode_att(
        self,
        q: torch.Tensor,
        k: Tuple[torch.Tensor, torch.Tensor],
        v: Tuple[torch.Tensor, torch.Tensor],
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ):
        assert (
            att_control.use_alibi is False
            and att_control.use_sliding_window is False
            and att_control.use_att_sink is False
        )
        k, k_scale = k
        v, v_scale = v

        return self.ppl_int4kv_decode_att(
            q=q,
            k=k,
            k_scale=k_scale,
            v=v,
            v_scale=v_scale,
            alloc_func=alloc_func,
        )

    def ppl_int4kv_decode_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        k_scale: torch.Tensor,
        v: torch.Tensor,
        v_scale: torch.Tensor,
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        from ...triton_kernel.att.decode_att.int4kv.ppl_int4kv_flash_decoding import (
            token_decode_attention_flash_decoding,
        )

        return token_decode_attention_flash_decoding(
            q=q,
            infer_state=self.infer_state,
            cache_k=k,
            cache_k_scale=k_scale,
            cache_v=v,
            cache_v_scale=v_scale,
            alloc_tensor_func=alloc_func,
        )
