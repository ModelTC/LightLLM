import dataclasses
import torch
import triton
from typing import Tuple
from ..base_att import BaseAttBackend, BasePrefillAttState, BaseDecodeAttState, AttControl
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.utils.sgl_utils import flash_attn_with_kvcache, flash_attn_varlen_func
from lightllm.utils.envs_utils import get_env_start_args, get_page_size
from lightllm.common.basemodel.triton_kernel.fa3_utils import page_table_copy
from lightllm.common.basemodel.triton_kernel.gen_prefill_params import gen_cumsum_pad0_tensor


class PagedMlaFa3AttBackend(BaseAttBackend):
    def __init__(self, model, page_size=None):
        super().__init__(model=model)
        self.page_size = page_size or get_page_size()
        self.get_page_table_buffer()

    def get_page_table_buffer(self):
        model = self.model
        if not hasattr(self, "_shared_page_table_buffer"):
            shared_len = model.graph_max_batch_size * triton.cdiv(model.graph_max_len_in_batch, self.page_size)
            self._shared_page_table_buffer = [
                torch.empty(shared_len, dtype=torch.int32).to(get_current_device_id()),
                torch.empty(shared_len, dtype=torch.int32).to(get_current_device_id()),
            ]
        return self._shared_page_table_buffer

    def create_att_prefill_state(self, infer_state):
        return PagedMlaFa3PrefillAttState(backend=self, infer_state=infer_state)

    def create_att_decode_state(self, infer_state):
        return PagedMlaFa3DecodeAttState(backend=self, infer_state=infer_state)


@dataclasses.dataclass
class PagedMlaFa3PrefillAttState(BasePrefillAttState):
    cu_seqlens_q: torch.Tensor = None
    cu_seqlens_k: torch.Tensor = None

    def init_state(self):
        self.cu_seqlens_q = self.infer_state.b1_cu_q_seq_len.int()
        self.cu_seqlens_k = self.infer_state.b1_cu_kv_seq_len.int()

    def prefill_att(
        self, q, k: Tuple[torch.Tensor, torch.Tensor], v, att_control: AttControl = AttControl(), alloc_func=torch.empty
    ):
        assert (
            att_control.use_alibi is False
            and att_control.use_sliding_window is False
            and att_control.use_att_sink is False
        )
        return self._mla_prefill_att(q=q, k=k, v=v, att_control=att_control, alloc_func=alloc_func)

    def _mla_prefill_att(
        self, q, k: Tuple[torch.Tensor, torch.Tensor], v, att_control: AttControl, alloc_func=torch.empty
    ):
        k_nope, k_rope = k
        q_head_num = q.shape[1]
        k = torch.cat([k_nope, torch.repeat_interleave(k_rope, q_head_num, dim=-2)], dim=-1)
        assert q.ndim == 3 and k.ndim == 3 and v.ndim == 3
        assert att_control.mla_prefill
        return flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=self.cu_seqlens_q,
            cu_seqlens_k=self.cu_seqlens_k,
            max_seqlen_q=self.infer_state.max_q_seq_len,
            max_seqlen_k=self.infer_state.max_kv_seq_len,
            softmax_scale=att_control.mla_prefill_dict["softmax_scale"],
            causal=True,
            return_softmax_lse=False,
        )


@dataclasses.dataclass
class PagedMlaFa3DecodeAttState(BaseDecodeAttState):
    cu_seqlens_q: torch.Tensor = None
    cu_seqlens_k: torch.Tensor = None
    page_table: torch.Tensor = None
    b_att_seq_len: torch.Tensor = None
    decode_max_q_seq_len: int = None

    def init_state(self):
        args_mtp_step = get_env_start_args().mtp_step
        if args_mtp_step > 0:
            mtp_size = args_mtp_step + 1
            b_q_seq_len = torch.full(
                (self.infer_state.b_seq_len.shape[0] // mtp_size,),
                fill_value=mtp_size,
                dtype=torch.int32,
                device=self.infer_state.b_seq_len.device,
            )
            b_kv_seq_len = self.infer_state.b_seq_len[mtp_size - 1 :: mtp_size]
            b1_cu_q_seq_len, b1_cu_kv_seq_len = gen_cumsum_pad0_tensor(b_q_seq_len, b_kv_seq_len)
            self.cu_seqlens_q = b1_cu_q_seq_len.int()
            self.cu_seqlens_k = b1_cu_kv_seq_len.int()
        else:
            self.cu_seqlens_q = self.infer_state.b1_cu_q_seq_len.int()
            self.cu_seqlens_k = self.infer_state.b1_cu_kv_seq_len.int()

        att_batch_size = self.infer_state.batch_size // (args_mtp_step + 1)
        assert self.infer_state.batch_size % (args_mtp_step + 1) == 0
        model = self.backend.model
        table_len = triton.cdiv(self.infer_state.max_kv_seq_len, self.backend.page_size)
        if (
            self.infer_state.batch_size <= model.graph_max_batch_size
            and self.infer_state.max_kv_seq_len <= model.graph_max_len_in_batch
        ):
            page_buffer = self.backend.get_page_table_buffer()
            shared_table_len = triton.cdiv(model.graph_max_len_in_batch, self.backend.page_size)
            self.page_table = page_buffer[self.infer_state.microbatch_index][
                : att_batch_size * shared_table_len
            ].reshape(att_batch_size, shared_table_len)
        else:
            self.page_table = torch.empty(
                (att_batch_size, table_len),
                dtype=torch.int32,
                device=self.infer_state.input_ids.device,
            )

        if args_mtp_step > 0:
            page_table_copy(
                page_table=self.page_table[:, :table_len],
                req_to_token_indexs=model.req_manager.req_to_token_indexs,
                b_req_idx=self.infer_state.b_req_idx[args_mtp_step :: (args_mtp_step + 1)],
            )
            self.b_att_seq_len = self.infer_state.b_seq_len[args_mtp_step :: (args_mtp_step + 1)].contiguous()
            self.decode_max_q_seq_len = args_mtp_step + 1
        else:
            page_table_copy(
                page_table=self.page_table[:, :table_len],
                req_to_token_indexs=model.req_manager.req_to_token_indexs,
                b_req_idx=self.infer_state.b_req_idx,
            )
            self.b_att_seq_len = self.infer_state.b_seq_len
            self.decode_max_q_seq_len = 1

    def decode_att(
        self, q: Tuple[torch.Tensor, torch.Tensor], k, v, att_control: AttControl = AttControl(), alloc_func=torch.empty
    ):
        assert (
            att_control.use_alibi is False
            and att_control.use_sliding_window is False
            and att_control.use_att_sink is False
        )
        assert v is None
        return self._mla_decode_att(q=q, k=k, v=v, att_control=att_control, alloc_func=alloc_func)

    def _mla_decode_att(
        self, q: Tuple[torch.Tensor, torch.Tensor], k, v, att_control: AttControl, alloc_func=torch.empty
    ):
        q_nope, q_rope = q
        qk_rope_head_dim = 64
        kv_lora_rank = k.shape[-1] - qk_rope_head_dim
        return flash_attn_with_kvcache(
            q=q_rope,
            k_cache=k[:, :, -qk_rope_head_dim:].view(-1, self.backend.page_size, 1, qk_rope_head_dim),
            v_cache=k[:, :, :-qk_rope_head_dim].view(-1, self.backend.page_size, 1, kv_lora_rank),
            qv=q_nope,
            page_table=self.page_table,
            cache_seqlens=self.b_att_seq_len,
            cu_seqlens_q=self.cu_seqlens_q,
            cu_seqlens_k_new=self.cu_seqlens_k,
            max_seqlen_q=self.decode_max_q_seq_len,
            softmax_scale=att_control.mla_decode_dict["softmax_scale"],
            causal=True,
            window_size=(-1, -1),
            softcap=0.0,
            k_descale=None,
            v_descale=None,
            return_softmax_lse=False,
        )
