import dataclasses
import torch
from ..base_att import BaseAttBackend, BasePrefillAttState, BaseDecodeAttState, AttControl
from typing import Optional, TYPE_CHECKING
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.utils.sgl_utils import flash_attn_with_kvcache, flash_attn_with_kvcache_autotune
from lightllm.common.basemodel.triton_kernel.fa3_utils import (
    build_dynamic_spec_fa3_decode_params,
    page_table_copy,
)
from lightllm.common.basemodel.triton_kernel.gen_prefill_params import gen_cumsum_pad0_tensor
from lightllm.utils.envs_utils import get_env_start_args


class Fa3AttBackend(BaseAttBackend):
    def __init__(self, model):
        super().__init__(model=model)
        self.get_page_table_buffer()  # init

    def get_page_table_buffer(self):
        """
        用于减少 decode graph 捕获的时候, 造成显存二次方增长的情况.
        """
        model = self.model
        if not hasattr(self, "_shared_page_table_buffer"):
            max_att_batch_size = model.graph_max_batch_size
            if not get_env_start_args().mtp_dynamic_verify:
                # FA3 merges each fixed speculative block into one attention sequence.
                max_att_batch_size //= model.decode_batch_multiplier

            buffer_count = 2 if model.args.enable_decode_microbatch_overlap else 1
            self._shared_page_table_buffer = [
                torch.empty(
                    max_att_batch_size * model.graph_max_len_in_batch,
                    dtype=torch.int32,
                    device=get_current_device_id(),
                )
                for _ in range(buffer_count)
            ]
        return self._shared_page_table_buffer

    def create_att_prefill_state(self, infer_state) -> "Fa3PrefillAttState":
        return Fa3PrefillAttState(backend=self, infer_state=infer_state)

    def create_att_decode_state(self, infer_state) -> "Fa3DecodeAttState":
        return Fa3DecodeAttState(backend=self, infer_state=infer_state)


@dataclasses.dataclass
class Fa3PrefillAttState(BasePrefillAttState):
    cu_seqlens_q: torch.Tensor = None
    cu_seqlens_k: torch.Tensor = None
    page_table: torch.Tensor = None

    def init_state(self):
        self.cu_seqlens_q = self.infer_state.b1_cu_q_seq_len.int()
        self.cu_seqlens_k = self.infer_state.b1_cu_kv_seq_len.int()
        self.page_table = torch.empty(
            (self.infer_state.batch_size, self.infer_state.max_kv_seq_len),
            dtype=torch.int32,
            device=self.infer_state.input_ids.device,
        )
        self.page_table.copy_(
            self.infer_state.req_manager.req_to_token_indexs[
                self.infer_state.b_req_idx, : self.infer_state.max_kv_seq_len
            ]
        )

    def prefill_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        assert att_control.use_alibi is False
        return self._nomarl_prefill_att(
            q=q,
            k=k,
            v=v,
            att_control=att_control,
            alloc_func=alloc_func,
        )

    def _nomarl_prefill_att(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, att_control: AttControl, alloc_func=torch.empty
    ) -> torch.Tensor:
        self.backend: Fa3AttBackend = self.backend  # for typing

        if att_control.use_sliding_window:
            window_size = att_control.sliding_window
        else:
            window_size = (-1, -1)

        if att_control.use_att_sink:
            sink_weight: torch.Tensor = att_control.sink_weight
        else:
            sink_weight = None

        k_descale, v_descale = None, None  # disable quantization
        Lq = q.shape[-1]
        sm_scale = 1.0 / (Lq ** 0.5)
        o = flash_attn_with_kvcache(
            q=q,
            k_cache=k.view(k.shape[0], 1, k.shape[1], k.shape[2]),
            v_cache=v.view(v.shape[0], 1, v.shape[1], v.shape[2]),
            page_table=self.page_table,
            cache_seqlens=self.infer_state.b_seq_len,
            cu_seqlens_q=self.cu_seqlens_q,
            cu_seqlens_k_new=self.cu_seqlens_k,
            max_seqlen_q=self.infer_state.max_q_seq_len,
            softmax_scale=sm_scale,
            causal=self.infer_state.prefill_causal,
            window_size=window_size,
            softcap=0.0,
            k_descale=k_descale,
            v_descale=v_descale,
            return_softmax_lse=False,
            sinks=sink_weight,
        )
        return o


@dataclasses.dataclass
class Fa3DecodeAttState(BaseDecodeAttState):
    cu_seqlens_q: torch.Tensor = None
    cu_seqlens_k: torch.Tensor = None
    page_table: torch.Tensor = None
    b_att_seq_len: torch.Tensor = None
    # 在是否开启mtp 的不同模式下，其设置不同的值，可以加速算子的运行。
    decode_max_q_seq_len: int = None

    def init_state(self):
        self.backend: Fa3AttBackend = self.backend
        draft_step = self.infer_state.draft_step
        decode_rows_per_request = draft_step + 1
        uses_dynamic_spec_verify_layout = self.backend.uses_dynamic_spec_verify_layout(self.infer_state)

        if draft_step > 0 and not uses_dynamic_spec_verify_layout:
            assert self.infer_state.batch_size % decode_rows_per_request == 0, (
                "FA3 fixed-layout decode requires batch_size to be divisible by draft_step + 1, "
                f"got batch_size={self.infer_state.batch_size}, draft_step={draft_step}."
            )

        # 修正 mtp 在 fa3 下的输入。
        if uses_dynamic_spec_verify_layout:
            (b_q_seq_len, b_kv_seq_len, b_att_req_idx, self.b_att_seq_len,) = build_dynamic_spec_fa3_decode_params(
                b_req_idx=self.infer_state.b_req_idx,
                b_seq_len=self.infer_state.b_seq_len,
                b_mark_shared_group=self.infer_state.b_mark_shared_group,
                att_batch_size=self.infer_state.batch_size,
                hold_req_id=self.backend.model.req_manager.HOLD_REQUEST_ID,
            )
        elif draft_step > 0:
            b_q_seq_len = torch.full(
                (self.infer_state.b_seq_len.shape[0] // decode_rows_per_request,),
                fill_value=decode_rows_per_request,
                dtype=torch.int32,
                device=self.infer_state.b_seq_len.device,
            )
            b_kv_seq_len = self.infer_state.b_seq_len[draft_step::decode_rows_per_request]
            b_att_req_idx = self.infer_state.b_req_idx[draft_step::decode_rows_per_request]
            self.b_att_seq_len = b_kv_seq_len.contiguous()
        else:
            b_att_req_idx = self.infer_state.b_req_idx
            self.b_att_seq_len = self.infer_state.b_seq_len

        if draft_step > 0:
            b1_cu_q_seq_len, b1_cu_kv_seq_len = gen_cumsum_pad0_tensor(b_q_seq_len, b_kv_seq_len)
            self.cu_seqlens_q = b1_cu_q_seq_len.int()
            self.cu_seqlens_k = b1_cu_kv_seq_len.int()
        else:
            self.cu_seqlens_q = self.infer_state.b1_cu_q_seq_len.int()
            self.cu_seqlens_k = self.infer_state.b1_cu_kv_seq_len.int()

        att_batch_size = b_att_req_idx.shape[0]
        model = self.backend.model
        # 可以使用 cuda graph的时候从 buffer中申请
        if (
            self.infer_state.batch_size <= model.graph_max_batch_size
            and self.infer_state.max_kv_seq_len <= model.graph_max_len_in_batch
        ):
            page_buffer = self.backend.get_page_table_buffer()
            self.page_table = page_buffer[self.infer_state.microbatch_index][
                : att_batch_size * model.graph_max_len_in_batch
            ].reshape(att_batch_size, model.graph_max_len_in_batch)
        else:
            self.page_table = torch.empty(
                (att_batch_size, self.infer_state.max_kv_seq_len),
                dtype=torch.int32,
                device=self.infer_state.input_ids.device,
            )

        page_table_copy(
            page_table=self.page_table[:, : self.infer_state.max_kv_seq_len],
            req_to_token_indexs=model.req_manager.req_to_token_indexs,
            b_req_idx=b_att_req_idx,
        )
        self.decode_max_q_seq_len = decode_rows_per_request
        return

    def copy_for_decode_cuda_graph(self, new_state: "Fa3DecodeAttState"):
        super().copy_for_decode_cuda_graph(new_state)

    def decode_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ):
        assert att_control.use_alibi is False
        return self._normal_decode_att(
            q=q,
            k=k,
            v=v,
            att_control=att_control,
            alloc_func=alloc_func,
        )

    def _normal_decode_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl,
        alloc_func=torch.empty,
    ):
        if att_control.use_sliding_window:
            window_size = att_control.sliding_window
        else:
            window_size = (-1, -1)

        if att_control.use_att_sink:
            sink_weight: torch.Tensor = att_control.sink_weight
        else:
            sink_weight = None

        k_descale, v_descale = None, None  # disable quantization
        Lq = q.shape[-1]
        sm_scale = 1.0 / (Lq ** 0.5)
        o = flash_attn_with_kvcache_autotune(
            q=q,
            k_cache=k.view(k.shape[0], 1, k.shape[1], k.shape[2]),
            v_cache=v.view(v.shape[0], 1, v.shape[1], v.shape[2]),
            page_table=self.page_table,
            cache_seqlens=self.b_att_seq_len,
            cu_seqlens_q=self.cu_seqlens_q,
            cu_seqlens_k_new=self.cu_seqlens_k,
            max_seqlen_q=self.decode_max_q_seq_len,
            softmax_scale=sm_scale,
            causal=self.infer_state.decode_causal,
            window_size=window_size,
            softcap=0.0,
            k_descale=k_descale,
            v_descale=v_descale,
            return_softmax_lse=False,
            sinks=sink_weight,
        )
        return o
