import dataclasses
import torch
import triton
from ..base_att import BaseAttBackend, BasePrefillAttState, BaseDecodeAttState, AttControl
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.utils.sgl_utils import flash_attn_with_kvcache
from lightllm.utils.envs_utils import get_env_start_args, get_page_size
from lightllm.common.basemodel.triton_kernel.fa3_utils import page_table_copy
from lightllm.common.basemodel.triton_kernel.gen_prefill_params import gen_cumsum_pad0_tensor
from typing import Any


class PagedFa3AttBackend(BaseAttBackend):

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

    def get_decode_seq_len_cpu_buffers(self, min_len: int):
        """Pinned CPU int32 buffers reused for npu_fused_infer_attention_score list args."""
        model = self.model
        cap = max(min_len, model.graph_max_batch_size)
        if not hasattr(self, "_decode_seq_len_cpu_q") or self._decode_seq_len_cpu_q.shape[0] < min_len:
            self._decode_seq_len_cpu_q = torch.empty(cap, dtype=torch.int32, pin_memory=True)
            self._decode_seq_len_cpu_kv = torch.empty(cap, dtype=torch.int32, pin_memory=True)
        return self._decode_seq_len_cpu_q, self._decode_seq_len_cpu_kv

    def create_att_prefill_state(self, infer_state):
        return PagedFa3PrefillAttState(backend=self, infer_state=infer_state)

    def create_att_decode_state(self, infer_state):
        return PagedFa3DecodeAttState(backend=self, infer_state=infer_state)


@dataclasses.dataclass
class PagedFa3PrefillAttState(BasePrefillAttState):
    cu_seqlens_q: torch.Tensor = None
    cu_seqlens_k: torch.Tensor = None
    page_table: torch.Tensor = None
    atten_mask: torch.Tensor = None

    def init_state(self):
        self.cu_seqlens_q = self.infer_state.b1_cu_q_seq_len.int()
        self.cu_seqlens_k = self.infer_state.b1_cu_kv_seq_len.int()
        table_len = triton.cdiv(self.infer_state.max_kv_seq_len, self.backend.page_size)
        self.page_table = torch.empty(
            (self.infer_state.batch_size, table_len),
            dtype=torch.int32,
            device=self.infer_state.input_ids.device,
        )
        page_table_copy(
            page_table=self.page_table,
            req_to_token_indexs=self.infer_state.req_manager.req_to_token_indexs,
            b_req_idx=self.infer_state.b_req_idx,
            page_size=self.backend.page_size,
        )
        if self.atten_mask is None:
            self.atten_mask = torch.triu(torch.ones([2048, 2048]), diagonal=1).to(dtype=torch.int8, device=self.infer_state.input_ids.device)

    def prefill_att(self, q, k, v, att_control: AttControl = AttControl(), alloc_func=torch.empty):
        assert att_control.use_alibi is False
        return self._normal_prefill_att(q=q, k=k, v=v, att_control=att_control, alloc_func=alloc_func)

    def _normal_prefill_att(self, q, k, v, att_control: AttControl, alloc_func=torch.empty):
        if att_control.use_sliding_window:
            window_size = att_control.sliding_window
        else:
            window_size = (-1, -1)

        if att_control.use_att_sink:
            sink_weight = att_control.sink_weight
        else:
            sink_weight = None

        sm_scale = 1.0 / (q.shape[-1] ** 0.5)

        if q.device.type == "npu":
            import torch_npu

            N_KV, HEAD_DIM = k.shape[-2:]
            # to (num_blocks, block_size, hidden_size)
            key = k.view(-1, self.backend.page_size, N_KV * HEAD_DIM)
            value = v.view(-1, self.backend.page_size, N_KV * HEAD_DIM)
            out = torch_npu.npu_fused_infer_attention_score(
                query=q,
                key=key,
                value=value,
                input_layout="TND",
                sparse_mode=3,
                atten_mask=self.atten_mask,
                scale=sm_scale,
                actual_seq_lengths=self.infer_state.b1_cu_q_seq_len_cpu,
                actual_seq_lengths_kv=self.infer_state.b_cu_kv_seq_len_cpu,
                num_heads=q.shape[-2],
                num_key_value_heads=N_KV,
                block_table=self.page_table,
                block_size=self.backend.page_size,
            )[0]
            return out
        else:
            return flash_attn_with_kvcache(
                q=q,
                k_cache=k.view(-1, self.backend.page_size, k.shape[1], k.shape[2]),
                v_cache=v.view(-1, self.backend.page_size, v.shape[1], v.shape[2]),
                page_table=self.page_table,
                cache_seqlens=self.infer_state.b_seq_len,
                cu_seqlens_q=self.cu_seqlens_q,
                cu_seqlens_k_new=self.cu_seqlens_k,
                max_seqlen_q=self.infer_state.max_q_seq_len,
                softmax_scale=sm_scale,
                causal=True,
                window_size=window_size,
                softcap=0.0,
                k_descale=None,
                v_descale=None,
                return_softmax_lse=False,
                sinks=sink_weight,
            )


@dataclasses.dataclass
class PagedFa3DecodeAttState(BaseDecodeAttState):
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
                page_size=self.backend.page_size,
            )
            self.b_att_seq_len = self.infer_state.b_seq_len[args_mtp_step :: (args_mtp_step + 1)].contiguous()
            self.decode_max_q_seq_len = args_mtp_step + 1
        else:
            page_table_copy(
                page_table=self.page_table[:, :table_len],
                req_to_token_indexs=model.req_manager.req_to_token_indexs,
                b_req_idx=self.infer_state.b_req_idx,
                page_size=self.backend.page_size,
            )
            self.b_att_seq_len = self.infer_state.b_seq_len
            self.decode_max_q_seq_len = 1

    def decode_att(self, q, k, v, att_control: AttControl = AttControl(), alloc_func=torch.empty):
        assert att_control.use_alibi is False
        return self._normal_decode_att(q=q, k=k, v=v, att_control=att_control, alloc_func=alloc_func)

    def _normal_decode_att(self, q, k, v, att_control: AttControl, alloc_func=torch.empty):
        if att_control.use_sliding_window:
            window_size = att_control.sliding_window
        else:
            window_size = (-1, -1)

        if att_control.use_att_sink:
            sink_weight = att_control.sink_weight
        else:
            sink_weight = None

        sm_scale = 1.0 / (q.shape[-1] ** 0.5)
        if q.device.type == "npu":
            import torch_npu

            N_Q = q.shape[-2]
            N_KV, HEAD_DIM = k.shape[-2:]

            k = k.view(-1, self.backend.page_size, N_KV * HEAD_DIM)
            v = v.view(-1, self.backend.page_size, N_KV * HEAD_DIM)

            output = torch_npu.npu_fused_infer_attention_score(
                query=q,
                key=k,
                value=v,
                input_layout="TND",
                scale=sm_scale,
                actual_seq_lengths=self.infer_state.b1_cu_q_seq_len_cpu,
                actual_seq_lengths_kv=self.infer_state.b_cu_kv_seq_len_cpu,
                num_heads=N_Q,
                num_key_value_heads=N_KV,
                block_table=self.page_table,
                block_size=self.backend.page_size,
            )[0]

            return output
        else:
            return flash_attn_with_kvcache(
                q=q,
                k_cache=k.view(-1, self.backend.page_size, k.shape[1], k.shape[2]),
                v_cache=v.view(-1, self.backend.page_size, v.shape[1], v.shape[2]),
                page_table=self.page_table,
                cache_seqlens=self.b_att_seq_len,
                cu_seqlens_q=self.cu_seqlens_q,
                cu_seqlens_k_new=self.cu_seqlens_k,
                max_seqlen_q=self.decode_max_q_seq_len,
                softmax_scale=sm_scale,
                causal=True,
                window_size=window_size,
                softcap=0.0,
                k_descale=None,
                v_descale=None,
                return_softmax_lse=False,
                sinks=sink_weight,
            )


def update_attn_params(
    batch_size: int,
    actual_seq_lengths: list[int],
    actual_seq_lengths_kv: list[int],
    update_stream: Any,
):
    import torch_npu
    from lightllm.common.basemodel.acl_graph import get_attn_params

    attn_params = get_attn_params()
    handles = attn_params.handles[batch_size]
    events = attn_params.events[batch_size]
    workspace = attn_params.workspaces[batch_size]
    params_list = attn_params.attn_params[batch_size]

    with torch.npu.stream(update_stream):
        for handle, event, attn_param in zip(handles, events, params_list):
            (q, k, v, sm_scale, N_Q, N_KV, page_table, block_size, output, softmax_lse) = attn_param
            torch.npu.graph_task_update_begin(update_stream, handle)
            torch_npu.npu_fused_infer_attention_score.out(
                q,
                k,
                v,
                input_layout="TND",
                scale=sm_scale,
                actual_seq_lengths=actual_seq_lengths,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                num_heads=N_Q,
                num_key_value_heads=N_KV,
                block_table=page_table,
                block_size=block_size,
                workspace=workspace,
                out=[output, softmax_lse],
            )
            torch.npu.graph_task_update_end(update_stream)
            event.record(update_stream)
