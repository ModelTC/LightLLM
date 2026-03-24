import dataclasses
import torch
import triton
from ..base_att import BaseAttBackend, BasePrefillAttState, BaseDecodeAttState, AttControl
from lightllm.utils.dist_utils import get_dp_world_size, get_current_device_id
from ...triton_kernel.repack_kv_index import paged_repack_kv_index
from lightllm.utils.envs_utils import get_page_size
from ..flashinfer.env_utils import set_flashinfer_envs


class PagedFlashInferAttBackend(BaseAttBackend):
    def __init__(self, model, page_size=None):
        set_flashinfer_envs()
        super().__init__(model=model)
        self.page_size = page_size or get_page_size()
        tp_world_size = get_dp_world_size()
        self.tp_q_head_num = model.config["num_attention_heads"] // tp_world_size
        self.tp_kv_head_num = max(model.config["num_key_value_heads"] // tp_world_size, 1)
        head_dim = model.config["hidden_size"] // model.config["num_attention_heads"]
        self.head_dim = model.config.get("head_dim", head_dim)
        self.workspace_buffer = torch.empty(512 * 1024 * 1024, dtype=torch.int8, device=get_current_device_id())
        self.max_seq_length = model.max_seq_length
        buffer_len = model.graph_max_batch_size * triton.cdiv(self.max_seq_length, self.page_size)
        self.kv_indices_buffer = [
            torch.empty(buffer_len, dtype=torch.int32, device=get_current_device_id()),
            torch.empty(buffer_len, dtype=torch.int32, device=get_current_device_id()),
        ]
        self.q_data_type = model.data_type
        self.kv_data_type = model.data_type

    def create_att_prefill_state(self, infer_state):
        return PagedFlashInferPrefillAttState(backend=self, infer_state=infer_state)

    def create_att_decode_state(self, infer_state):
        return PagedFlashInferDecodeAttState(backend=self, infer_state=infer_state)


@dataclasses.dataclass
class PagedFlashInferPrefillAttState(BasePrefillAttState):
    prefill_wrapper: object = None

    def init_state(self):
        self.backend: PagedFlashInferAttBackend = self.backend
        import flashinfer

        batch_size = self.infer_state.batch_size
        device = self.infer_state.input_ids.device
        q_starts = self.infer_state.b1_cu_q_seq_len.int()
        kv_starts = self.infer_state.b1_cu_kv_seq_len.int()
        b_page_len = triton.cdiv(self.infer_state.b_seq_len, self.backend.page_size)
        kv_starts[1:] = b_page_len.cumsum(0)
        kv_last_page_len = self.infer_state.b_seq_len - (b_page_len - 1) * self.backend.page_size
        kv_indices = torch.empty(
            batch_size * triton.cdiv(self.backend.max_seq_length, self.backend.page_size),
            dtype=torch.int32,
            device=device,
        )
        paged_repack_kv_index(
            self.infer_state.req_manager.req_to_token_indexs,
            self.infer_state.b_req_idx,
            b_page_len,
            kv_starts[:-1],
            triton.cdiv(self.infer_state.max_kv_seq_len, self.backend.page_size),
            kv_indices,
        )
        self.prefill_wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            self.backend.workspace_buffer,
            qo_indptr_buf=q_starts,
            paged_kv_indptr_buf=kv_starts,
            paged_kv_indices_buf=kv_indices,
            paged_kv_last_page_len_buf=kv_last_page_len,
        )
        self.prefill_wrapper.plan(
            q_starts,
            kv_starts,
            kv_indices,
            kv_last_page_len,
            self.backend.tp_q_head_num,
            self.backend.tp_kv_head_num,
            self.backend.head_dim,
            self.backend.page_size,
            causal=True,
            pos_encoding_mode="NONE",
            logits_soft_cap=0.0,
            q_data_type=self.backend.q_data_type,
            kv_data_type=self.backend.kv_data_type,
        )

    def prefill_att(self, q, k, v, att_control: AttControl = AttControl(), alloc_func=torch.empty):
        assert (
            att_control.use_alibi is False
            and att_control.use_sliding_window is False
            and att_control.use_att_sink is False
        )
        o_tensor = alloc_func(q.shape, q.dtype, device="cuda")
        self.prefill_wrapper.run(
            q,
            (
                k.view(-1, self.backend.page_size, k.shape[1], k.shape[2]),
                v.view(-1, self.backend.page_size, v.shape[1], v.shape[2]),
            ),
            out=o_tensor,
        )
        return o_tensor


@dataclasses.dataclass
class PagedFlashInferDecodeAttState(BaseDecodeAttState):
    kv_last_page_len_buffer: torch.Tensor = None
    kv_indices: torch.Tensor = None
    kv_starts: torch.Tensor = None
    decode_wrapper: object = None

    def init_state(self):
        import flashinfer

        self.backend: PagedFlashInferAttBackend = self.backend
        device = self.infer_state.input_ids.device
        model = self.backend.model
        b_page_len = triton.cdiv(self.infer_state.b_seq_len, self.backend.page_size)
        self.kv_last_page_len_buffer = self.infer_state.b_seq_len - (b_page_len - 1) * self.backend.page_size
        buffer_len = self.infer_state.batch_size * triton.cdiv(self.backend.max_seq_length, self.backend.page_size)
        if (
            self.infer_state.batch_size <= model.graph_max_batch_size
            and self.infer_state.max_kv_seq_len <= model.graph_max_len_in_batch
        ):
            self.kv_indices = self.backend.kv_indices_buffer[self.infer_state.microbatch_index][:buffer_len]
        else:
            self.kv_indices = torch.empty(buffer_len, dtype=torch.int32, device=device)

        self.kv_starts = self.infer_state.b1_cu_kv_seq_len.int()
        self.kv_starts[1:] = b_page_len.cumsum(0)
        paged_repack_kv_index(
            self.infer_state.req_manager.req_to_token_indexs,
            self.infer_state.b_req_idx,
            b_page_len,
            self.kv_starts[:-1],
            triton.cdiv(self.infer_state.max_kv_seq_len, self.backend.page_size),
            self.kv_indices,
        )
        self.decode_wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
            self.backend.workspace_buffer,
            "NHD",
            use_cuda_graph=True,
            use_tensor_cores=True,
            paged_kv_indptr_buffer=self.kv_starts,
            paged_kv_indices_buffer=self.kv_indices,
            paged_kv_last_page_len_buffer=self.kv_last_page_len_buffer,
        )
        self.decode_wrapper.plan(
            self.kv_starts,
            self.kv_indices,
            self.kv_last_page_len_buffer,
            self.backend.tp_q_head_num,
            self.backend.tp_kv_head_num,
            self.backend.head_dim,
            self.backend.page_size,
            q_data_type=self.backend.q_data_type,
            kv_data_type=self.backend.kv_data_type,
            non_blocking=True,
        )

    def copy_for_decode_cuda_graph(self, new_state):
        super().copy_for_decode_cuda_graph(new_state)
        self.decode_wrapper.plan(
            new_state.kv_starts,
            new_state.kv_indices,
            new_state.kv_last_page_len_buffer,
            new_state.backend.tp_q_head_num,
            new_state.backend.tp_kv_head_num,
            new_state.backend.head_dim,
            new_state.backend.page_size,
            q_data_type=new_state.backend.q_data_type,
            kv_data_type=new_state.backend.kv_data_type,
            non_blocking=True,
        )

    def decode_att(self, q, k, v, att_control: AttControl = AttControl(), alloc_func=torch.empty):
        assert (
            att_control.use_alibi is False
            and att_control.use_sliding_window is False
            and att_control.use_att_sink is False
        )
        o_tensor = alloc_func(q.shape, q.dtype)
        self.decode_wrapper.run(
            q,
            (
                k.view(-1, self.backend.page_size, k.shape[1], k.shape[2]),
                v.view(-1, self.backend.page_size, v.shape[1], v.shape[2]),
            ),
            out=o_tensor,
        )
        return o_tensor
