import dataclasses
import torch
from ..base_att import BaseAttBackend, BasePrefillAttState, BaseDecodeAttState, AttControl
from lightllm.utils.dist_utils import get_dp_world_size, get_current_device_id
from ...triton_kernel.repack_kv_index import repack_kv_index
from typing import Tuple


class MlaFlashInferAttBackend(BaseAttBackend):
    def __init__(self, model):
        super().__init__(model=model)
        num_heads = model.config["num_attention_heads"]
        self.tp_q_head_num = num_heads // get_dp_world_size()
        self.qk_nope_head_dim = model.qk_nope_head_dim
        self.qk_rope_head_dim = model.qk_rope_head_dim
        self.kv_lora_rank = model.kv_lora_rank
        self.q_data_type = model.data_type
        self.kv_data_type = model.data_type
        self.workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=get_current_device_id())
        self.max_seq_length = model.max_seq_length
        self.softmax_scale = (self.qk_nope_head_dim + self.qk_rope_head_dim) ** (-0.5)
        self.kv_indices_buffer = [
            torch.empty(
                model.graph_max_batch_size * self.max_seq_length, dtype=torch.int32, device=get_current_device_id()
            ),
            torch.empty(
                model.graph_max_batch_size * self.max_seq_length, dtype=torch.int32, device=get_current_device_id()
            ),
        ]

        from lightllm.models.llama.yarn_rotary_utils import get_deepseek_mscale

        if model.config["rope_scaling"] is not None:
            rope_scaling = model.config["rope_scaling"]
            mscale_all_dim = rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = rope_scaling["factor"]
            if mscale_all_dim:
                mscale = get_deepseek_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale
        return

    def create_att_prefill_state(self, infer_state) -> "MlaFlashInferPrefillAttState":
        return MlaFlashInferPrefillAttState(backend=self, infer_state=infer_state)

    def create_att_decode_state(self, infer_state) -> "MlaFlashInferDecodeAttState":
        return MlaFlashInferDecodeAttState(backend=self, infer_state=infer_state)


@dataclasses.dataclass
class MlaFlashInferPrefillAttState(BasePrefillAttState):
    prefill_wrapper: object = None

    def init_state(self):
        self.backend: MlaFlashInferAttBackend = self.backend

        import flashinfer

        q_starts = self.infer_state.b1_cu_q_seq_len.int()
        kv_starts = self.infer_state.b1_cu_kv_seq_len.int()
        if self.prefill_wrapper is None:
            self.prefill_wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
                self.backend.workspace_buffer, "NHD"
            )
        self.prefill_wrapper.plan(
            qo_indptr=q_starts,
            kv_indptr=kv_starts,
            num_qo_heads=self.backend.tp_q_head_num,
            num_kv_heads=self.backend.tp_q_head_num,
            head_dim_qk=self.backend.qk_nope_head_dim + self.backend.qk_rope_head_dim,
            head_dim_vo=self.backend.qk_nope_head_dim,
            q_data_type=self.backend.q_data_type,
            causal=True,
            sm_scale=self.backend.softmax_scale,
        )
        return

    def prefill_att(
        self,
        q: torch.Tensor,
        k: Tuple[torch.Tensor, torch.Tensor],
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        assert (
            att_control.use_alibi is False
            and att_control.use_sliding_window is False
            and att_control.use_att_sink is False
        )
        return self._mla_prefill_att(
            q=q,
            k=k,
            v=v,
            alloc_func=alloc_func,
        )

    def _mla_prefill_att(
        self, q: torch.Tensor, k: Tuple[torch.Tensor, torch.Tensor], v: torch.Tensor, alloc_func=torch.empty
    ) -> torch.Tensor:
        self.backend: MlaFlashInferAttBackend = self.backend  # for typing
        k_nope, k_rope = k
        o_tensor = alloc_func((q.shape[0], q.shape[1], k_nope.shape[2]), q.dtype, device="cuda")
        q_head_num = q.shape[1]
        k = torch.cat([k_nope, torch.repeat_interleave(k_rope, q_head_num, dim=-2)], dim=-1)
        self.prefill_wrapper.run(q, k, v, out=o_tensor)
        return o_tensor


@dataclasses.dataclass
class MlaFlashInferDecodeAttState(BaseDecodeAttState):
    kv_indices: torch.Tensor = None
    kv_starts: torch.Tensor = None
    decode_wrapper: object = None

    def init_state(self):
        import flashinfer

        self.backend: MlaFlashInferAttBackend = self.backend
        model = self.backend.model
        device = self.infer_state.input_ids.device
        batch_size = self.infer_state.batch_size

        self.kv_starts = self.infer_state.b1_cu_kv_seq_len

        self.q_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device="cuda")
        if batch_size <= model.graph_max_batch_size and self.infer_state.max_kv_seq_len <= model.graph_max_len_in_batch:
            self.kv_indices = self.backend.kv_indices_buffer[self.infer_state.microbatch_index][
                : batch_size * self.backend.max_seq_length
            ]
        else:
            self.kv_indices = torch.empty(
                batch_size * self.backend.max_seq_length,
                dtype=torch.int32,
                device=device,
            )

        repack_kv_index(
            self.infer_state.req_manager.req_to_token_indexs,
            self.infer_state.b_req_idx,
            self.infer_state.b_seq_len,
            self.infer_state.b_kv_start_loc,
            self.infer_state.max_kv_seq_len,
            self.kv_indices,
        )
        assert self.decode_wrapper is None

        self.decode_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
            self.backend.workspace_buffer,
            use_cuda_graph=True,
            qo_indptr=self.q_indptr,
            kv_indices=self.kv_indices,
            kv_indptr=self.kv_starts,
            kv_len_arr=self.infer_state.b_seq_len,
        )
        self.decode_wrapper.plan(
            self.q_indptr,
            self.kv_starts,
            self.kv_indices,
            self.infer_state.b_seq_len,
            self.backend.tp_q_head_num,
            self.backend.kv_lora_rank,
            self.backend.qk_rope_head_dim,
            1,
            False,  # causal
            self.backend.softmax_scale,
            self.backend.q_data_type,
            self.backend.kv_data_type,
        )
        return

    def copy_for_decode_cuda_graph(self, new_state: "MlaFlashInferDecodeAttState"):
        super().copy_for_decode_cuda_graph(new_state)
        self.decode_wrapper.plan(
            new_state.q_indptr,
            new_state.kv_starts,
            new_state.kv_indices,
            new_state.infer_state.b_seq_len,
            new_state.backend.tp_q_head_num,
            new_state.backend.kv_lora_rank,
            new_state.backend.qk_rope_head_dim,
            1,
            False,  # causal
            new_state.backend.softmax_scale,
            new_state.backend.q_data_type,
            new_state.backend.kv_data_type,
        )

    def decode_att(
        self,
        q: Tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ):
        assert (
            att_control.use_alibi is False
            and att_control.use_sliding_window is False
            and att_control.use_att_sink is False
        )

        assert v is None

        return self._mla_decode_att(
            q=q,
            k=k,
            v=v,
            att_control=att_control,
            alloc_func=alloc_func,
        )

    def _mla_decode_att(
        self,
        q: Tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl,
        alloc_func=torch.empty,
    ):
        qk_rope_head_dim = 64
        q_nope, q_rope = q
        o_tensor = alloc_func(q_nope.shape, dtype=q_nope.dtype, device=q_nope.device)
        assert att_control.mla_decode

        self.decode_wrapper.run(
            q_nope,
            q_rope,
            k[:, :, :-qk_rope_head_dim],
            k[:, :, -qk_rope_head_dim:],
            out=o_tensor,
            return_lse=False,
        )
        return o_tensor
