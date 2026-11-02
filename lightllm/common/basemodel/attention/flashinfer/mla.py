from ..mla import BoundMla, MlaPrefillInput, MlaDecodeInput
import dataclasses
import weakref
import torch
import triton
from ..base_att import BaseAttBackend, BaseAttState
from lightllm.utils.dist_utils import get_dp_world_size, get_current_device_id
from ...triton_kernel.flashinfer_mla_plan import fill_mla_decode_plan_for_cuda_graph
from typing import Tuple
from .env_utils import set_flashinfer_envs
from .utils import should_init_decode_wrapper
from .paged_metadata import prepare_page_metadata, page_index_memory_demands, reserve_page_index_buffers


class MlaFlashInferAttBackend(BaseAttBackend):
    supports_mla = True
    workspace_buffer_key = "flashinfer_mla"
    workspace_buffer_size = 256 * 1024 * 1024

    def reserve_runtime_resources(self):
        args = self.model.args
        lanes = 2 if args.enable_decode_microbatch_overlap or args.enable_prefill_microbatch_overlap else 1
        for lane in range(lanes):
            self.get_gpu_workspace_buffer(self.workspace_buffer_key, self.workspace_buffer_size, lane_id=lane)
        reserve_page_index_buffers(self)

    def runtime_memory_demands(self):
        args = self.model.args
        lanes = 2 if args.enable_decode_microbatch_overlap or args.enable_prefill_microbatch_overlap else 1
        return self.workspace_memory_demands(
            self.workspace_buffer_key, self.workspace_buffer_size, lanes
        ) + page_index_memory_demands(self)

    def __init__(self, model, attention_shape=None):
        set_flashinfer_envs()
        super().__init__(model=model)
        self._init_infer_page_size()
        num_heads = model.config["num_attention_heads"]
        self.tp_q_head_num = num_heads // get_dp_world_size()
        self.qk_nope_head_dim = model.qk_nope_head_dim
        self.qk_rope_head_dim = model.qk_rope_head_dim
        self.kv_lora_rank = model.kv_lora_rank
        self.v_head_dim = model.v_head_dim
        self.q_data_type = model.data_type
        self.kv_data_type = model.data_type
        self.max_seq_length = model.max_seq_length
        self.max_page_num = triton.cdiv(self.max_seq_length, self.infer_page_size)
        self.softmax_scale = (self.qk_nope_head_dim + self.qk_rope_head_dim) ** (-0.5)
        self.kv_indices_buffer = None

        from lightllm.models.llama.yarn_rotary_utils import get_deepseek_mscale

        if model.config["rope_scaling"] is not None:
            rope_scaling = model.config["rope_scaling"]
            mscale_all_dim = rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = rope_scaling["factor"]
            if mscale_all_dim:
                mscale = get_deepseek_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale
        if attention_shape is not None:
            (
                self.tp_q_head_num,
                self.qk_nope_head_dim,
                self.qk_rope_head_dim,
                self.kv_lora_rank,
                self.v_head_dim,
                self.q_data_type,
                self.softmax_scale,
            ) = attention_shape
            self.kv_data_type = self.q_data_type
        return

    def for_attention_spec(self, spec):
        shape = (
            self.tp_q_head_num,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.kv_lora_rank,
            self.v_head_dim,
            self.q_data_type,
            self.softmax_scale,
        )
        return self if shape == spec.shape_key else type(self)(model=self.model, attention_shape=spec.shape_key)

    def _init_infer_page_size(self):
        self.infer_page_size = self.model.args.page_size
        assert self.model.args.page_size % self.infer_page_size == 0, (
            f"model page_size {self.model.args.page_size} "
            f"must be divisible by infer_page_size {self.infer_page_size}"
        )

    def create_att_prefill_state(self, infer_state) -> "MlaFlashInferPrefillAttState":
        return MlaFlashInferPrefillAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))

    def create_att_decode_state(self, infer_state) -> "MlaFlashInferDecodeAttState":
        return MlaFlashInferDecodeAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))


@dataclasses.dataclass
class MlaFlashInferPrefillAttState(BaseAttState):
    prefill_wrapper: object = None

    def init_state(self):
        self.backend: MlaFlashInferAttBackend = self.backend

        import flashinfer

        q_starts = self.infer_state.b1_cu_q_seq_len.int()
        kv_starts = self.infer_state.b1_cu_kv_seq_len.int()
        if self.prefill_wrapper is None:
            self.prefill_wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
                self.backend.get_gpu_workspace_buffer(
                    key_name=self.backend.workspace_buffer_key,
                    workspace_size=self.backend.workspace_buffer_size,
                    lane_id=self.infer_state.microbatch_index,
                ),
                "NHD",
            )
        self.prefill_wrapper.plan(
            qo_indptr=q_starts,
            kv_indptr=kv_starts,
            num_qo_heads=self.backend.tp_q_head_num,
            num_kv_heads=self.backend.tp_q_head_num,
            head_dim_qk=self.backend.qk_nope_head_dim + self.backend.qk_rope_head_dim,
            head_dim_vo=self.backend.v_head_dim,
            q_data_type=self.backend.q_data_type,
            causal=True,
            sm_scale=self.backend.softmax_scale,
        )
        return

    def run(self, inputs: MlaPrefillInput, binding: BoundMla, alloc_func=torch.empty):
        q, k, v = inputs.q, (inputs.k_nope, inputs.k_rope), inputs.v
        self.backend: MlaFlashInferAttBackend = self.backend  # for typing
        k_nope, k_rope = k
        o_tensor = alloc_func((q.shape[0], q.shape[1], v.shape[-1]), q.dtype, device="cuda")
        q_head_num = q.shape[1]
        k = torch.cat([k_nope, torch.repeat_interleave(k_rope, q_head_num, dim=-2)], dim=-1)
        self.prefill_wrapper.run(q, k, v, out=o_tensor)
        return o_tensor


@dataclasses.dataclass
class MlaFlashInferDecodeAttState(BaseAttState):
    graph_buffer_names = ("kv_indices", "kv_starts", "q_indptr_host", "q_indptr")
    kv_indices: torch.Tensor = None
    kv_starts: torch.Tensor = None
    q_indptr_host: torch.Tensor = None
    q_indptr: torch.Tensor = None
    decode_wrapper: object = None

    def _should_init_decode_wrapper(self) -> bool:
        return should_init_decode_wrapper(self.infer_state)

    def init_state(self):
        import flashinfer

        self.backend: MlaFlashInferAttBackend = self.backend
        batch_size = self.infer_state.batch_size
        pages = prepare_page_metadata(self.backend, self.infer_state)
        self.kv_starts, self.kv_indices = pages.starts, pages.indices
        self.q_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=self.infer_state.input_ids.device)
        self.q_indptr_host = torch.arange(batch_size + 1, dtype=torch.int32, device="cpu")
        if not self._should_init_decode_wrapper():
            return

        assert self.decode_wrapper is None

        self.decode_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
            self.backend.get_gpu_workspace_buffer(
                key_name=self.backend.workspace_buffer_key,
                workspace_size=self.backend.workspace_buffer_size,
                lane_id=self.infer_state.microbatch_index,
            ),
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
            self.backend.infer_page_size,
            False,  # causal
            self.backend.softmax_scale,
            self.backend.q_data_type,
            self.backend.kv_data_type,
        )
        return

    def copy_for_decode_cuda_graph(self, new_state: "MlaFlashInferDecodeAttState"):
        super().copy_for_decode_cuda_graph(new_state)
        self._refresh_cuda_graph_decode_plan(new_state.infer_state.b_seq_len, new_state.infer_state.max_kv_seq_len)
        return

    def _refresh_cuda_graph_decode_plan(self, kv_lens: torch.Tensor, max_kv_len: int):
        # Prefer the GPU-generated split plan for long decode; use exact non-split for
        # short or unsupported graph shapes.
        fill_mla_decode_plan_for_cuda_graph(
            decode_wrapper=self.decode_wrapper,
            kv_page_indptr=self.kv_starts,
            kv_lens=kv_lens,
            batch_size=self.infer_state.batch_size,
            num_heads=self.backend.tp_q_head_num,
            max_kv_len=max_kv_len,
        )

    def run(self, inputs: MlaDecodeInput, binding: BoundMla, alloc_func=torch.empty):
        q, k = (inputs.q_nope, inputs.q_rope), binding.cache
        qk_rope_head_dim = binding.rope_dim
        q_nope, q_rope = q
        o_tensor = alloc_func(q_nope.shape, dtype=q_nope.dtype, device=q_nope.device)

        self.decode_wrapper.run(
            q_nope,
            q_rope,
            k[:, :, :-qk_rope_head_dim].view(-1, self.backend.infer_page_size, 1, k.shape[-1] - qk_rope_head_dim),
            k[:, :, -qk_rope_head_dim:].view(-1, self.backend.infer_page_size, 1, qk_rope_head_dim),
            out=o_tensor,
            return_lse=False,
        )
        return o_tensor
