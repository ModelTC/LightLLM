import dataclasses

import torch
import triton

from lightllm.common.attention.api import BackendResources
from lightllm.common.attention.api.backend import LayerBinding
from lightllm.common.attention.api.inputs import DenseInputs
from lightllm.common.attention.backends._flashinfer import set_flashinfer_envs
from lightllm.common.attention.backends.base import BaseAttentionBackend, PreparedPlan
from lightllm.common.basemodel.triton_kernel.gen_prefill_params import gen_cumsum_pad0_tensor
from lightllm.common.basemodel.triton_kernel.repack_kv_index import repack_kv_index
from lightllm.utils.dist_utils import get_dp_world_size


@dataclasses.dataclass
class FlashInferPrefillPlan(PreparedPlan):
    prefill_wrapper: object = None


@dataclasses.dataclass
class FlashInferDecodePlan(PreparedPlan):
    kv_last_page_len_buffer: torch.Tensor = None
    kv_indices: torch.Tensor = None
    kv_starts: torch.Tensor = None
    decode_wrapper: object = None


class FlashInferAttBackend(BaseAttentionBackend):
    provider = "flashinfer"
    dependency = "flashinfer"
    activation_dtypes = frozenset({torch.float16, torch.bfloat16})
    supports_expert_parallel = False
    cache_formats = frozenset({"None"})
    workspace_buffer_key = "flashinfer_fp"
    workspace_buffer_size = 512 * 1024 * 1024

    def __init__(self, model):
        set_flashinfer_envs()
        super().__init__(model=model)
        self._init_infer_page_size()
        tp_world_size = get_dp_world_size()
        self.tp_q_head_num = model.config["num_attention_heads"] // tp_world_size
        self.tp_kv_head_num = max(model.config["num_key_value_heads"] // tp_world_size, 1)
        head_dim = model.config["hidden_size"] // model.config["num_attention_heads"]
        self.head_dim = model.config.get("head_dim", head_dim)
        self.max_seq_length = model.max_seq_length
        self.max_page_num = triton.cdiv(self.max_seq_length, self.infer_page_size)
        self.q_data_type = model.data_type
        self.kv_data_type = model.data_type

    def for_spec(self, spec):
        backend = super().for_spec(spec)
        backend.tp_q_head_num = spec.num_q_heads
        backend.tp_kv_head_num = spec.cache.key_heads
        backend.head_dim = spec.head_size
        return backend

    def resource_requirements(self, capacity, *, capture):
        return BackendResources(((self.workspace_buffer_key, self.workspace_buffer_size, torch.uint8),))

    def _init_infer_page_size(self):
        self.infer_page_size = self.page_size
        assert (
            self.page_size % self.infer_page_size == 0
        ), f"model page_size {self.page_size} must be divisible by infer_page_size {self.infer_page_size}"

    def _allocate_prefill_plan(self, capacity, capture, workspace):
        return FlashInferPrefillPlan(capacity, capture, workspace)

    def _allocate_decode_plan(self, capacity, capture, workspace):
        return FlashInferDecodePlan(capacity, capture, workspace)

    def _forward_prefill(self, layer: LayerBinding, plan, inputs: DenseInputs, *, out=None) -> torch.Tensor:
        alloc_func = layer.alloc_tensor
        q = inputs.query
        k = layer.params.key
        v = layer.params.value
        return self._prefill_nomarl_prefill_att(layer, plan, q=q, k=k, v=v, alloc_func=alloc_func)

    def _prefill_nomarl_prefill_att(
        self, layer, plan, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, alloc_func=torch.empty
    ) -> torch.Tensor:
        o_tensor = alloc_func(q.shape, q.dtype, device="cuda")
        plan.prefill_wrapper.run(
            q,
            (
                k.view(-1, self.infer_page_size, k.shape[1], k.shape[2]),
                v.view(-1, self.infer_page_size, v.shape[1], v.shape[2]),
            ),
            out=o_tensor,
        )
        return o_tensor

    def _decode_refresh_cuda_graph_decode_plan(self, plan, max_kv_len: int):
        from flashinfer.decode import fast_decode_plan

        uniform_kv_indptr_cpu = torch.arange(plan.batch.batch_size + 1, dtype=torch.int32, device="cpu") * triton.cdiv(
            max_kv_len, self.infer_page_size
        )
        fast_decode_plan(
            plan.decode_wrapper,
            indptr=plan.kv_starts,
            indices=plan.kv_indices,
            last_page_len=plan.kv_last_page_len_buffer,
            num_qo_heads=self.tp_q_head_num,
            num_kv_heads=self.tp_kv_head_num,
            head_dim=self.head_dim,
            page_size=self.infer_page_size,
            q_data_type=self.q_data_type,
            kv_data_type=self.kv_data_type,
            non_blocking=True,
            global_override_indptr_cpu=uniform_kv_indptr_cpu,
        )
        return

    def _forward_decode(self, layer: LayerBinding, plan, inputs: DenseInputs, *, out=None):
        alloc_func = layer.alloc_tensor
        q = inputs.query
        k = layer.params.key
        v = layer.params.value
        return self._decode_normal_decode_att(layer, plan, q=q, k=k, v=v, alloc_func=alloc_func)

    def _decode_normal_decode_att(
        self, layer, plan, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, alloc_func=torch.empty
    ):
        o_tensor = alloc_func(q.shape, q.dtype)
        plan.decode_wrapper.run(
            q,
            (
                k.view(-1, self.infer_page_size, k.shape[1], k.shape[2]),
                v.view(-1, self.infer_page_size, v.shape[1], v.shape[2]),
            ),
            out=o_tensor,
        )
        return o_tensor

    def _prepare_prefill(self, plan):
        import flashinfer

        batch = plan.batch
        q_starts = batch.b1_cu_q_seq_len.int()
        b_page_len = (batch.b_seq_len + self.infer_page_size - 1) // self.infer_page_size
        kv_starts, _ = gen_cumsum_pad0_tensor(b_page_len, b_page_len)
        last_page_len = batch.b_seq_len - (b_page_len - 1) * self.infer_page_size
        indices = plan.workspace.get("flashinfer_indices", max(1, batch.batch_size * self.max_page_num), torch.int32)
        repack_kv_index(
            req_to_token_indexs=batch.req_to_token_indexs,
            b_req_idx=batch.b_req_idx,
            b_token_len=batch.b_seq_len,
            b_page_start_loc=kv_starts[:-1],
            max_token_len=batch.max_kv_seq_len,
            out_page_indices=indices,
            page_size=self.infer_page_size,
        )
        plan.prefill_wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            plan.workspace.get(self.workspace_buffer_key, self.workspace_buffer_size),
            qo_indptr_buf=q_starts,
            paged_kv_indptr_buf=kv_starts,
            paged_kv_indices_buf=indices,
            paged_kv_last_page_len_buf=last_page_len,
        )
        plan.prefill_wrapper.plan(
            q_starts,
            kv_starts,
            indices,
            last_page_len,
            self.tp_q_head_num,
            self.tp_kv_head_num,
            self.head_dim,
            self.infer_page_size,
            causal=self.causal,
            pos_encoding_mode="NONE",
            logits_soft_cap=0.0,
            q_data_type=self.q_data_type,
            kv_data_type=self.kv_data_type,
        )

    def _prepare_decode(self, plan):
        import flashinfer

        batch = plan.batch
        b_page_len = (batch.b_seq_len + self.infer_page_size - 1) // self.infer_page_size
        starts, _ = gen_cumsum_pad0_tensor(b_page_len, b_page_len)
        last_page_len = batch.b_seq_len - (b_page_len - 1) * self.infer_page_size
        if plan.kv_starts is None:
            plan.kv_starts, plan.kv_last_page_len_buffer = starts, last_page_len
            plan.kv_indices = plan.workspace.get(
                "flashinfer_indices", max(1, batch.batch_size * self.max_page_num), torch.int32
            )
        else:
            plan.kv_starts.copy_(starts, non_blocking=True)
            plan.kv_last_page_len_buffer.copy_(last_page_len, non_blocking=True)
        repack_kv_index(
            req_to_token_indexs=batch.req_to_token_indexs,
            b_req_idx=batch.b_req_idx,
            b_token_len=batch.b_seq_len,
            b_page_start_loc=plan.kv_starts[:-1],
            max_token_len=plan.current_max_kv_seq_len,
            out_page_indices=plan.kv_indices,
            page_size=self.infer_page_size,
        )
        if plan.decode_wrapper is None:
            plan.decode_wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
                plan.workspace.get(self.workspace_buffer_key, self.workspace_buffer_size),
                "NHD",
                use_cuda_graph=plan.capture,
                use_tensor_cores=True,
                paged_kv_indptr_buffer=plan.kv_starts,
                paged_kv_indices_buffer=plan.kv_indices,
                paged_kv_last_page_len_buffer=plan.kv_last_page_len_buffer,
            )
            plan.decode_wrapper.plan(
                plan.kv_starts,
                plan.kv_indices,
                plan.kv_last_page_len_buffer,
                self.tp_q_head_num,
                self.tp_kv_head_num,
                self.head_dim,
                self.infer_page_size,
                q_data_type=self.q_data_type,
                kv_data_type=self.kv_data_type,
                non_blocking=True,
            )
        else:
            self._decode_refresh_cuda_graph_decode_plan(plan, plan.current_max_kv_seq_len)
