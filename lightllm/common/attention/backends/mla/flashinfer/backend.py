import dataclasses
from typing import Tuple

import torch
import triton

from lightllm.common.attention.api import BackendResources
from lightllm.common.attention.api.backend import LayerAttentionBinding
from lightllm.common.attention.api.inputs import MLADecodeInputs, MLAPrefillInputs
from lightllm.common.attention.api.spec import AttentionFamily
from lightllm.common.attention.backends._flashinfer import set_flashinfer_envs
from lightllm.common.attention.backends.base import AttentionExecutionContext, BaseAttentionBackend
from lightllm.common.basemodel.triton_kernel.flashinfer_mla_plan import fill_mla_decode_plan_for_cuda_graph
from lightllm.common.basemodel.triton_kernel.gen_prefill_params import gen_cumsum_pad0_tensor
from lightllm.common.basemodel.triton_kernel.repack_kv_index import repack_kv_index
from lightllm.utils.dist_utils import get_dp_world_size


@dataclasses.dataclass
class MlaFlashInferPrefillContext(AttentionExecutionContext):
    prefill_wrapper: object = None


@dataclasses.dataclass
class MlaFlashInferDecodeContext(AttentionExecutionContext):
    kv_indices: torch.Tensor = None
    kv_starts: torch.Tensor = None
    q_indptr: torch.Tensor = None
    decode_wrapper: object = None


class MlaFlashInferAttBackend(BaseAttentionBackend):
    provider = "flashinfer"
    dependency = "flashinfer"
    activation_dtypes = frozenset({torch.float16, torch.bfloat16})
    supports_expert_parallel = False
    cache_formats = frozenset({"None"})
    family = AttentionFamily.MLA
    workspace_buffer_key = "flashinfer_mla"
    workspace_buffer_size = 256 * 1024 * 1024

    def __init__(self, model):
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
        from lightllm.models.llama.yarn_rotary_utils import get_deepseek_mscale

        if model.config["rope_scaling"] is not None:
            rope_scaling = model.config["rope_scaling"]
            mscale_all_dim = rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = rope_scaling["factor"]
            if mscale_all_dim:
                mscale = get_deepseek_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale
        return

    def for_spec(self, spec):
        backend = super().for_spec(spec)
        backend.tp_q_head_num = spec.num_q_heads
        return backend

    def resource_requirements(self, capacity, *, capture):
        return BackendResources(((self.workspace_buffer_key, self.workspace_buffer_size, torch.uint8),))

    def _init_infer_page_size(self):
        self.infer_page_size = self.page_size
        assert (
            self.page_size % self.infer_page_size == 0
        ), f"model page_size {self.page_size} must be divisible by infer_page_size {self.infer_page_size}"

    def _allocate_prefill_context(self, capacity, capture, workspace):
        return MlaFlashInferPrefillContext(capacity, capture, workspace)

    def _allocate_decode_context(self, capacity, capture, workspace):
        return MlaFlashInferDecodeContext(capacity, capture, workspace)

    def _prepare_prefill(self, context):
        import flashinfer

        q_starts = context.batch.b1_cu_q_seq_len.int()
        kv_starts = context.batch.b1_cu_kv_seq_len.int()
        if context.prefill_wrapper is None:
            context.prefill_wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
                context.workspace.get(self.workspace_buffer_key, self.workspace_buffer_size), "NHD"
            )
        context.prefill_wrapper.plan(
            qo_indptr=q_starts,
            kv_indptr=kv_starts,
            num_qo_heads=self.tp_q_head_num,
            num_kv_heads=self.tp_q_head_num,
            head_dim_qk=self.qk_nope_head_dim + self.qk_rope_head_dim,
            head_dim_vo=self.v_head_dim,
            q_data_type=self.q_data_type,
            causal=True,
            sm_scale=self.softmax_scale,
        )
        return

    def _forward_prefill(
        self, layer: LayerAttentionBinding, context, inputs: MLAPrefillInputs, *, out=None
    ) -> torch.Tensor:
        alloc_func = layer.alloc_tensor
        q = inputs.query
        k = (inputs.key_nope, inputs.key_rope)
        v = inputs.value
        return self._prefill_mla_prefill_att(layer, context, q=q, k=k, v=v, alloc_func=alloc_func)

    def _prefill_mla_prefill_att(
        self,
        layer,
        context,
        q: torch.Tensor,
        k: Tuple[torch.Tensor, torch.Tensor],
        v: torch.Tensor,
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        k_nope, k_rope = k
        o_tensor = alloc_func((q.shape[0], q.shape[1], v.shape[-1]), q.dtype, device="cuda")
        q_head_num = q.shape[1]
        k = torch.cat([k_nope, torch.repeat_interleave(k_rope, q_head_num, dim=-2)], dim=-1)
        context.prefill_wrapper.run(q, k, v, out=o_tensor)
        return o_tensor

    def _decode_refresh_cuda_graph_decode_plan(self, context, kv_lens: torch.Tensor, max_kv_len: int):
        fill_mla_decode_plan_for_cuda_graph(
            decode_wrapper=context.decode_wrapper,
            kv_page_indptr=context.kv_starts,
            kv_lens=kv_lens,
            batch_size=context.batch.batch_size,
            num_heads=self.tp_q_head_num,
            max_kv_len=max_kv_len,
        )

    def _forward_decode(self, layer: LayerAttentionBinding, context, inputs: MLADecodeInputs, *, out=None):
        alloc_func = layer.alloc_tensor
        q = (inputs.query_nope, inputs.query_rope)
        k = layer.params.kv
        v = None
        assert v is None
        return self._decode_mla_decode_att(layer, context, q=q, k=k, v=v, inputs=inputs, alloc_func=alloc_func)

    def _decode_mla_decode_att(
        self,
        layer,
        context,
        q: Tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        v: torch.Tensor,
        inputs,
        alloc_func=torch.empty,
    ):
        qk_rope_head_dim = layer.params.qk_rope_head_dim
        q_nope, q_rope = q
        o_tensor = alloc_func(q_nope.shape, dtype=q_nope.dtype, device=q_nope.device)
        context.decode_wrapper.run(
            q_nope,
            q_rope,
            k[:, :, :-qk_rope_head_dim].view(-1, self.infer_page_size, 1, k.shape[-1] - qk_rope_head_dim),
            k[:, :, -qk_rope_head_dim:].view(-1, self.infer_page_size, 1, qk_rope_head_dim),
            out=o_tensor,
            return_lse=False,
        )
        return o_tensor

    def _prepare_decode(self, context):
        import flashinfer

        batch = context.batch
        b_page_len = (batch.b_seq_len + self.infer_page_size - 1) // self.infer_page_size
        starts, _ = gen_cumsum_pad0_tensor(b_page_len, b_page_len)
        if context.kv_starts is None:
            context.kv_starts = starts
            context.kv_indices = context.workspace.get(
                "flashinfer_indices", max(1, batch.batch_size * self.max_page_num), torch.int32
            )
            context.q_indptr = torch.arange(batch.batch_size + 1, dtype=torch.int32, device=batch.device)
        else:
            context.kv_starts.copy_(starts, non_blocking=True)
        repack_kv_index(
            req_to_token_indexs=batch.req_to_token_indexs,
            b_req_idx=batch.b_req_idx,
            b_token_len=batch.b_seq_len,
            b_page_start_loc=context.kv_starts[:-1],
            max_token_len=context.current_max_kv_seq_len,
            out_page_indices=context.kv_indices,
            page_size=self.infer_page_size,
        )
        if context.decode_wrapper is None:
            context.decode_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
                context.workspace.get(self.workspace_buffer_key, self.workspace_buffer_size),
                use_cuda_graph=context.capture,
                qo_indptr=context.q_indptr,
                kv_indices=context.kv_indices,
                kv_indptr=context.kv_starts,
                kv_len_arr=batch.b_seq_len,
            )
            context.decode_wrapper.plan(
                context.q_indptr,
                context.kv_starts,
                context.kv_indices,
                batch.b_seq_len,
                self.tp_q_head_num,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                self.infer_page_size,
                False,
                self.softmax_scale,
                self.q_data_type,
                self.kv_data_type,
            )
        else:
            self._decode_refresh_cuda_graph_decode_plan(context, batch.b_seq_len, context.current_max_kv_seq_len)
