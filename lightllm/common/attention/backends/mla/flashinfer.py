from collections.abc import Callable
from lightllm.common.basemodel.infer_struct import GraphMode, InferStateInfo
from functools import cached_property
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch
import triton


from lightllm.common.attention.base_backend import AttRuntimeConfig, DecodeAttBackend, PrefillAttBackend
from lightllm.utils.dist_utils import get_current_device_id
from ..flashinfer_env import set_flashinfer_envs
from . import MLADecodeAttConfig, MLADecodeAttInputs, MLAPrefillAttConfig, MLAPrefillAttInputs

if TYPE_CHECKING:
    from flashinfer.mla import BatchMLAPagedAttentionWrapper
    from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper
    from lightllm.common.basemodel.basemodel import TpPartBaseModel


@dataclass(frozen=True)
class FlashInferMLAPrefillAttState:
    """步骤的扩展键值上计划的因果稀疏 KV 封装对象。"""

    wrapper: "BatchPrefillWithRaggedKVCacheWrapper"


@dataclass(frozen=True)
class FlashInferMLADecodeAttState:
    """步骤的扩展键值上计划的分页 MLA 封装对象，每个请求一行查询。"""

    wrapper: "BatchMLAPagedAttentionWrapper | None"
    kv_starts: torch.Tensor
    kv_indices: torch.Tensor
    q_indptr: torch.Tensor


class FlashInferMLAPrefillBackend(
    PrefillAttBackend[MLAPrefillAttConfig, FlashInferMLAPrefillAttState, MLAPrefillAttInputs]
):
    def __init__(
        self,
        att_config: MLAPrefillAttConfig,
        *,
        role: str = "main",
        runtime_config: AttRuntimeConfig = AttRuntimeConfig(),
    ) -> None:
        super().__init__(att_config=att_config, role=role, runtime_config=runtime_config)
        if not att_config.causal:
            raise NotImplementedError("FlashInfer MLA prefill supports only causal attention")

    def prefill(
        self,
        att_state: FlashInferMLAPrefillAttState,
        layer_inputs: MLAPrefillAttInputs,
        *,
        out: torch.Tensor,
        alloc_func: Callable[..., torch.Tensor] = torch.empty,
    ) -> torch.Tensor:
        """填充并返回 out [query_tokens, num_heads, v_head_dim]；keys 扩展到 alloc_func 临时缓冲区。"""
        att_config = self.att_config
        q, k_nope = layer_inputs.q, layer_inputs.k_nope
        nope_dim, heads = att_config.qk_nope_head_dim, att_config.num_heads
        k = alloc_func((k_nope.shape[0], heads, q.shape[-1]), dtype=q.dtype, device=q.device)
        k[:, :, :nope_dim].copy_(k_nope)
        k[:, :, nope_dim:].copy_(layer_inputs.k_rope.expand(-1, heads, -1))
        att_state.wrapper.run(q=q, k=k, v=layer_inputs.v, out=out)
        return out

    @staticmethod
    def get_name() -> str:
        return "flashinfer_mla_prefill"

    def create_att_state(self, infer_state: InferStateInfo) -> FlashInferMLAPrefillAttState:
        """为步骤的 query 和 kv 偏移量规划一个新的封装对象。"""
        config = self.att_config
        wrapper = self._wrapper_class(
            float_workspace_buffer=self.get_workspace(name="flashinfer_mla", role=self.role, numel=_WORKSPACE_BYTES),
            kv_layout="NHD",
        )
        wrapper.plan(
            qo_indptr=infer_state.b1_cu_q_seq_len.int(),
            kv_indptr=infer_state.b1_cu_kv_seq_len.int(),
            num_qo_heads=config.num_heads,
            num_kv_heads=config.num_heads,
            head_dim_qk=config.qk_nope_head_dim + config.qk_rope_head_dim,
            head_dim_vo=config.v_head_dim,
            q_data_type=config.dtype,
            causal=True,
            sm_scale=config.softmax_scale,
        )
        return FlashInferMLAPrefillAttState(wrapper=wrapper)

    @cached_property
    def _wrapper_class(self):
        set_flashinfer_envs()
        import flashinfer

        return flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper


class FlashInferMLADecodeBackend(DecodeAttBackend[MLADecodeAttConfig, FlashInferMLADecodeAttState, MLADecodeAttInputs]):
    @dataclass(frozen=True)
    class RuntimeConfig(AttRuntimeConfig):
        page_size: int = 1
        max_seq_len: int = 0
        max_batch_size: int = 0
        device: str | int = "cuda"

    @classmethod
    def create_runtime_config(cls, model: "TpPartBaseModel") -> RuntimeConfig:
        return replace(
            cls.RuntimeConfig.from_model(model=model),
            page_size=model.args.page_size,
            max_seq_len=model.max_seq_length,
            max_batch_size=model.graph_max_batch_size,
            device=get_current_device_id(),
        )

    def __init__(
        self,
        att_config: MLADecodeAttConfig,
        *,
        role: str = "main",
        runtime_config: RuntimeConfig = RuntimeConfig(),
    ) -> None:
        super().__init__(att_config=att_config, role=role, runtime_config=runtime_config)
        # 每个 query 行读取其自身的请求前缀，该前缀在请求的行之间具有因果关系。
        if not att_config.causal:
            raise NotImplementedError("FlashInfer MLA decode supports only causal attention")

    def decode(
        self,
        att_state: FlashInferMLADecodeAttState,
        layer_inputs: MLADecodeAttInputs,
        *,
        out: torch.Tensor,
        alloc_func: Callable[..., torch.Tensor] = torch.empty,
    ) -> torch.Tensor:
        """填充并返回 out，其形状与 q_nope 相同；规划的工作区保留临时缓冲区。"""
        att_config = self.att_config
        kv, rank, page_size = layer_inputs.kv_cache, att_config.kv_lora_rank, self._page_size
        att_state.wrapper.run(
            q_nope=layer_inputs.q_nope,
            q_pe=layer_inputs.q_rope,
            ckv_cache=kv[:, :, :rank].view(-1, page_size, 1, rank),
            kpe_cache=kv[:, :, rank:].view(-1, page_size, 1, att_config.qk_rope_head_dim),
            out=out,
            return_lse=False,
        )
        return out

    @staticmethod
    def get_name() -> str:
        return "flashinfer_mla_decode"

    def create_att_state(self, infer_state: InferStateInfo) -> FlashInferMLADecodeAttState:
        """规划步骤的封装对象；每个 query 行都是其自身的请求。

        捕获状态持有 wrapper、页偏移和索引。重放步骤只生成动态页参数，
        复制到捕获状态后再刷新 wrapper 的计划表。
        """
        config = self.att_config
        page_size = self._page_size
        import flashinfer
        from lightllm.common.basemodel.triton_kernel.gen_prefill_params import gen_cumsum_pad0_tensor
        from lightllm.common.basemodel.triton_kernel.repack_kv_index import repack_kv_index

        device = infer_state.b_seq_len.device
        # 部分最终页仍然算作一页。
        b_page_len = (infer_state.b_seq_len + (page_size - 1)) // page_size
        kv_starts, _ = gen_cumsum_pad0_tensor(b_page_len, b_page_len)
        page_ids = infer_state.batch_size * triton.cdiv(x=self.runtime_config.max_seq_len, y=page_size)
        if infer_state.batch_size <= self.runtime_config.max_batch_size:
            kv_indices = self._kv_indices[infer_state.microbatch_index][:page_ids]
        else:
            # 捕获的封装对象必须读取固定的微批次缓冲区。
            assert not infer_state.uses_cuda_graph, "CUDA graph batches must fit the page-index buffers"
            kv_indices = torch.empty(page_ids, dtype=torch.int32, device=device)
        offsets = FlashInferMLADecodeBackend.create_att_state.__dict__.setdefault("_query_offsets", {})
        offset_key = (device, infer_state.batch_size)
        q_indptr = offsets.get(offset_key)
        if q_indptr is None:
            q_indptr = offsets.setdefault(
                offset_key, torch.arange(end=infer_state.batch_size + 1, dtype=torch.int32, device=device)
            )
        repack_kv_index(
            req_to_token_indexs=infer_state.req_to_token_indexs,
            b_req_idx=infer_state.b_req_idx,
            b_token_len=infer_state.b_seq_len,
            b_page_start_loc=kv_starts[:-1],
            max_token_len=infer_state.max_kv_seq_len,
            out_page_indices=kv_indices,
            page_size=page_size,
        )
        if infer_state.graph_mode is GraphMode.REPLAY:
            return FlashInferMLADecodeAttState(
                wrapper=None, kv_starts=kv_starts, kv_indices=kv_indices, q_indptr=q_indptr
            )
        wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
            float_workspace_buffer=self.get_workspace(name="flashinfer_mla", role=self.role, numel=_WORKSPACE_BYTES),
            use_cuda_graph=True,
            qo_indptr=q_indptr,
            kv_indices=kv_indices,
            kv_indptr=kv_starts,
            kv_len_arr=infer_state.b_seq_len,
        )
        wrapper.plan(
            qo_indptr=q_indptr,
            kv_indptr=kv_starts,
            kv_indices=kv_indices,
            kv_len_arr=infer_state.b_seq_len,
            num_heads=config.num_heads,
            head_dim_ckv=config.kv_lora_rank,
            head_dim_kpe=config.qk_rope_head_dim,
            page_size=page_size,
            # 每一行都是一个独立的请求，因此一个请求内行与行之间的因果关系来自于长度信息。
            causal=False,
            sm_scale=config.softmax_scale,
            q_data_type=config.dtype,
            kv_data_type=config.dtype,
        )
        return FlashInferMLADecodeAttState(
            wrapper=wrapper, kv_starts=kv_starts, kv_indices=kv_indices, q_indptr=q_indptr
        )

    def refresh_cuda_graph_state(self, att_state: FlashInferMLADecodeAttState, infer_state: InferStateInfo) -> None:
        """页偏移复制到捕获状态后，刷新其 wrapper 的 GPU 调度计划。"""
        from lightllm.common.basemodel.triton_kernel.flashinfer_mla_plan import fill_mla_decode_plan_for_cuda_graph

        fill_mla_decode_plan_for_cuda_graph(
            decode_wrapper=att_state.wrapper,
            kv_page_indptr=att_state.kv_starts,
            kv_lens=infer_state.b_seq_len,
            batch_size=infer_state.batch_size,
            num_heads=self.att_config.num_heads,
            max_kv_len=infer_state.max_kv_seq_len,
        )

    @cached_property
    def _page_size(self) -> int:
        if self.runtime_config.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive before preparing FlashInfer state")
        set_flashinfer_envs()
        return self.runtime_config.page_size

    @cached_property
    def _kv_indices(self) -> list[torch.Tensor]:
        config = self.runtime_config
        pages_per_batch = config.max_batch_size * triton.cdiv(x=config.max_seq_len, y=config.page_size)
        return [torch.empty(size=(pages_per_batch,), dtype=torch.int32, device=config.device) for _ in range(2)]


# FlashInfer MLA 的工作区具有固定大小，与 GQA 的工作区分开（--flashinfer_workspace_size）。
_WORKSPACE_BYTES = 256 * 1024 * 1024
