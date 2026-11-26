"""FlashMLA 稀疏 MLA：在 bf16 或打包的 FP8 缓存上进行 prefill，以及在打包的 FP8 缓存上进行 decode。"""
from collections.abc import Callable

from lightllm.common.basemodel.infer_struct import InferStateInfo

from dataclasses import dataclass
from functools import lru_cache
from inspect import signature
from typing import Optional

import torch

from lightllm.common.attention.base_backend import AttRuntimeConfig, DecodeAttBackend, PrefillAttBackend
from . import SparseMLADecodeAttInputs, SparseMLAAttConfig, SparseMLAPrefillAttInputs


@dataclass(frozen=True)
class FlashMLASparsePrefillAttState:
    """mem_index 保存该步骤新生成的 token 的缓存槽；has_cached_prefix 表示任何请求是否从之前的步骤读取了键。"""

    mem_index: Optional[torch.Tensor]
    has_cached_prefix: bool


@dataclass(frozen=True)
class FlashMLASparseDecodeAttState:
    """FlashMLA 的 tile 调度器元数据，在该步骤首次内核调用时填充。

    direct_decode_bound 设置时，限制每个 top-k 行的有效条目，这使得在使用“fp8_group64_e8m0”缓存的小 CUDA Graph步骤中可以使用直接的 Triton 内核。
    """

    sched_meta: object
    uses_cuda_graph: bool = False
    direct_decode_bound: Optional[int] = None


class FlashMLASparsePrefillBackend(
    PrefillAttBackend[SparseMLAAttConfig, FlashMLASparsePrefillAttState, SparseMLAPrefillAttInputs]
):
    def __init__(
        self,
        att_config: SparseMLAAttConfig,
        *,
        role: str = "main",
        runtime_config: AttRuntimeConfig = AttRuntimeConfig(),
    ) -> None:
        super().__init__(att_config=att_config, role=role, runtime_config=runtime_config)
        if att_config.kv_cache_format not in ("bf16", "fp8_ds_mla"):
            raise NotImplementedError(f"FlashMLA sparse prefill does not support {att_config.kv_cache_format!r}")

    def prefill(
        self,
        att_state: FlashMLASparsePrefillAttState,
        layer_inputs: SparseMLAPrefillAttInputs,
        *,
        out: torch.Tensor,
        alloc_func: Callable[..., torch.Tensor] = torch.empty,
    ) -> torch.Tensor:
        """填充并返回 out [query_tokens, num_heads, kv_lora_rank]。

        打包的 FP8 缓存不直接读取：该步骤的键来自 current_kv，当某些请求具有已缓存前缀时，所选的历史记录将紧接它们进行反量化。
        """
        att_config, q = self.att_config, layer_inputs.q
        assert q.shape[1] == att_config.num_heads and q.dtype == att_config.dtype, "q does not match att_config"
        if att_config.kv_cache_format == "bf16":
            kv, indices = layer_inputs.kv_cache, layer_inputs.topk_mem_indices
            if indices.ndim == 2:
                indices = indices.unsqueeze(1)
            return self._bf16_prefill(q=q, kv=kv, indices=indices, out=out)

        from flash_mla import flash_mla_sparse_fwd

        assert layer_inputs.current_kv is not None and layer_inputs.topk_indices is not None
        kv, indices = layer_inputs.current_kv, layer_inputs.topk_indices
        if att_state.has_cached_prefix:
            from lightllm.common.basemodel.triton_kernel.kv_copy.prefill_compact_kv_flashmla_fp8 import (
                get_prefill_kv_cache_and_remap_indices_triton,
            )

            kv, indices = get_prefill_kv_cache_and_remap_indices_triton(
                packed_kv=layer_inputs.kv_cache,
                topk_mem_indices=layer_inputs.topk_mem_indices,
                prefill_mem_index=att_state.mem_index,
                prefill_cache_kv=layer_inputs.current_kv,
                prefill_dtype=layer_inputs.current_kv.dtype,
            )
        if indices.ndim == 2:
            indices = indices.unsqueeze(1)
        flash_mla_sparse_fwd(
            q=q, kv=kv, indices=indices, sm_scale=att_config.softmax_scale, d_v=att_config.kv_lora_rank, out=out
        )
        return out

    def create_att_state(self, infer_state: InferStateInfo) -> FlashMLASparsePrefillAttState:
        return FlashMLASparsePrefillAttState(
            mem_index=infer_state.mem_index, has_cached_prefix=(infer_state.max_cache_len or 0) > 0
        )

    @staticmethod
    def get_name() -> str:
        return "flashmla_sparse_mla_prefill"

    @staticmethod
    @lru_cache(maxsize=1)
    def _bf16_prefill_out_kernel():
        try:
            from flash_mla import flash_mla_sparse_fwd
        except ImportError:
            return None
        return flash_mla_sparse_fwd if "out" in signature(flash_mla_sparse_fwd).parameters else None

    def _bf16_prefill(self, q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, out: torch.Tensor | None):
        """直接使用调用者的输出，或者将旧内核的临时存储限制为 128 个查询行。"""
        if q.shape[0] == 0:
            return out
        softmax_scale = self.att_config.softmax_scale
        heads = q.shape[1]
        padded_heads = -(-heads // 64) * 64
        if heads == padded_heads and out.is_contiguous():
            kernel = self._bf16_prefill_out_kernel()
            if kernel is not None:
                kernel(q=q, kv=kv, indices=indices, sm_scale=softmax_scale, d_v=out.shape[2], out=out)
                return out

        from sgl_kernel.flash_mla import flash_mla_sparse_fwd

        chunk_size = 128
        for start in range(0, q.shape[0], chunk_size):
            rows = slice(start, start + chunk_size)
            query_chunk = q[rows]
            if heads != padded_heads:
                query_chunk = torch.nn.functional.pad(query_chunk, (0, 0, 0, padded_heads - heads))
            result = flash_mla_sparse_fwd(
                q=query_chunk, kv=kv, indices=indices[rows], sm_scale=softmax_scale, d_v=out.shape[2]
            )[0]
            out[rows].copy_(result[:, :heads])
            del result
        return out


class FlashMLASparseDecodeBackend(
    DecodeAttBackend[SparseMLAAttConfig, FlashMLASparseDecodeAttState, SparseMLADecodeAttInputs]
):
    def __init__(
        self,
        att_config: SparseMLAAttConfig,
        *,
        role: str = "main",
        runtime_config: AttRuntimeConfig = AttRuntimeConfig(),
    ) -> None:
        super().__init__(att_config=att_config, role=role, runtime_config=runtime_config)
        if att_config.kv_cache_format not in ("fp8_ds_mla", "fp8_group64_e8m0"):
            raise NotImplementedError("FlashMLA sparse decode supports only the packed FP8 caches")
        # 2048 个池化条目再加最多 3 个池尾条目，可使用直接解码内核。
        self.direct_decode_bound = 2051 if (att_config.topk, att_config.kpool) == (2048, 4) else None

    def decode(
        self,
        att_state: FlashMLASparseDecodeAttState,
        layer_inputs: SparseMLADecodeAttInputs,
        *,
        out: torch.Tensor,
        alloc_func: Callable[..., torch.Tensor] = torch.empty,
    ) -> torch.Tensor:
        """填充并返回 out，其形状与 q_nope 相同；每个请求有一个查询行。"""
        import flash_mla

        att_config = self.att_config
        q_nope = layer_inputs.q_nope
        heads = q_nope.shape[1]
        assert heads == att_config.num_heads and q_nope.dtype == att_config.dtype, "q does not match att_config"
        indices = layer_inputs.topk_mem_indices
        if indices.ndim == 2:
            indices = indices.unsqueeze(1)
        assert indices.shape[1] == 1, "FlashMLA sparse decode expects one query row per request"
        packed = layer_inputs.kv_cache
        if att_config.kv_cache_format == "fp8_group64_e8m0":
            from lightllm.common.attention.kernels.sparse_mla import fp8_group64_decode
            from lightllm.common.attention.kernels.sparse_mla.pad_query_heads import pad_query_heads

            if fp8_group64_decode.can_use_fp8_sparse_direct_decode(
                q_nope=q_nope,
                packed_kv=packed,
                topk=indices,
                kv_lora_rank=att_config.kv_lora_rank,
                bound=att_state.direct_decode_bound,
                is_cuda_graph_capture=att_state.uses_cuda_graph,
            ):
                fp8_group64_decode.fp8_sparse_direct_decode(
                    q_nope=q_nope, packed_kv=packed, topk=indices, softmax_scale=att_config.softmax_scale, out=out
                )
                return out
            q = pad_query_heads(q_nope, is_cuda_graph_capture=att_state.uses_cuda_graph)
        else:
            q = torch.cat(tensors=[q_nope, layer_inputs.q_rope], dim=-1).unsqueeze(1).contiguous()
        kv = torch.as_strided(
            packed,
            size=(packed.shape[0], 1, 1, packed.shape[-1]),
            stride=(packed.stride(0), packed.shape[-1], packed.shape[-1], packed.stride(-1)),
        )
        # FlashMLA 就地写入，除非需要对注意力头进行填充以适应其 64 个头的块。
        padded = q.shape[2] != heads
        result, _ = flash_mla.flash_mla_with_kvcache(
            q=q,
            k_cache=kv,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=att_config.kv_lora_rank,
            tile_scheduler_metadata=att_state.sched_meta,
            softmax_scale=att_config.softmax_scale,
            causal=False,
            is_fp8_kvcache=True,
            indices=indices,
            out=None if padded else out.unsqueeze(1),
        )
        if padded:
            out.copy_(result[:, 0, :heads])
        return out

    @staticmethod
    def get_name() -> str:
        return "flashmla_sparse_mla_decode"

    def create_att_state(self, infer_state: InferStateInfo) -> FlashMLASparseDecodeAttState:
        """调度对象由本步状态持有，捕获后随 CUDA Graph 一起保留。"""
        import flash_mla

        sched_meta = flash_mla.get_mla_metadata()[0]
        return FlashMLASparseDecodeAttState(
            sched_meta=sched_meta,
            uses_cuda_graph=infer_state.uses_cuda_graph,
            direct_decode_bound=self.direct_decode_bound,
        )
