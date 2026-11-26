"""FA3 MLA。解码与 FA3 GQA 使用相同的行分组与页表填充函数。"""
from collections.abc import Callable

from lightllm.common.basemodel.infer_struct import InferStateInfo

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from lightllm.common.attention.base_backend import DecodeAttBackend, PrefillAttBackend
from ..gqa.fa3 import FA3DecodeRuntimeConfig, FA3GQADecodeAttState, FA3GQADecodeBackend, FA3GQAPrefillBackend
from . import MLADecodeAttConfig, MLADecodeAttInputs, MLAPrefillAttConfig, MLAPrefillAttInputs

if TYPE_CHECKING:
    from lightllm.common.basemodel.basemodel import TpPartBaseModel


@dataclass(frozen=True)
class FA3MLAPrefillAttState:
    """int32 累积查询和 kv 偏移量，用于步骤的打包和扩展键值。"""

    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int


class FA3MLAPrefillBackend(PrefillAttBackend[MLAPrefillAttConfig, FA3MLAPrefillAttState, MLAPrefillAttInputs]):
    def prefill(
        self,
        att_state: FA3MLAPrefillAttState,
        layer_inputs: MLAPrefillAttInputs,
        *,
        out: torch.Tensor,
        alloc_func: Callable[..., torch.Tensor] = torch.empty,
    ) -> torch.Tensor:
        """填充并返回 out [query_tokens, num_heads, v_head_dim]；键扩展到 alloc_func 临时缓冲区。

        FA3 将因果掩码对齐到每个请求的最后一个键，因此已缓存前缀保持可见。
        """
        from lightllm.utils.sgl_utils import flash_attn_varlen_func

        att_config = self.att_config
        q, k_nope = layer_inputs.q, layer_inputs.k_nope
        nope_dim, heads = att_config.qk_nope_head_dim, att_config.num_heads
        k = alloc_func((k_nope.shape[0], heads, q.shape[-1]), dtype=q.dtype, device=q.device)
        k[:, :, :nope_dim].copy_(k_nope)
        k[:, :, nope_dim:].copy_(layer_inputs.k_rope.expand(-1, heads, -1))
        flash_attn_varlen_func(
            q=q,
            k=k,
            v=layer_inputs.v,
            cu_seqlens_q=att_state.cu_seqlens_q,
            cu_seqlens_k=att_state.cu_seqlens_k,
            max_seqlen_q=att_state.max_seqlen_q,
            max_seqlen_k=att_state.max_seqlen_k,
            softmax_scale=att_config.softmax_scale,
            causal=att_config.causal,
            return_softmax_lse=False,
            out=out,
        )
        return out

    @staticmethod
    def get_name() -> str:
        return "fa3_mla_prefill"

    def create_att_state(self, infer_state: InferStateInfo) -> FA3MLAPrefillAttState:
        assert (
            infer_state.b1_cu_q_seq_len is not None and infer_state.b1_cu_kv_seq_len is not None
        ), "prefill metadata fields required"
        return FA3MLAPrefillAttState(
            cu_seqlens_q=infer_state.b1_cu_q_seq_len.int(),
            cu_seqlens_k=infer_state.b1_cu_kv_seq_len.int(),
            max_seqlen_q=infer_state.max_q_seq_len,
            max_seqlen_k=infer_state.max_kv_seq_len,
        )

    @classmethod
    def requires_cpu_cache_sync(cls) -> bool:
        return FA3GQAPrefillBackend.fa3_requires_cpu_cache_sync()


class FA3MLADecodeBackend(DecodeAttBackend[MLADecodeAttConfig, FA3GQADecodeAttState, MLADecodeAttInputs]):
    @classmethod
    def create_runtime_config(cls, model: "TpPartBaseModel") -> FA3DecodeRuntimeConfig:
        return FA3DecodeRuntimeConfig.from_model(model=model)

    def __init__(
        self,
        att_config: MLADecodeAttConfig,
        *,
        role: str = "main",
        runtime_config: FA3DecodeRuntimeConfig = FA3DecodeRuntimeConfig(),
    ) -> None:
        super().__init__(att_config=att_config, role=role, runtime_config=runtime_config)
        self._page_tables: list[torch.Tensor] = []

    def decode(
        self,
        att_state: FA3GQADecodeAttState,
        layer_inputs: MLADecodeAttInputs,
        *,
        out: torch.Tensor,
        alloc_func: Callable[..., torch.Tensor] = torch.empty,
    ) -> torch.Tensor:
        """填充并返回 out，其形状与 q_nope 相同。旋转变量部分是 FA3 的 q/k，潜在部分是其 qv/v。"""
        from lightllm.utils.sgl_utils import flash_attn_with_kvcache

        att_config = self.att_config
        kv, rank, page_size = layer_inputs.kv_cache, att_config.kv_lora_rank, self.runtime_config.page_size
        flash_attn_with_kvcache(
            q=layer_inputs.q_rope,
            k_cache=kv[:, :, rank:].view(-1, page_size, 1, att_config.qk_rope_head_dim),
            v_cache=kv[:, :, :rank].view(-1, page_size, 1, rank),
            qv=layer_inputs.q_nope,
            page_table=att_state.page_table,
            cache_seqlens=att_state.cache_seqlens,
            cu_seqlens_q=att_state.cu_seqlens_q,
            cu_seqlens_k_new=att_state.cu_seqlens_k,
            max_seqlen_q=att_state.max_seqlen_q,
            softmax_scale=att_config.softmax_scale,
            causal=att_config.causal,
            window_size=(-1, -1),
            softcap=0.0,
            k_descale=None,
            v_descale=None,
            return_softmax_lse=False,
            out=out,
        )
        return out

    @staticmethod
    def get_name() -> str:
        return "fa3_mla_decode"

    def create_att_state(self, infer_state: InferStateInfo) -> FA3GQADecodeAttState:
        return FA3GQADecodeBackend.prepare_fa3_decode_state(
            batch=infer_state,
            config=self.runtime_config,
            page_size=self.runtime_config.page_size,
            page_tables=self._page_tables,
        )

    @classmethod
    def requires_cpu_cache_sync(cls) -> bool:
        return FA3GQAPrefillBackend.fa3_requires_cpu_cache_sync()
