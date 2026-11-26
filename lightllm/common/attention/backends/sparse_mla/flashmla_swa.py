"""FlashMLA 在滑动窗口缓存上执行稀疏注意力，并可选地使用压缩缓存（DeepSeek V4）。"""
from collections.abc import Callable

from lightllm.common.basemodel.infer_struct import InferStateInfo

from dataclasses import dataclass

import torch

from lightllm.common.attention.base_backend import PrefillAttBackend, DecodeAttBackend
from . import SWASparseMLAAttConfig, SWASparseMLAAttInputs


@dataclass(frozen=True)
class FlashMLASWAAttState:
    """每个实例服务一个压缩比。Decode 额外为 CUDA Graph 捕获准备一份独立的调度元数据。"""

    sched_meta: object
    capture_sched_meta: object | None = None


class FlashMLASWAPrefillBackend(PrefillAttBackend[SWASparseMLAAttConfig, FlashMLASWAAttState, SWASparseMLAAttInputs]):
    def create_att_state(self, infer_state: InferStateInfo) -> FlashMLASWAAttState:
        from flash_mla import get_mla_metadata

        return FlashMLASWAAttState(sched_meta=get_mla_metadata()[0])

    def prefill(
        self,
        att_state: FlashMLASWAAttState,
        layer_inputs: SWASparseMLAAttInputs,
        *,
        out: torch.Tensor | None,
        alloc_func: Callable[..., torch.Tensor] = torch.empty,
    ):
        return FlashMLASWAPrefillBackend._attention(
            att_config=self.att_config, att_state=att_state, layer_inputs=layer_inputs, out=out
        )

    @staticmethod
    def get_name() -> str:
        return "flashmla_swa_sparse_mla_prefill"

    @staticmethod
    def _attention(
        att_config: SWASparseMLAAttConfig,
        att_state: FlashMLASWAAttState,
        layer_inputs: SWASparseMLAAttInputs,
        *,
        out: torch.Tensor | None,
    ):
        """填充并返回 [行数, num_heads, head_dim_v]，由 FlashMLA 就地写入。"""
        import flash_mla

        assert layer_inputs.q.shape[1] == att_config.num_heads, "q does not match the att_config"
        assert (layer_inputs.compressed_cache is not None) == (att_config.compress_ratio != 0)
        assert (layer_inputs.attn_sink is not None) == att_config.has_attn_sink
        sched_meta = att_state.sched_meta
        if att_state.capture_sched_meta is not None and torch.cuda.is_current_stream_capturing():
            sched_meta = att_state.capture_sched_meta
        flash_mla.flash_mla_with_kvcache(
            q=layer_inputs.q.unsqueeze(1),
            k_cache=layer_inputs.swa_cache,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=att_config.head_dim_v,
            tile_scheduler_metadata=sched_meta,
            num_splits=None,
            softmax_scale=att_config.softmax_scale,
            causal=False,
            is_fp8_kvcache=True,
            indices=layer_inputs.swa_indices,
            attn_sink=layer_inputs.attn_sink,
            topk_length=layer_inputs.swa_lengths,
            extra_k_cache=layer_inputs.compressed_cache,
            extra_indices_in_kvcache=layer_inputs.extra_indices,
            extra_topk_length=layer_inputs.extra_lengths,
            out=out.unsqueeze(1),
        )
        return out


class FlashMLASWADecodeBackend(DecodeAttBackend[SWASparseMLAAttConfig, FlashMLASWAAttState, SWASparseMLAAttInputs]):
    def create_att_state(self, infer_state: InferStateInfo) -> FlashMLASWAAttState:
        from flash_mla import get_mla_metadata

        # 捕获使用独立的元数据，让规划内核在每次 Graph 重放时运行。
        return FlashMLASWAAttState(sched_meta=get_mla_metadata()[0], capture_sched_meta=get_mla_metadata()[0])

    def decode(
        self,
        att_state: FlashMLASWAAttState,
        layer_inputs: SWASparseMLAAttInputs,
        *,
        out: torch.Tensor | None,
        alloc_func: Callable[..., torch.Tensor] = torch.empty,
    ):
        return FlashMLASWAPrefillBackend._attention(
            att_config=self.att_config, att_state=att_state, layer_inputs=layer_inputs, out=out
        )

    @staticmethod
    def get_name() -> str:
        return "flashmla_swa_sparse_mla_decode"
