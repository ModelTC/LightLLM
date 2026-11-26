from collections.abc import Callable
from lightllm.common.basemodel.infer_struct import InferStateInfo
from functools import cached_property
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Optional

import torch
import triton

from lightllm.common.attention.base_backend import AttRuntimeConfig, DecodeAttBackend, PrefillAttBackend
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.utils.envs_utils import enable_env_vars
from lightllm.utils.log_utils import init_logger
from . import GQAAttConfig, GQAAttInputs

if TYPE_CHECKING:
    from lightllm.common.basemodel.basemodel import TpPartBaseModel

logger = init_logger(__name__)


@dataclass(frozen=True)
class FA3GQAPrefillAttState:
    """分页缓存的元数据，用于打包的 prefill 查询，每条请求一行页表。

    page_table[i, j] 是存储请求 i 的标记 [j * page_size, (j + 1) * page_size) 的物理页。cu_seqlens_q/cu_seqlens_k 是 int32
    累积查询和 KV 偏移量，将每个请求的新键放置在分页缓存中。
    """

    page_table: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    # 每个query token：其图像跨度的独占结束，文本为 0。
    b_image_token_end: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class FA3GQADecodeAttState:
    """分页缓存的元数据，用于 decode，每条注意力请求一行页表。

    在投机解码中，一个注意力请求将一个逻辑请求的查询行分组；cu_seqlens_q 给出 int32 查询偏移量，cache_seqlens 给出每个请求的 KV 长度。max_seqlen_k 是此步骤的实际最大值，
    仅作为内核配置提示。
    """

    page_table: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int


class FA3GQAPrefillBackend(PrefillAttBackend[GQAAttConfig, FA3GQAPrefillAttState, GQAAttInputs]):
    @dataclass(frozen=True)
    class RuntimeConfig(AttRuntimeConfig):
        page_size: int = 1

    @classmethod
    def create_runtime_config(cls, model: "TpPartBaseModel") -> RuntimeConfig:
        return replace(cls.RuntimeConfig.from_model(model=model), page_size=model.args.page_size)

    def __init__(
        self,
        att_config: GQAAttConfig,
        *,
        role: str = "main",
        runtime_config: RuntimeConfig = RuntimeConfig(),
    ) -> None:
        """支持任何缩放、因果或非因果注意力、单侧或双侧窗口和接收器。

        每头 FP8 缓存（kv_quant='fp8_per_head'）支持默认缩放，无需窗口或
        接收器；q 用静态 q 缩放或动态地按请求量化。
        """
        super().__init__(att_config=att_config, role=role, runtime_config=runtime_config)
        FA3GQAPrefillBackend._check_supported(att_config=att_config)
        if att_config.bidirectional_image_spans:
            from lightllm.utils.neo_fa3_utils import HAS_FLASH_ATTN_INTERFACE, NEO_FA3_INSTALL_HINT

            if not HAS_FLASH_ATTN_INTERFACE:
                raise NotImplementedError(
                    "FA3 bidirectional image spans need FA3 with image_token_end support.\n" + NEO_FA3_INSTALL_HINT
                )

    def prefill(
        self,
        att_state: FA3GQAPrefillAttState,
        layer_inputs: GQAAttInputs,
        *,
        out: torch.Tensor,
        alloc_func: Callable[..., torch.Tensor] = torch.empty,
    ) -> torch.Tensor:
        """用 q 的形状、dtype 和设备填充并返回 out。

        缓存槽形成完整的页面，当前查询的 KV 已在缓存中。
        Q/K/V 为只读；out 不得与它们重叠。FA3 管理自己的临时缓冲区，因此前向传播
        不调用 alloc_func。
        """
        from lightllm.utils.sgl_utils import flash_attn_with_kvcache

        att_config = self.att_config
        FA3GQAPrefillBackend._check_inputs(att_config=att_config, layer_inputs=layer_inputs)
        if att_config.bidirectional_image_spans:
            from lightllm.utils.neo_fa3_utils import flash_attn_with_kvcache_neo

            assert att_state.b_image_token_end is not None, "bidirectional image spans need b_image_token_end"
            # Neo FA3 构建返回其自己的输出，因此将其复制到 out 中。
            out.copy_(
                flash_attn_with_kvcache_neo(
                    q=layer_inputs.q,
                    k_cache=FA3GQAPrefillBackend._paged(cache=layer_inputs.k_cache, page_size=self._page_size),
                    v_cache=FA3GQAPrefillBackend._paged(cache=layer_inputs.v_cache, page_size=self._page_size),
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
                    image_token_end=att_state.b_image_token_end,
                )
            )
            return out
        if att_config.kv_quant is not None:
            q, k, v, q_descale, k_descale, v_descale = FA3GQAPrefillBackend._fp8_operands(
                att_config=att_config, layer_inputs=layer_inputs, att_state=att_state, page_size=self._page_size
            )
            flash_attn_with_kvcache(
                q=q,
                k_cache=k,
                v_cache=v,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                page_table=att_state.page_table,
                cache_seqlens=att_state.cache_seqlens,
                cu_seqlens_q=att_state.cu_seqlens_q,
                cu_seqlens_k_new=att_state.cu_seqlens_k,
                max_seqlen_q=att_state.max_seqlen_q,
                causal=att_config.causal,
                window_size=(-1, -1),
                softcap=0.0,
                return_softmax_lse=False,
                out=out,
            )
            return out
        flash_attn_with_kvcache(
            q=layer_inputs.q,
            k_cache=FA3GQAPrefillBackend._paged(cache=layer_inputs.k_cache, page_size=self._page_size),
            v_cache=FA3GQAPrefillBackend._paged(cache=layer_inputs.v_cache, page_size=self._page_size),
            page_table=att_state.page_table,
            cache_seqlens=att_state.cache_seqlens,
            cu_seqlens_q=att_state.cu_seqlens_q,
            cu_seqlens_k_new=att_state.cu_seqlens_k,
            max_seqlen_q=att_state.max_seqlen_q,
            softmax_scale=att_config.softmax_scale,
            causal=att_config.causal,
            window_size=att_config.sliding_window,
            softcap=0.0,
            return_softmax_lse=False,
            sinks=layer_inputs.sinks,
            out=out,
        )
        return out

    @staticmethod
    def get_name() -> str:
        return "fa3_gqa_prefill"

    def create_att_state(self, infer_state: InferStateInfo) -> FA3GQAPrefillAttState:
        """为该步骤的请求构建一个新的页表。"""
        from lightllm.common.basemodel.triton_kernel.fa3_utils import page_table_copy

        page_size = self._page_size

        page_table = torch.empty(
            size=(infer_state.batch_size, triton.cdiv(x=infer_state.max_kv_seq_len, y=page_size)),
            dtype=torch.int32,
            device=infer_state.b_seq_len.device,
        )
        page_table_copy(
            page_table=page_table,
            req_to_token_indexs=infer_state.req_to_token_indexs,
            b_req_idx=infer_state.b_req_idx,
            page_size=page_size,
        )
        return FA3GQAPrefillAttState(
            page_table=page_table,
            cache_seqlens=infer_state.b_seq_len,
            cu_seqlens_q=infer_state.b1_cu_q_seq_len.int(),
            cu_seqlens_k=infer_state.b1_cu_kv_seq_len.int(),
            max_seqlen_q=infer_state.max_q_seq_len,
            b_image_token_end=infer_state.b_image_token_end,
        )

    @classmethod
    def requires_cpu_cache_sync(cls) -> bool:
        return FA3GQAPrefillBackend.fa3_requires_cpu_cache_sync()

    @staticmethod
    def fa3_page_size(kv_quant: Optional[str], page_size: int) -> int:
        if kv_quant is None:
            return page_size
        logger.warning(f"FP8 FA3 temporarily uses page size 1 with model page_size={page_size}")
        return 1

    @staticmethod
    def fa3_requires_cpu_cache_sync() -> bool:
        # 设为 1 / TRUE / ON 时，FA3 与 CPU cache 搬运在 GPU 上互相等待。
        return enable_env_vars("LIGHTLLM_FA3_CPU_CACHE_SYNC")

    @cached_property
    def _page_size(self) -> int:
        return FA3GQAPrefillBackend.fa3_page_size(
            kv_quant=self.att_config.kv_quant, page_size=self.runtime_config.page_size
        )

    @staticmethod
    def _check_supported(att_config: GQAAttConfig) -> None:
        if att_config.dtype not in (torch.float16, torch.bfloat16) or att_config.head_dim > 256:
            raise NotImplementedError(
                f"FA3 GQA supports FP16/BF16 with head_dim <= 256, got {att_config.dtype}, {att_config.head_dim}"
            )
        if att_config.has_alibi:
            raise NotImplementedError("FA3 GQA does not support ALiBi")
        if att_config.bidirectional_image_spans and (
            att_config.kv_quant is not None or att_config.sliding_window != (-1, -1) or att_config.has_sinks
        ):
            raise NotImplementedError(
                "FA3 GQA supports bidirectional image spans without KV quantization, window or sinks"
            )
        if att_config.kv_quant not in (None, "fp8_per_head"):
            raise NotImplementedError(f"FA3 GQA supports kv_quant None or 'fp8_per_head', got {att_config.kv_quant!r}")
        if att_config.kv_quant is not None and (
            att_config.sliding_window != (-1, -1) or att_config.has_sinks or not att_config.has_default_scale
        ):
            raise NotImplementedError(
                "FA3 GQA with an FP8 cache supports only the default scale without window or sinks"
            )
        if att_config.kv_quant is not None and att_config.dtype != torch.bfloat16:
            # FA3 只将 FP8 注意力输出写为 BF16。
            raise NotImplementedError(f"FA3 GQA with an FP8 cache needs a BF16 model, got {att_config.dtype}")

    @staticmethod
    def _paged(cache: torch.Tensor, page_size: int) -> torch.Tensor:
        return cache.view(-1, page_size, cache.shape[1], cache.shape[2])

    @staticmethod
    def _check_inputs(att_config: GQAAttConfig, layer_inputs: GQAAttInputs) -> None:
        assert (
            layer_inputs.q.shape[1:] == (att_config.num_heads, att_config.head_dim)
            and layer_inputs.q.dtype == att_config.dtype
        )
        assert layer_inputs.k_cache.shape[1:] == (att_config.num_kv_heads, att_config.head_dim)
        assert (
            layer_inputs.sinks is not None
        ) == att_config.has_sinks, "sinks must be given exactly when the att_config has sinks"
        assert (layer_inputs.kv_scales is not None) == (
            att_config.kv_quant is not None
        ), "kv_scales must be given exactly when the att_config has kv_quant"
        if att_config.kv_quant is not None:
            assert (layer_inputs.kv_scales.q_scale is not None) == att_config.has_q_scale

    @staticmethod
    def _fp8_operands(
        att_config: GQAAttConfig,
        layer_inputs: GQAAttInputs,
        att_state: FA3GQAPrefillAttState | FA3GQADecodeAttState,
        page_size: int,
    ):
        """按 KV 头量化 q，返回 FA3 所需的 FP8 q、分页缓存及形状为 [requests, kv_heads] 的反量化 scale。"""
        from lightllm.common.basemodel.triton_kernel.quantization.q_per_head_fp8_quant import (
            q_per_head_fp8_quant,
            q_per_head_static_fp8_quant,
        )

        kv_heads, scales = att_config.num_kv_heads, layer_inputs.kv_scales
        requests = att_state.cache_seqlens.shape[0]
        q = layer_inputs.q.reshape(layer_inputs.q.shape[0], kv_heads, -1)
        if scales.q_scale is None:
            # 没有静态缩放时，用各自的每头缩放量化每个请求的查询。
            q, q_scale = q_per_head_fp8_quant(
                q=q,
                seq_lens=att_state.cu_seqlens_q[1:] - att_state.cu_seqlens_q[:-1],
                b1_start_loc=att_state.cu_seqlens_q,
            )
        else:
            q = q_per_head_static_fp8_quant(q=q, scales=scales.q_scale)
            q_scale = scales.q_scale.view(1, kv_heads).expand(requests, kv_heads)

        return (
            q.reshape(-1, att_config.num_heads, att_config.head_dim),
            FA3GQAPrefillBackend._paged(cache=layer_inputs.k_cache, page_size=page_size).view(torch.float8_e4m3fn),
            FA3GQAPrefillBackend._paged(cache=layer_inputs.v_cache, page_size=page_size).view(torch.float8_e4m3fn),
            q_scale.view(requests, kv_heads),
            scales.k_scale.view(1, kv_heads).expand(requests, kv_heads),
            scales.v_scale.view(1, kv_heads).expand(requests, kv_heads),
        )


@dataclass(frozen=True)
class FA3DecodeRuntimeConfig(AttRuntimeConfig):
    """GQA 和 MLA 使用的私有 FA3 解码页表的容量。"""

    page_size: int = 1
    max_seq_len: int = 0
    max_decode_rows: int = 0
    microbatch_count: int = 1
    device: str | int = "cuda"

    @classmethod
    def from_model(cls, model: "TpPartBaseModel") -> "FA3DecodeRuntimeConfig":
        args = model.args
        max_rows = args.running_max_req_size * (args.mtp_step + 1)
        if args.enable_tpsp_mix_mode:
            max_rows = triton.cdiv(x=max_rows, y=model.tp_world_size_) * model.tp_world_size_
        return replace(
            super().from_model(model=model),
            page_size=args.page_size,
            max_seq_len=model.max_seq_length,
            max_decode_rows=max(max_rows, model.graph_max_batch_size),
            microbatch_count=2 if args.enable_decode_microbatch_overlap else 1,
            device=get_current_device_id(),
        )


class FA3GQADecodeBackend(DecodeAttBackend[GQAAttConfig, FA3GQADecodeAttState, GQAAttInputs]):
    @classmethod
    def create_runtime_config(cls, model: "TpPartBaseModel") -> FA3DecodeRuntimeConfig:
        return FA3DecodeRuntimeConfig.from_model(model=model)

    def __init__(
        self,
        att_config: GQAAttConfig,
        *,
        role: str = "main",
        runtime_config: FA3DecodeRuntimeConfig = FA3DecodeRuntimeConfig(),
    ) -> None:
        """支持任何缩放、因果或非因果注意力、单侧或双侧窗口和接收器。

        每头 FP8 缓存（kv_quant='fp8_per_head'）支持默认缩放，无需窗口或
        接收器；q 用静态 q 缩放或动态地按请求量化。
        """
        super().__init__(att_config=att_config, role=role, runtime_config=runtime_config)
        FA3GQAPrefillBackend._check_supported(att_config=att_config)
        if att_config.bidirectional_image_spans:
            raise NotImplementedError("bidirectional image spans apply to prefill only")
        self._page_tables: list[torch.Tensor] = []

    def decode(
        self,
        att_state: FA3GQADecodeAttState,
        layer_inputs: GQAAttInputs,
        *,
        out: torch.Tensor,
        alloc_func: Callable[..., torch.Tensor] = torch.empty,
    ) -> torch.Tensor:
        """填充并返回 out，其形状与 q 相同；布局规则与 prefill 一致。

        q 每行对应一个批处理行；注意力状态将它们分组为注意力请求。
        """
        from lightllm.utils.sgl_utils import flash_attn_with_kvcache, flash_attn_with_kvcache_autotune

        att_config = self.att_config
        FA3GQAPrefillBackend._check_inputs(att_config=att_config, layer_inputs=layer_inputs)
        if att_config.kv_quant is not None:
            q, k, v, q_descale, k_descale, v_descale = FA3GQAPrefillBackend._fp8_operands(
                att_config=att_config, layer_inputs=layer_inputs, att_state=att_state, page_size=self._page_size
            )
            flash_attn_with_kvcache(
                q=q,
                k_cache=k,
                v_cache=v,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                page_table=att_state.page_table,
                cache_seqlens=att_state.cache_seqlens,
                cu_seqlens_q=att_state.cu_seqlens_q,
                cu_seqlens_k_new=att_state.cu_seqlens_k,
                max_seqlen_q=att_state.max_seqlen_q,
                causal=att_config.causal,
                window_size=(-1, -1),
                softcap=0.0,
                return_softmax_lse=False,
                out=out,
            )
            return out
        flash_attn_with_kvcache_autotune(
            q=layer_inputs.q,
            k_cache=FA3GQAPrefillBackend._paged(cache=layer_inputs.k_cache, page_size=self._page_size),
            v_cache=FA3GQAPrefillBackend._paged(cache=layer_inputs.v_cache, page_size=self._page_size),
            page_table=att_state.page_table,
            cache_seqlens=att_state.cache_seqlens,
            cu_seqlens_q=att_state.cu_seqlens_q,
            # KV 已在缓存中，因此不会追加新的 K/V 段。
            cu_seqlens_k_new=None,
            max_seqlen_q=att_state.max_seqlen_q,
            max_seqlen_k=att_state.max_seqlen_k,
            softmax_scale=att_config.softmax_scale,
            causal=att_config.causal,
            window_size=att_config.sliding_window,
            softcap=0.0,
            return_softmax_lse=False,
            sinks=layer_inputs.sinks,
            out=out,
        )
        return out

    @staticmethod
    def get_name() -> str:
        return "fa3_gqa_decode"

    def create_att_state(self, infer_state: InferStateInfo) -> FA3GQADecodeAttState:
        return FA3GQADecodeBackend.prepare_fa3_decode_state(
            batch=infer_state, config=self.runtime_config, page_size=self._page_size, page_tables=self._page_tables
        )

    @classmethod
    def requires_cpu_cache_sync(cls) -> bool:
        return FA3GQAPrefillBackend.fa3_requires_cpu_cache_sync()

    @staticmethod
    def prepare_fa3_decode_state(
        batch: InferStateInfo,
        config: "FA3DecodeRuntimeConfig",
        page_size: int,
        page_tables: list[torch.Tensor],
    ) -> FA3GQADecodeAttState:
        """将查询行分组为注意力请求并填充其页表行。

        页表使用固定图容量；派生长度每步生成，由 replay 更新捕获状态。
        """
        from lightllm.common.basemodel.triton_kernel.fa3_utils import page_table_copy

        cu_seqlens_q, cu_seqlens_k, cache_seqlens, att_req_idx = FA3GQADecodeBackend._group_rows(
            batch=batch, config=config
        )
        width = config.cuda_graph_max_kv_seq_len if batch.uses_cuda_graph else batch.max_kv_seq_len
        if width is None:
            raise ValueError(f"a {batch.graph_mode.value} step needs the CUDA graph kv length capacity")
        page_table = FA3GQADecodeBackend.fa3_page_table_view(
            config=config,
            page_size=page_size,
            page_tables=page_tables,
            rows=att_req_idx.shape[0],
            max_kv_len=width,
            microbatch_index=batch.microbatch_index,
        )
        page_table_copy(
            page_table=page_table[:, : triton.cdiv(x=batch.max_kv_seq_len, y=page_size)],
            req_to_token_indexs=batch.req_to_token_indexs,
            b_req_idx=att_req_idx,
            page_size=page_size,
        )
        return FA3GQADecodeAttState(
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=config.decode_rows_per_request,
            max_seqlen_k=batch.max_kv_seq_len,
        )

    @staticmethod
    def fa3_page_table_view(
        config: FA3DecodeRuntimeConfig,
        page_size: int,
        page_tables: list[torch.Tensor],
        rows: int,
        max_kv_len: int,
        microbatch_index: int,
    ) -> torch.Tensor:
        """按需为后端分配各微批次的页表，返回当前步骤需要的视图。"""
        if rows > config.max_decode_rows:
            raise RuntimeError(f"FA3 attention batch size {rows} exceeds page-table capacity {config.max_decode_rows}")
        if max_kv_len > config.max_seq_len:
            raise RuntimeError(
                f"FA3 max KV sequence length {max_kv_len} exceeds page-table capacity {config.max_seq_len}"
            )
        if not page_tables:
            numel = config.max_decode_rows * triton.cdiv(x=config.max_seq_len, y=page_size)
            page_tables.extend(
                [
                    torch.empty(size=(numel,), dtype=torch.int32, device=config.device)
                    for _ in range(config.microbatch_count)
                ]
            )
        pages = triton.cdiv(x=max_kv_len, y=page_size)
        return page_tables[microbatch_index][: rows * pages].view(rows, pages)

    @cached_property
    def _page_size(self) -> int:
        return FA3GQAPrefillBackend.fa3_page_size(
            kv_quant=self.att_config.kv_quant, page_size=self.runtime_config.page_size
        )

    @staticmethod
    def _group_rows(batch: InferStateInfo, config: AttRuntimeConfig = AttRuntimeConfig()):
        """返回每个注意力请求的 int32 查询/KV 偏移量、KV 长度和请求槽。"""
        from lightllm.common.basemodel.triton_kernel.gen_prefill_params import gen_cumsum_pad0_tensor

        rows = config.decode_rows_per_request
        if config.dynamic_decode_rows:
            from lightllm.common.basemodel.triton_kernel.fa3_utils import build_dynamic_spec_fa3_decode_params
            from lightllm.common.basemodel.triton_kernel.mtp_utils import build_mtp_shared_group_markers

            markers = build_mtp_shared_group_markers(batch.b_req_idx, hold_req_id=batch.pad_req_idx)
            q_lens, kv_lens, att_req_idx, cache_seqlens = build_dynamic_spec_fa3_decode_params(
                b_req_idx=batch.b_req_idx,
                b_seq_len=batch.b_seq_len,
                b_mark_mtp_shared_group=markers,
                att_batch_size=batch.batch_size,
                hold_req_id=batch.pad_req_idx,
            )
        elif rows > 1:
            assert batch.batch_size % rows == 0, (
                "FA3 fixed-layout decode requires batch_size to be divisible by the rows per request, "
                f"got batch_size={batch.batch_size}, rows={rows}"
            )
            # 每个请求的最后一行看到其最长的前缀。
            # 固定布局的查询偏移只读共享，缓存归本方法管理。
            offsets = FA3GQADecodeBackend._group_rows.__dict__.setdefault("_query_offsets", {})
            key = (batch.b_seq_len.device, batch.batch_size, rows)
            cu_seqlens_q = offsets.get(key)
            if cu_seqlens_q is None:
                cu_seqlens_q = offsets.setdefault(
                    key,
                    torch.arange(
                        start=0, end=batch.batch_size + 1, step=rows, dtype=torch.int32, device=batch.b_seq_len.device
                    ),
                )
            kv_lens = batch.b_seq_len[rows - 1 :: rows]
            att_req_idx = batch.b_req_idx[rows - 1 :: rows]
            cache_seqlens = kv_lens.contiguous()
            cu_seqlens_k = torch.empty(size=(kv_lens.numel() + 1,), dtype=torch.int32, device=kv_lens.device)
            cu_seqlens_k[0] = 0
            torch.cumsum(input=kv_lens, dim=0, dtype=torch.int32, out=cu_seqlens_k[1:])
            return cu_seqlens_q, cu_seqlens_k, cache_seqlens, att_req_idx
        else:
            return batch.b1_cu_q_seq_len.int(), batch.b1_cu_kv_seq_len.int(), batch.b_seq_len, batch.b_req_idx
        cu_seqlens_q, cu_seqlens_k = gen_cumsum_pad0_tensor(q_lens, kv_lens)
        return cu_seqlens_q.int(), cu_seqlens_k.int(), cache_seqlens, att_req_idx
