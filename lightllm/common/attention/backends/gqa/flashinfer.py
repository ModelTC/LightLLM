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
from . import GQAAttConfig, GQAAttInputs

if TYPE_CHECKING:
    from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
    from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper
    from lightllm.common.basemodel.basemodel import TpPartBaseModel


@dataclass(frozen=True)
class FlashInferGQAPrefillAttState:
    """针对分页缓存上打包的因果 prefill 的计划 NHD 封装对象。"""

    wrapper: "BatchPrefillWithPagedKVCacheWrapper"


@dataclass(frozen=True)
class FlashInferGQADecodeAttState:
    """针对每个批处理行一个查询的计划 NHD 张量核心封装对象。"""

    wrapper: "BatchDecodeWithPagedKVCacheWrapper | None"
    kv_starts: torch.Tensor
    kv_indices: torch.Tensor
    kv_last_page_len: torch.Tensor


class FlashInferGQAPrefillBackend(PrefillAttBackend[GQAAttConfig, FlashInferGQAPrefillAttState, GQAAttInputs]):
    @dataclass(frozen=True)
    class RuntimeConfig(AttRuntimeConfig):
        page_size: int = 1
        max_seq_len: int = 0
        workspace_bytes: int = 256 * 1024 * 1024

    @classmethod
    def create_runtime_config(cls, model: "TpPartBaseModel") -> RuntimeConfig:
        return replace(
            cls.RuntimeConfig.from_model(model=model),
            page_size=model.args.page_size,
            max_seq_len=model.max_seq_length,
            workspace_bytes=model.args.flashinfer_workspace_size * 1024 * 1024,
        )

    def __init__(
        self,
        att_config: GQAAttConfig,
        *,
        role: str = "main",
        runtime_config: RuntimeConfig = RuntimeConfig(),
    ) -> None:
        super().__init__(att_config=att_config, role=role, runtime_config=runtime_config)
        if runtime_config.workspace_bytes <= 0:
            raise ValueError("the FlashInfer workspace size (--flashinfer_workspace_size) must be positive")
        FlashInferGQAPrefillBackend._check_supported(att_config=att_config)

    def prefill(
        self,
        att_state: FlashInferGQAPrefillAttState,
        layer_inputs: GQAAttInputs,
        *,
        out: torch.Tensor,
        alloc_func: Callable[..., torch.Tensor] = torch.empty,
    ) -> torch.Tensor:
        """填充并返回 out，其形状、dtype 和设备与 q 相同。

        Q/out 是连续的 FP16/BF16 张量，与缓存和 att_state 位于同一设备。缓存槽构成完整页。
        槽的步长可以有 padding，也支持共享存储中的 K/V 切片。Q/K/V 只读，
        out 与这些输入及 attention 状态缓冲区的存储必须分离。临时缓冲区使用封装对象
        在规划时拿到的工作区，前向计算无需调用 alloc_func。
        """
        return FlashInferGQAPrefillBackend._run(
            att_config=self.att_config,
            wrapper=att_state.wrapper,
            page_size=self._page_size,
            layer_inputs=layer_inputs,
            out=out,
        )

    @staticmethod
    def get_name() -> str:
        return "flashinfer_gqa_prefill"

    def create_att_state(self, infer_state: InferStateInfo) -> FlashInferGQAPrefillAttState:
        """为步骤的打包查询和分页缓存构建并计划一个 prefill 封装对象。"""
        config = self.att_config
        page_size = self._page_size
        import flashinfer

        q_starts = infer_state.b1_cu_q_seq_len.int()
        max_pages = triton.cdiv(x=self.runtime_config.max_seq_len, y=page_size)
        kv_indices = torch.empty(
            size=(infer_state.batch_size * max_pages,), dtype=torch.int32, device=infer_state.b_seq_len.device
        )
        kv_starts, kv_last_page_len = FlashInferGQAPrefillBackend._page_metadata(
            batch=infer_state, page_size=page_size, kv_indices=kv_indices
        )
        wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            float_workspace_buffer=self.get_workspace(
                name="flashinfer_fp", role=self.role, numel=self.runtime_config.workspace_bytes
            ),
            qo_indptr_buf=q_starts,
            paged_kv_indptr_buf=kv_starts,
            paged_kv_indices_buf=kv_indices,
            paged_kv_last_page_len_buf=kv_last_page_len,
        )
        wrapper.plan(
            qo_indptr=q_starts,
            paged_kv_indptr=kv_starts,
            paged_kv_indices=kv_indices,
            paged_kv_last_page_len=kv_last_page_len,
            num_qo_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim_qk=config.head_dim,
            page_size=page_size,
            causal=True,
            pos_encoding_mode="NONE",
            logits_soft_cap=0.0,
            q_data_type=config.dtype,
            kv_data_type=FlashInferGQAPrefillBackend._kv_dtype(att_config=config),
        )
        return FlashInferGQAPrefillAttState(wrapper=wrapper)

    @cached_property
    def _page_size(self) -> int:
        init = self.runtime_config
        return FlashInferGQAPrefillBackend._prepare_page_size(
            att_config=self.att_config, page_size=init.page_size, max_seq_len=init.max_seq_len
        )

    @staticmethod
    def _page_metadata(batch: InferStateInfo, page_size: int, kv_indices: torch.Tensor):
        """返回每个请求的页偏移量和最后一页长度，并将页 ID 写入 kv_indices。"""
        from lightllm.common.basemodel.triton_kernel.gen_prefill_params import gen_cumsum_pad0_tensor
        from lightllm.common.basemodel.triton_kernel.repack_kv_index import repack_kv_index

        # 部分最终页仍然算作一页。
        b_page_len = (batch.b_seq_len + (page_size - 1)) // page_size
        kv_starts, _ = gen_cumsum_pad0_tensor(b_page_len, b_page_len)
        kv_last_page_len = batch.b_seq_len - (b_page_len - 1) * page_size
        repack_kv_index(
            req_to_token_indexs=batch.req_to_token_indexs,
            b_req_idx=batch.b_req_idx,
            b_token_len=batch.b_seq_len,
            b_page_start_loc=kv_starts[:-1],
            max_token_len=batch.max_kv_seq_len,
            out_page_indices=kv_indices,
            page_size=page_size,
        )
        return kv_starts, kv_last_page_len

    @staticmethod
    def _check_supported(att_config: GQAAttConfig) -> None:
        if not (
            att_config.causal
            and att_config.sliding_window == (-1, -1)
            and att_config.has_default_scale
            and not att_config.has_sinks
            and att_config.kv_quant in (None, "fp8_per_tensor")
            and not att_config.has_alibi
            and not att_config.bidirectional_image_spans
        ):
            raise NotImplementedError(
                "FlashInfer GQA supports only causal attention without sliding window or sinks, "
                "with softmax_scale 1/sqrt(head_dim) and kv_quant None or 'fp8_per_tensor'"
            )

    @staticmethod
    def _run(
        att_config: GQAAttConfig,
        wrapper: "BatchPrefillWithPagedKVCacheWrapper | BatchDecodeWithPagedKVCacheWrapper",
        page_size: int,
        layer_inputs: GQAAttInputs,
        out: torch.Tensor,
    ):
        assert layer_inputs.q.dtype == att_config.dtype, "q does not match the att_config"
        assert (layer_inputs.kv_scales is not None) == (
            att_config.kv_quant is not None
        ), "kv_scales must be given exactly when the att_config has kv_quant"
        k, v = layer_inputs.k_cache, layer_inputs.v_cache
        k = k.view(-1, page_size, k.shape[1], k.shape[2])
        v = v.view(-1, page_size, v.shape[1], v.shape[2])
        if att_config.kv_quant is None:
            wrapper.run(q=layer_inputs.q, paged_kv_cache=(k, v), out=out)
            return out
        wrapper.run(
            q=layer_inputs.q,
            paged_kv_cache=(k.view(torch.float8_e4m3fn), v.view(torch.float8_e4m3fn)),
            k_scale=layer_inputs.kv_scales.k_scale,
            v_scale=layer_inputs.kv_scales.v_scale,
            out=out,
        )
        return out

    @staticmethod
    def _prepare_page_size(att_config: GQAAttConfig, page_size: int, max_seq_len: int) -> int:
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive before preparing FlashInfer state")
        set_flashinfer_envs()
        return 1 if att_config.kv_quant == "fp8_per_tensor" else page_size

    @staticmethod
    def _kv_dtype(att_config: GQAAttConfig) -> torch.dtype:
        return torch.float8_e4m3fn if att_config.kv_quant == "fp8_per_tensor" else att_config.dtype


class FlashInferGQADecodeBackend(DecodeAttBackend[GQAAttConfig, FlashInferGQADecodeAttState, GQAAttInputs]):
    @dataclass(frozen=True)
    class RuntimeConfig(AttRuntimeConfig):
        page_size: int = 1
        max_seq_len: int = 0
        max_batch_size: int = 0
        workspace_bytes: int = 256 * 1024 * 1024
        device: str | int = "cuda"

    @classmethod
    def create_runtime_config(cls, model: "TpPartBaseModel") -> RuntimeConfig:
        return replace(
            cls.RuntimeConfig.from_model(model=model),
            page_size=model.args.page_size,
            max_seq_len=model.max_seq_length,
            max_batch_size=model.graph_max_batch_size,
            workspace_bytes=model.args.flashinfer_workspace_size * 1024 * 1024,
            device=get_current_device_id(),
        )

    def __init__(
        self,
        att_config: GQAAttConfig,
        *,
        role: str = "main",
        runtime_config: RuntimeConfig = RuntimeConfig(),
    ) -> None:
        super().__init__(att_config=att_config, role=role, runtime_config=runtime_config)
        if runtime_config.workspace_bytes <= 0:
            raise ValueError("the FlashInfer workspace size (--flashinfer_workspace_size) must be positive")
        FlashInferGQAPrefillBackend._check_supported(att_config=att_config)

    def decode(
        self,
        att_state: FlashInferGQADecodeAttState,
        layer_inputs: GQAAttInputs,
        *,
        out: torch.Tensor,
        alloc_func: Callable[..., torch.Tensor] = torch.empty,
    ) -> torch.Tensor:
        """填充并返回 out，其形状与 q 相同，每批行一个查询；布局规则与 prefill 一致。"""
        return FlashInferGQAPrefillBackend._run(
            att_config=self.att_config,
            wrapper=att_state.wrapper,
            page_size=self._page_size,
            layer_inputs=layer_inputs,
            out=out,
        )

    @staticmethod
    def get_name() -> str:
        return "flashinfer_gqa_decode"

    def create_att_state(self, infer_state: InferStateInfo) -> FlashInferGQADecodeAttState:
        """构建注意力状态，为这一步规划 FlashInfer 的 decode 封装对象。

        每个元数据行都被视为独立的请求，因此投机解码行不需要重新分组。捕获状态持有 wrapper 和页参数。重放步骤只生成新的页参数，
        复制到捕获状态后通过 fast_decode_plan 重新规划。
        """
        config = self.att_config
        page_size = self._page_size
        import flashinfer

        init = self.runtime_config
        page_ids = infer_state.batch_size * triton.cdiv(x=init.max_seq_len, y=page_size)
        if infer_state.batch_size <= init.max_batch_size:
            kv_indices = self._kv_indices[infer_state.microbatch_index][:page_ids]
        else:
            # 捕获的封装对象必须读取固定的每微批次缓冲区。
            assert not infer_state.uses_cuda_graph, "CUDA graph batches must fit the page-index buffers"
            kv_indices = torch.empty(size=(page_ids,), dtype=torch.int32, device=infer_state.b_seq_len.device)
        kv_starts, kv_last_page_len = FlashInferGQAPrefillBackend._page_metadata(
            batch=infer_state, page_size=page_size, kv_indices=kv_indices
        )

        if infer_state.graph_mode is GraphMode.REPLAY:
            return FlashInferGQADecodeAttState(
                wrapper=None, kv_starts=kv_starts, kv_indices=kv_indices, kv_last_page_len=kv_last_page_len
            )

        wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
            float_workspace_buffer=self.get_workspace(name="flashinfer_fp", role=self.role, numel=init.workspace_bytes),
            kv_layout="NHD",
            use_cuda_graph=True,
            use_tensor_cores=True,
            paged_kv_indptr_buffer=kv_starts,
            paged_kv_indices_buffer=kv_indices,
            paged_kv_last_page_len_buffer=kv_last_page_len,
        )
        wrapper.plan(
            indptr=kv_starts,
            indices=kv_indices,
            last_page_len=kv_last_page_len,
            num_qo_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            page_size=page_size,
            q_data_type=config.dtype,
            kv_data_type=FlashInferGQAPrefillBackend._kv_dtype(att_config=config),
            non_blocking=True,
        )
        return FlashInferGQADecodeAttState(
            wrapper=wrapper, kv_starts=kv_starts, kv_indices=kv_indices, kv_last_page_len=kv_last_page_len
        )

    def refresh_cuda_graph_state(self, att_state: FlashInferGQADecodeAttState, infer_state: InferStateInfo) -> None:
        """动态页参数复制后，重新规划捕获状态持有的 wrapper。"""
        self._fast_replan(
            wrapper=att_state.wrapper,
            kv_starts=att_state.kv_starts,
            kv_indices=att_state.kv_indices,
            kv_last_page_len=att_state.kv_last_page_len,
            batch=infer_state,
        )

    @cached_property
    def _page_size(self) -> int:
        init = self.runtime_config
        return FlashInferGQAPrefillBackend._prepare_page_size(
            att_config=self.att_config, page_size=init.page_size, max_seq_len=init.max_seq_len
        )

    @cached_property
    def _kv_indices(self) -> list[torch.Tensor]:
        # 捕获的图读取这些地址，每个微批次和后端对应一个分配。
        init = self.runtime_config
        max_pages = triton.cdiv(x=init.max_seq_len, y=self._page_size)
        return [
            torch.empty(size=(init.max_batch_size * max_pages,), dtype=torch.int32, device=init.device)
            for _ in range(2)
        ]

    def _fast_replan(
        self,
        wrapper: "BatchDecodeWithPagedKVCacheWrapper",
        kv_starts: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_last_page_len: torch.Tensor,
        batch: InferStateInfo,
    ) -> None:
        from flashinfer.decode import fast_decode_plan

        config = self.att_config
        page_size = self._page_size
        uniform_kv_indptr_cpu = torch.arange(batch.batch_size + 1, dtype=torch.int32, device="cpu") * triton.cdiv(
            x=batch.max_kv_seq_len, y=page_size
        )
        fast_decode_plan(
            self=wrapper,
            indptr=kv_starts,
            indices=kv_indices,
            last_page_len=kv_last_page_len,
            num_qo_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            page_size=page_size,
            q_data_type=config.dtype,
            kv_data_type=FlashInferGQAPrefillBackend._kv_dtype(att_config=config),
            non_blocking=True,
            global_override_indptr_cpu=uniform_kv_indptr_cpu,
        )
