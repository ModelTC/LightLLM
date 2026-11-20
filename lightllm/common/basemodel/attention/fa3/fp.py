import dataclasses
import weakref
import torch
import triton
from ..base_att import BaseAttBackend, BasePrefillAttState, BaseDecodeAttState, AttControl, build_attention_batch
from typing import Optional
from lightllm.utils.envs_utils import enable_env_vars
from lightllm.common.attention.backends.gqa import GQALayerConfig, GQALayerInputs
from lightllm.common.attention.backends.gqa.fa3 import (
    FA3GQADecodeBackend,
    FA3GQADecodeStepArgs,
    FA3GQAPrefillBackend,
    FA3GQAPrefillStepArgs,
)


class Fa3AttBackend(BaseAttBackend):
    """Common fixed page-table storage for FA3 attention backends."""

    # 设为 1 / TRUE / ON 时，FA3 与 CPU cache 搬运在 GPU 上互相等待。
    requires_cpu_cache_sync = enable_env_vars("LIGHTLLM_FA3_CPU_CACHE_SYNC")
    page_table_buffers = None

    def __init__(self, model):
        super().__init__(model=model)
        self._init_infer_page_size()
        self.prefill_step_builder = FA3GQAPrefillBackend.get_step_builder_cls()(page_size=self.infer_page_size)
        # BaseAttBackend reuses one instance per model but reruns __init__ on every construction.
        # Keep the decode step builder: captured CUDA graphs read the tensors it keeps.
        if getattr(self, "decode_step_builder", None) is None:
            self.decode_step_builder = FA3GQADecodeBackend.get_step_builder_cls()(
                page_size=self.infer_page_size, page_table_view=self.get_page_table_view
            )

    def _init_infer_page_size(self):
        self.infer_page_size = self.model.args.page_size
        assert self.model.args.page_size % self.infer_page_size == 0, (
            f"model page_size {self.model.args.page_size} "
            f"must be divisible by infer_page_size {self.infer_page_size}"
        )

    # 延迟到首次获取 page table 时再初始化，避免 PD 分离模式下的 prefill 节点
    # 分配仅供 decode 使用的 buffer，减少显存浪费。
    def _init_page_table_buffers(self):
        if self.page_table_buffers is not None:
            return

        model = self.model
        args = model.args
        # Dynamic verification may keep the target row and all MTP draft rows for
        # every running request. TPSP and CUDA Graph may pad that physical batch
        # further, so the fixed buffer must cover both execution paths.
        running_max_batch_size = args.running_max_req_size * (args.mtp_step + 1)
        if args.enable_tpsp_mix_mode:
            tp_size = model.tp_world_size_
            running_max_batch_size = (running_max_batch_size + tp_size - 1) // tp_size * tp_size
        self.page_table_max_batch_size = max(running_max_batch_size, model.graph_max_batch_size)
        # max_seq_length is max_req_total_len plus the MTP headroom reserved when
        # the model is initialized.
        self.supported_max_seq_len = model.max_seq_length
        buffer_count = 2 if args.enable_decode_microbatch_overlap else 1
        workspace_size = self.page_table_max_batch_size * triton.cdiv(self.supported_max_seq_len, self.infer_page_size)
        self.page_table_buffers = [
            self.get_gpu_workspace_buffer(
                key_name=f"fa3_page_table_{buffer_index}",
                workspace_size=workspace_size,
                dtype=torch.int32,
            )
            for buffer_index in range(buffer_count)
        ]

    def get_page_table_view(self, att_batch_size, max_kv_len, microbatch_index):
        """Lazily initialize storage and return a contiguous page-table view."""
        self._init_page_table_buffers()
        if att_batch_size > self.page_table_max_batch_size:
            raise RuntimeError(
                f"FA3 attention batch size {att_batch_size} exceeds page-table capacity "
                f"{self.page_table_max_batch_size}"
            )
        if max_kv_len > self.supported_max_seq_len:
            raise RuntimeError(
                f"FA3 max KV sequence length {max_kv_len} exceeds page-table capacity " f"{self.supported_max_seq_len}"
            )
        max_page_len = triton.cdiv(max_kv_len, self.infer_page_size)
        return self.page_table_buffers[microbatch_index][: att_batch_size * max_page_len].reshape(
            att_batch_size, max_page_len
        )

    def create_att_prefill_state(self, infer_state) -> "Fa3PrefillAttState":
        return Fa3PrefillAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))

    def create_att_decode_state(self, infer_state) -> "Fa3DecodeAttState":
        return Fa3DecodeAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))


def _gqa_layer_inputs_and_config(q, k, v, att_control: AttControl, causal: bool):
    assert att_control.use_alibi is False, "FA3 GQA does not support alibi"
    sinks = att_control.sink_weight if att_control.use_att_sink else None
    layer_config = GQALayerConfig(
        num_heads=q.shape[1],
        num_kv_heads=k.shape[1],
        head_dim=q.shape[2],
        dtype=q.dtype,
        sliding_window=att_control.sliding_window if att_control.use_sliding_window else (-1, -1),
        causal=causal,
        has_sinks=sinks is not None,
    )
    return GQALayerInputs(q=q, k_cache=k, v_cache=v, sinks=sinks), layer_config


@dataclasses.dataclass
class Fa3PrefillAttState(BasePrefillAttState):
    step_args: Optional[FA3GQAPrefillStepArgs] = None
    # Mirrors of the step args; the FP8 and neo subclasses still call FA3 directly with them.
    cu_seqlens_q: torch.Tensor = None
    cu_seqlens_k: torch.Tensor = None
    page_table: torch.Tensor = None
    causal: bool = None

    def init_state(self):
        self.causal = self.backend.uses_causal_attention()
        step_args = self.backend.prefill_step_builder.build(build_attention_batch(self.infer_state, self.backend))
        self.step_args = step_args
        self.cu_seqlens_q, self.cu_seqlens_k, self.page_table = (
            step_args.cu_seqlens_q,
            step_args.cu_seqlens_k,
            step_args.page_table,
        )

    def prefill_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        layer_inputs, layer_config = _gqa_layer_inputs_and_config(q, k, v, att_control, self.causal)
        impl = self.backend.get_attention_impl(FA3GQAPrefillBackend, layer_config)
        return impl.forward(
            self.step_args, layer_inputs, out=alloc_func(q.shape, dtype=q.dtype, device=q.device), alloc_func=alloc_func
        )


@dataclasses.dataclass
class Fa3DecodeAttState(BaseDecodeAttState):
    step_args: Optional[FA3GQADecodeStepArgs] = None
    # Mirrors of the step args; the FP8 subclass still calls FA3 directly with them.
    cu_seqlens_q: torch.Tensor = None
    cu_seqlens_k: torch.Tensor = None
    page_table: torch.Tensor = None
    b_att_seq_len: torch.Tensor = None
    decode_max_q_seq_len: int = None
    decode_max_kv_seq_len: int = None
    causal: bool = None

    def init_state(self):
        self.causal = self.backend.uses_causal_attention()
        # Replay steps also build: the step builder rewrites the tensors the graph captured.
        step_args = self.backend.decode_step_builder.build(build_attention_batch(self.infer_state, self.backend))
        self.step_args = step_args
        self.cu_seqlens_q, self.cu_seqlens_k, self.page_table = (
            step_args.cu_seqlens_q,
            step_args.cu_seqlens_k,
            step_args.page_table,
        )
        self.b_att_seq_len = step_args.cache_seqlens
        self.decode_max_q_seq_len, self.decode_max_kv_seq_len = step_args.max_seqlen_q, step_args.max_seqlen_k

    def decode_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ):
        layer_inputs, layer_config = _gqa_layer_inputs_and_config(q, k, v, att_control, self.causal)
        impl = self.backend.get_attention_impl(FA3GQADecodeBackend, layer_config)
        return impl.forward(
            self.step_args, layer_inputs, out=alloc_func(q.shape, dtype=q.dtype, device=q.device), alloc_func=alloc_func
        )
