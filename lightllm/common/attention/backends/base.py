from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Generic, TypeVar

import torch

from lightllm.common.attention._environment import check_attention_environment
from lightllm.common.attention.api.backend import AttentionBackend, LayerAttentionBinding, UnsupportedAttention
from lightllm.common.attention.api.batch import AttentionBatch, ContextCapacity
from lightllm.common.attention.api.spec import AttentionFamily, AttentionPhase, LayerSpec
from lightllm.common.attention.backends._workspace import AttentionWorkspace
from lightllm.platform import DeviceCaps, Platform


@dataclass(slots=True)
class AttentionExecutionContext:
    capacity: ContextCapacity
    capture: bool
    workspace: AttentionWorkspace
    batch: AttentionBatch | None = None
    current_max_kv_seq_len: int = 0
    backend_identity: AttentionBackend | None = None


Context = TypeVar("Context", bound=AttentionExecutionContext)
Inputs = TypeVar("Inputs")
Output = TypeVar("Output")


class BaseAttentionBackend(AttentionBackend[Context, Inputs, Output], Generic[Context, Inputs, Output]):
    requires_cpu_cache_sync = False
    family = AttentionFamily.DENSE
    provider = "triton"
    cache_formats = frozenset({"None"})
    supports_attention_sink = False
    supports_noncausal_decode = False
    supported_platforms = frozenset({Platform.CUDA})
    supported_phases = frozenset(AttentionPhase)
    supported_graph_phases = frozenset(AttentionPhase)
    activation_dtypes = frozenset({torch.float16, torch.bfloat16, torch.float32})
    dependency = "triton"
    min_device_capability = None
    supports_alibi = False
    supports_sliding_window = False
    supports_speculative_window = True
    supports_expert_parallel = True
    max_head_size = None

    def __init__(self, model):
        self.runtime = model.device_runtime
        self.execution = model.attention_execution
        self.page_size = model.args.page_size
        self.network_config = MappingProxyType(dict(model.config))
        self.is_draft_model = model.is_mtp_draft_model
        self.decode_draft_step = model.mtp_manager.get_decode_draft_step(model.is_mtp_draft_model)
        self.dynamic_verify = not self.is_draft_model and self.decode_draft_step > 0 and model.args.mtp_dynamic_verify
        self.causal = not (self.is_draft_model and model.args.mtp_mode in ("dspark", "dflash", "dflash2"))
        self.hold_req_id = model.req_manager.HOLD_REQUEST_ID

    @classmethod
    def create(cls, model, spec: LayerSpec):
        backend = cls(model)
        backend.configure_spec(spec)
        return backend

    def configure_spec(self, spec: LayerSpec) -> None:
        """Apply per-group dimensions before allocating contexts; default backends use layer params."""
        pass

    @classmethod
    def check_environment(cls, caps: DeviceCaps) -> None:
        check_attention_environment(
            caps,
            provider=cls.provider,
            supported_platforms=cls.supported_platforms,
            min_device_capability=cls.min_device_capability,
            dependency=cls.dependency,
        )

    @classmethod
    def check_support(cls, spec: LayerSpec, caps=None, execution=None) -> None:
        if spec.family is not cls.family:
            raise UnsupportedAttention(f"{cls.__name__} does not implement {spec.family.value}")
        if spec.phase not in cls.supported_phases:
            raise UnsupportedAttention(f"{cls.provider} does not implement {spec.phase.value}")
        if spec.cache.format not in cls.cache_formats:
            raise UnsupportedAttention(f"cache encoding {spec.cache.format} is unsupported")
        if spec.dtype not in cls.activation_dtypes:
            raise UnsupportedAttention(f"activation dtype {spec.dtype} is unsupported")
        if spec.use_alibi and not cls.supports_alibi:
            raise UnsupportedAttention("this backend does not support ALiBi")
        if spec.use_alibi and spec.sliding_window is not None:
            raise UnsupportedAttention("ALiBi with sliding-window attention is unsupported")
        if spec.use_sink and not cls.supports_attention_sink:
            raise UnsupportedAttention("this backend does not support attention sinks")
        if spec.sliding_window is not None and not cls.supports_sliding_window:
            raise UnsupportedAttention("this backend does not support sliding-window attention")
        if cls.max_head_size is not None and spec.head_size > cls.max_head_size:
            raise UnsupportedAttention(f"head size exceeds {cls.max_head_size}")
        if spec.phase is AttentionPhase.DECODE and not spec.causal and not cls.supports_noncausal_decode:
            raise UnsupportedAttention("this backend does not support non-causal block decode")
        if spec.use_mtp and spec.sliding_window is not None and not cls.supports_speculative_window:
            raise UnsupportedAttention("speculative decode does not implement sliding windows")
        if execution is not None:
            if execution.expert_parallel and not cls.supports_expert_parallel:
                raise UnsupportedAttention("this backend is incompatible with expert-parallel MoE execution")
            if spec.phase in execution.graph_phases and (
                spec.phase not in cls.supported_graph_phases or (caps is not None and not caps.supports_graph)
            ):
                raise UnsupportedAttention(f"{spec.phase.value} Graph is unsupported")

    def allocate_context(self, capacity: ContextCapacity, *, capture: bool, workspace) -> AttentionExecutionContext:
        if capacity.phase is AttentionPhase.PREFILL:
            context = self._allocate_prefill_context(capacity, capture, workspace)
        else:
            context = self._allocate_decode_context(capacity, capture, workspace)
        context.backend_identity = self
        return context

    def _allocate_prefill_context(self, capacity, capture, workspace):
        return AttentionExecutionContext(capacity, capture, workspace)

    def _allocate_decode_context(self, capacity, capture, workspace):
        return AttentionExecutionContext(capacity, capture, workspace)

    def prepare(self, context: AttentionExecutionContext, batch: AttentionBatch) -> None:
        if context.backend_identity is not self:
            raise ValueError("context belongs to a different attention backend")
        batch.validate()
        context.capacity.check(batch)
        context.current_max_kv_seq_len = batch.max_kv_seq_len
        if context.capture and context.batch is not None:
            context.batch.copy_inputs(batch)
        elif context.capture:
            # 图内分配和 launch 使用容量上界；图外 provider planning 仍用 current_max_kv_seq_len。
            context.batch = replace(
                batch,
                max_kv_seq_len=context.capacity.max_kv_seq_len,
                total_token_num=context.capacity.batch_size * context.capacity.max_kv_seq_len,
            )
        else:
            context.batch = batch
        if batch.phase is AttentionPhase.PREFILL:
            self._prepare_prefill(context)
        else:
            self._prepare_decode(context)

    def _prepare_prefill(self, context):
        """Optional host planning; kernels without provider plans need no preparation."""
        pass

    def _prepare_decode(self, context):
        """Optional host planning, including updates before Graph replay."""
        pass

    def update(self, context: AttentionExecutionContext) -> None:
        if context.backend_identity is not self:
            raise ValueError("context belongs to a different attention backend")
        if context.batch is None:
            raise RuntimeError("prepare attention inputs before update")
        if context.capacity.phase is AttentionPhase.PREFILL:
            self._update_prefill(context)
        else:
            self._update_decode(context)

    def _update_prefill(self, context):
        """Optional derived metadata; direct consumers of batch tensors need no update."""
        pass

    def _update_decode(self, context):
        """Optional GPU metadata generation, captured alongside layer execution."""
        pass

    def forward(self, layer: LayerAttentionBinding, context: AttentionExecutionContext, inputs, *, out=None):
        if context.batch is None:
            raise RuntimeError("prepare attention inputs before forward")
        if context.backend_identity is not self:
            raise ValueError("context belongs to a different attention backend")
        if layer.spec.phase is not context.capacity.phase:
            raise ValueError("layer attention and context phases differ")
        method = self._forward_prefill if context.capacity.phase is AttentionPhase.PREFILL else self._forward_decode
        result = method(layer, context, inputs, out=out)
        if out is None:
            return result
        attention = result[0] if isinstance(result, tuple) else result
        if attention.dtype != out.dtype or attention.device != out.device:
            raise ValueError("out must match the attention result dtype and device")
        if attention.untyped_storage().data_ptr() == out.untyped_storage().data_ptr():
            # Providers may return the valid heads of a padded output buffer.
            begin = attention.storage_offset()
            end = begin + sum((size - 1) * stride for size, stride in zip(attention.shape, attention.stride())) + 1
            out_end = (
                out.storage_offset() + sum((size - 1) * stride for size, stride in zip(out.shape, out.stride())) + 1
            )
            if attention.numel() and (begin < out.storage_offset() or end > out_end):
                raise ValueError("attention result view extends beyond out")
            return result
        if attention.numel() != out.numel():
            raise ValueError("out must match the attention result element count")
        out.copy_(attention.reshape(out.shape))
        return (out, *result[1:]) if isinstance(result, tuple) else out

    @property
    def infer_page_size(self) -> int:
        return self.page_size

    def uses_dynamic_spec_verify_layout(self):
        return self.dynamic_verify

    def uses_causal_attention(self):
        return self.causal
