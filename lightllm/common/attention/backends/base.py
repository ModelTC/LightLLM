import copy
import importlib
from dataclasses import dataclass, replace
from types import MappingProxyType

import torch

from lightllm.common.attention.api.backend import AttentionBackend, LayerAttentionBinding, UnsupportedAttention
from lightllm.common.attention.api.batch import AttentionBatch, PlanCapacity
from lightllm.common.attention.api.spec import AttentionFamily, AttentionPhase, LayerSpec
from lightllm.platform import DeviceCaps, Platform


@dataclass
class PreparedPlan:
    capacity: PlanCapacity
    capture: bool
    workspace: object
    batch: AttentionBatch | None = None
    current_max_kv_seq_len: int = 0
    backend_identity: object = None


class BaseAttentionBackend(AttentionBackend):
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
        self.mtp_state_index = None

    def for_spec(self, spec):
        backend = copy.copy(self)
        backend.spec = spec
        return backend

    @classmethod
    def check_environment(cls, caps: DeviceCaps) -> None:
        if caps.platform not in cls.supported_platforms:
            raise UnsupportedAttention(f"{cls.provider} does not support platform {caps.platform.value}")
        if not caps.available:
            raise UnsupportedAttention(f"device {caps.device} is unavailable")
        if cls.min_device_capability is not None and (
            caps.capability is None or caps.capability < cls.min_device_capability
        ):
            raise UnsupportedAttention(f"requires device capability {cls.min_device_capability}")
        try:
            if cls.dependency:
                importlib.import_module(cls.dependency)
        except (ImportError, OSError) as exc:
            raise UnsupportedAttention(str(exc)) from exc

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

    def allocate_plan(self, capacity: PlanCapacity, *, capture: bool, workspace) -> PreparedPlan:
        if capacity.phase is AttentionPhase.PREFILL:
            plan = self._allocate_prefill_plan(capacity, capture, workspace)
        else:
            plan = self._allocate_decode_plan(capacity, capture, workspace)
        plan.backend_identity = self
        return plan

    def _allocate_prefill_plan(self, capacity, capture, workspace):
        return PreparedPlan(capacity, capture, workspace)

    def _allocate_decode_plan(self, capacity, capture, workspace):
        return PreparedPlan(capacity, capture, workspace)

    def prepare(self, plan: PreparedPlan, batch: AttentionBatch) -> None:
        if plan.backend_identity is not self:
            raise ValueError("plan belongs to a different attention backend")
        batch.validate()
        plan.capacity.check(batch)
        plan.current_max_kv_seq_len = batch.max_kv_seq_len
        if plan.capture and plan.batch is not None:
            plan.batch.copy_inputs(batch)
        elif plan.capture:
            # 图内分配和 launch 使用容量上界；图外 provider planning 仍用 current_max_kv_seq_len。
            plan.batch = replace(
                batch,
                max_kv_seq_len=plan.capacity.max_kv_seq_len,
                total_token_num=plan.capacity.batch_size * plan.capacity.max_kv_seq_len,
            )
        else:
            plan.batch = batch
        if batch.phase is AttentionPhase.PREFILL:
            self._prepare_prefill(plan)
        else:
            self._prepare_decode(plan)

    def _prepare_prefill(self, plan):
        pass

    def _prepare_decode(self, plan):
        pass

    def update(self, plan: PreparedPlan) -> None:
        if plan.backend_identity is not self:
            raise ValueError("plan belongs to a different attention backend")
        if plan.batch is None:
            raise RuntimeError("prepare attention inputs before update")
        if plan.capacity.phase is AttentionPhase.PREFILL:
            self._update_prefill(plan)
        else:
            self._update_decode(plan)

    def _update_prefill(self, plan):
        pass

    def _update_decode(self, plan):
        pass

    def forward(self, layer: LayerAttentionBinding, plan: PreparedPlan, inputs, *, out=None):
        if plan.batch is None:
            raise RuntimeError("prepare attention inputs before forward")
        if plan.backend_identity is not self:
            raise ValueError("plan belongs to a different attention backend")
        if layer.spec.phase is not plan.capacity.phase:
            raise ValueError("layer attention and plan phases differ")
        method = self._forward_prefill if plan.capacity.phase is AttentionPhase.PREFILL else self._forward_decode
        result = method(layer, plan, inputs, out=out)
        if out is None:
            return result
        attention = result[0] if isinstance(result, tuple) else result
        if attention.data_ptr() == out.data_ptr():
            return result
        out.copy_(attention.view(out.shape))

        return (out, *result[1:]) if isinstance(result, tuple) else out

    def uses_dynamic_spec_verify_layout(self):
        return self.dynamic_verify

    def uses_causal_attention(self):
        return self.causal
