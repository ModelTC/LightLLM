import threading
import weakref

import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING, Tuple

from lightllm.utils.dist_utils import get_current_device_id
from lightllm.utils.envs_utils import get_env_start_args

if TYPE_CHECKING:
    from lightllm.common.basemodel.basemodel import TpPartBaseModel
    from lightllm.common.basemodel.infer_struct import InferStateInfo


class BaseAttBackend:
    """
    用于创建支持各种不同的AttBackend, 如 fa3, flashinfer, triton 实现等。
    每个 model 复用一个 backend 实例。
    """

    _instances = {}
    _workspace_buffers = {}
    _workspace_buffer_lock = threading.Lock()
    supported_kv_encodings = ("plain",)
    supported_kv_layouts = ("separate_kv_nhd",)
    supported_attention_features = ()
    supports_mla = False
    requires_cache_stream_sync = False

    def __new__(cls, *args, **kwargs):
        """
        Main 和 speculative draft model 可能使用不同的 CUDA graph 上限
        和缓存布局，不能只按 backend class 共享实例。
        """
        model = kwargs.get("model", args[0] if args else None)
        instance_key = (cls, model, kwargs.get("attention_shape"))
        if instance_key not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[instance_key] = instance
        return cls._instances[instance_key]

    def __init__(self, model: "TpPartBaseModel"):
        self.model = model

    def attention_support_error(self, spec, format, phase):
        for feature, required in (
            ("alibi", spec.alibi),
            ("window", spec.window != (-1, -1)),
            ("sink", spec.attention_sink),
            ("noncausal", not spec.causal),
        ):
            if required and feature not in self.supported_attention_features:
                return f"unsupported attention feature {feature}"
        if format.version != 1 or format.layout not in self.supported_kv_layouts:
            return f"unsupported KV layout {format.layout} version {format.version}"
        if format.encoding not in self.supported_kv_encodings:
            return f"unsupported KV encoding {format.encoding}"
        if format.encoding == "plain" and format.dtype != spec.dtype:
            return "query and unquantized KV dtypes must match"
        if format.group_size is not None and format.group_size != getattr(self, "quant_group_size", None):
            return "kernel quantization group size differs from the KV format"
        packing = {"int4": "adjacent_low_nibble_first_offset7", "int8": "signed_int8"}
        if format.encoding in packing and (format.packing != packing[format.encoding] or format.dtype != torch.int8):
            return "integer KV packing differs from the kernel encoding"
        return None

    def attention_metadata_key(self, spec, format, phase):
        return spec.shape_key, format

    def mla_support_error(self, spec, format, phase):
        if not self.supports_mla:
            return "provider does not implement MLA"
        if format.version != 1 or format.encoding != "mla_plain" or format.dtype != spec.dtype:
            return "unsupported MLA latent encoding or dtype"
        return None

    def for_attention_spec(self, spec):
        return self

    def reserve_runtime_resources(self):
        pass

    def runtime_memory_demands(self):
        return ()

    def workspace_memory_demands(self, key_name, workspace_size, lanes, dtype=torch.int8):
        from lightllm.common.attention_state.memory_plan import MemoryDemand

        device_id = get_current_device_id()
        return tuple(
            MemoryDemand(
                f"workspace:{key_name}:{workspace_size}:{dtype}:lane{lane}", "cuda", workspace_size * dtype.itemsize
            )
            for lane in range(lanes)
            if (device_id, key_name, workspace_size, dtype, lane) not in self._workspace_buffers
        )

    @staticmethod
    def get_gpu_workspace_buffer(
        key_name: str, workspace_size: int, dtype: torch.dtype = torch.int8, lane_id: int = 0
    ) -> torch.Tensor:
        """Return a process-local workspace shared by key name and CUDA device."""
        if not key_name:
            raise ValueError("workspace key_name must not be empty")
        if workspace_size <= 0:
            raise ValueError(f"workspace_size must be positive, got {workspace_size}")

        device_id = get_current_device_id()
        if lane_id < 0:
            raise ValueError("workspace lane must be nonnegative")
        buffer_key = (device_id, key_name, workspace_size, dtype, lane_id)
        with BaseAttBackend._workspace_buffer_lock:
            workspace_buffer = BaseAttBackend._workspace_buffers.get(buffer_key)
            if workspace_buffer is None:
                workspace_buffer = torch.empty(workspace_size, dtype=dtype, device=device_id)
                BaseAttBackend._workspace_buffers[buffer_key] = workspace_buffer
            return workspace_buffer

    def create_att_prefill_state(self, infer_state: "InferStateInfo") -> "BasePrefillAttState":
        raise NotImplementedError("not impl")

    def create_att_decode_state(self, infer_state: "InferStateInfo") -> "BaseDecodeAttState":
        raise NotImplementedError("not impl")

    def uses_dynamic_spec_verify_layout(self) -> bool:
        args = get_env_start_args()
        draft_step = self.model.mtp_manager.get_decode_draft_step(self.model.is_mtp_draft_model)
        is_main_model = not self.model.is_mtp_draft_model
        has_decode_draft_step = draft_step > 0
        dynamic_verify_enabled = args.mtp_dynamic_verify
        return is_main_model and has_decode_draft_step and dynamic_verify_enabled

    def uses_causal_attention(self) -> bool:
        args = get_env_start_args()
        is_parallel_block_draft = self.model.is_mtp_draft_model and args.mtp_mode in ("dspark", "dflash", "dflash2")
        return not is_parallel_block_draft


@dataclass
class AttControl:
    """
    prefill_att 和 decode_att 的入参，用于控制att backend 内部的行为, 选择正确的att 实现。
    """

    use_alibi: bool = False
    tp_alibi: torch.Tensor = None
    use_sliding_window: bool = False
    sliding_window: Tuple[int, int] = (-1, -1)
    use_att_sink: bool = False
    sink_weight: torch.Tensor = None
    logical_layer_id: Optional[int] = None
    kv_scales: Optional[torch.Tensor] = None
    q_scale: Optional[torch.Tensor] = None
    cpu_kv_scales: object = None


@dataclass
class BaseAttState(ABC):
    """Batch metadata with a weak reference to its owning InferStateInfo."""

    graph_buffer_names = None

    def prepare_for_capture(self):
        pass

    def copy_for_decode_cuda_graph(self, new_state: "BaseAttState"):
        """Copy tensor metadata into the buffers owned by the captured graph."""
        names = self.graph_buffer_names
        if names is None:
            names = tuple(name for name, value in vars(new_state).items() if isinstance(value, torch.Tensor))
        copies = []
        for name in names:
            old, new = getattr(self, name), getattr(new_state, name)
            if old is None and new is None:
                continue
            if (
                old is None
                or new is None
                or (old.shape, old.dtype, old.device, old.stride()) != (new.shape, new.dtype, new.device, new.stride())
            ):
                raise ValueError(f"graph metadata {name} changed layout; select another capture bucket")
            if old.data_ptr() != new.data_ptr():
                copies.append((old, new))
        for old, new in copies:
            old.copy_(new, non_blocking=True)

    backend: BaseAttBackend = None
    # InferStateInfo 已强引用本状态，这里反向使用弱引用，避免形成强引用环。
    # 否则状态及其 Tensor 需等待循环 GC 回收，可能延迟显存复用、增加显存峰值。
    # InferStateInfo 的生命周期由推理调用方或 CUDA Graph 缓存管理，不由 attention state 延长。
    _infer_state_weak_ref: Optional[weakref.ReferenceType["InferStateInfo"]] = None

    @property
    def infer_state(self) -> Optional["InferStateInfo"]:
        if self._infer_state_weak_ref is None:
            return None
        infer_state = self._infer_state_weak_ref()
        if infer_state is None:
            raise ReferenceError("attention state's InferStateInfo has been released")
        return infer_state

    def check_no_infer_state(self) -> None:
        """Reject InferStateInfo stored directly in any instance attribute.

        This shallow check does not inspect containers or callback closures.
        """
        # InferStateInfo imports attention states, so import it only when checking.
        from lightllm.common.basemodel.infer_struct import InferStateInfo

        for name, value in vars(self).items():
            if isinstance(value, InferStateInfo):
                raise RuntimeError(
                    f"{type(self).__name__}.{name} must not retain InferStateInfo; use _infer_state_weak_ref instead"
                )

    @abstractmethod
    def init_state(self) -> None:
        pass


@dataclass
class BasePrefillAttState(BaseAttState):
    @abstractmethod
    def prefill_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        raise NotImplementedError("not impl")


@dataclass
class BaseDecodeAttState(BaseAttState):
    @abstractmethod
    def decode_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        pass
