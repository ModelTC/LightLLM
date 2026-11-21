import threading
import weakref

import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING, Tuple, Dict

from lightllm.common.attention.backends.gqa import GQAKVScales
from lightllm.common.attention.backend import AttentionBackend, AttentionImpl, AttentionBatch
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.utils.envs_utils import get_env_start_args

if TYPE_CHECKING:
    from lightllm.common.basemodel.basemodel import TpPartBaseModel
    from lightllm.common.basemodel.infer_struct import InferStateInfo


class BaseAttBackend:
    """
    用于创建支持各种不同的AttBackend, 如 fa3, flashinfer, triton 实现等。
    每个 model 复用一个 backend 实例。

    这是模型侧的 adapter：把 InferStateInfo 和 AttControl 转换为
    lightllm.common.attention.backend 中的 AttentionBatch、layer_config 与 inputs，
    再调用其中的 AttentionStepBuilder 和 AttentionImpl；后者的 AttentionBackend
    只描述某个实现的 step builder 与 impl 类，与本类不是同一层。
    """

    requires_cpu_cache_sync = False

    _instances = {}
    _workspace_buffers = {}
    _workspace_buffer_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """
        Main 和 speculative draft model 可能使用不同的 CUDA graph 上限
        和缓存布局，不能只按 backend class 共享实例。
        """
        model = kwargs.get("model", args[0] if args else None)
        instance_key = (cls, model)
        if instance_key not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[instance_key] = instance
        return cls._instances[instance_key]

    def __init__(self, model: "TpPartBaseModel"):
        self.model = model
        self._attention_impls: Dict[tuple, AttentionImpl] = {}

    def get_attention_impl(self, backend: type[AttentionBackend], layer_config) -> AttentionImpl:
        """Return the impl of backend for layers with this layer_config, creating it on first use.

        An impl holds only static layer configuration, so all layers with an equal layer_config
        share one instance.
        """
        key = (backend, layer_config)
        impl = self._attention_impls.get(key)
        if impl is None:
            impl = self._attention_impls[key] = backend.get_impl_cls()(layer_config)
        return impl

    @staticmethod
    def get_gpu_workspace_buffer(key_name: str, workspace_size: int, dtype: torch.dtype = torch.int8) -> torch.Tensor:
        """Return a process-local workspace shared by key name and CUDA device."""
        if not key_name:
            raise ValueError("workspace key_name must not be empty")
        if workspace_size <= 0:
            raise ValueError(f"workspace_size must be positive, got {workspace_size}")

        device_id = get_current_device_id()
        buffer_key = (device_id, key_name, workspace_size, dtype)
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
        # Only parallel block-draft models attend non-causally within a block.
        if not self.model.is_mtp_draft_model:
            return True
        return get_env_start_args().mtp_mode not in ("dspark", "dflash", "dflash2")


def build_attention_batch(infer_state: "InferStateInfo", backend: BaseAttBackend) -> AttentionBatch:
    """Describe one step's requests for attention step builders, from its InferStateInfo.

    CUDA graph capture and replay steps carry the model's graph kv length capacity, so step builders
    can size and configure captured work for every length a replay may see. Decode steps of
    speculative decoding describe how query rows group into requests. Fields the step does not
    set are None.
    """
    model = backend.model
    uses_cuda_graph = infer_state.is_cuda_graph_capture or infer_state.is_cuda_graph_replay
    decode_rows_per_request, dynamic_decode_rows = 1, False
    if not infer_state.is_prefill and getattr(model, "mtp_manager", None) is not None:
        decode_rows_per_request = model.mtp_manager.get_decode_draft_step(model.is_mtp_draft_model) + 1
        dynamic_decode_rows = decode_rows_per_request > 1 and backend.uses_dynamic_spec_verify_layout()
    return AttentionBatch(
        req_to_token_indexs=infer_state.req_manager.req_to_token_indexs,
        b_req_idx=infer_state.b_req_idx,
        b_seq_len=infer_state.b_seq_len,
        batch_size=infer_state.batch_size,
        max_kv_seq_len=infer_state.max_kv_seq_len,
        total_token_num=infer_state.total_token_num,
        cu_q_seq_len=infer_state.b1_cu_q_seq_len,
        cu_kv_seq_len=infer_state.b1_cu_kv_seq_len,
        b_q_start_loc=infer_state.b_q_start_loc,
        b_ready_cache_len=infer_state.b_ready_cache_len,
        max_q_seq_len=infer_state.max_q_seq_len if infer_state.max_q_seq_len is not None else 1,
        decode_rows_per_request=decode_rows_per_request,
        dynamic_decode_rows=dynamic_decode_rows,
        pad_req_idx=getattr(infer_state.req_manager, "HOLD_REQUEST_ID", None),
        b_shared_seq_len=getattr(infer_state, "b_shared_seq_len", None),
        b_image_token_end=getattr(infer_state, "b_image_token_end", None),
        b_shared_radix_node_id=getattr(infer_state, "b_shared_radix_node_id", None),
        cuda_graph_max_kv_seq_len=model.graph_max_len_in_batch if uses_cuda_graph else None,
        microbatch_index=infer_state.microbatch_index,
    )


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
    # mla 专用传参项
    mla_prefill: bool = False
    mla_prefill_dict: Dict = None
    mla_decode: bool = False
    mla_decode_dict: Dict = None
    # nsa (native sparse attention) 专用传参项
    nsa_prefill: bool = False
    nsa_prefill_dict: Dict = None
    nsa_decode: bool = False
    nsa_decode_dict: Dict = None
    # linear attention 专用传参项
    linear_att_prefill: bool = False
    linear_att_prefill_dict: Dict = None
    linear_att_decode: bool = False
    linear_att_decode_dict: Dict = None
    # 静态 FP8 标定按此逻辑层号查询 State 的物理行；其他后端可不使用。
    logical_layer_id: Optional[int] = None
    # 量化 KV cache 的本层 scale，由模型层从 attention State 取得（get_layer_kv_scales）；未量化时为 None。
    kv_scales: Optional["GQAKVScales"] = None
    # 本层在图像片段内做双向 attention（片段由 InferStateInfo.b_image_token_end 给出），仅用于 prefill。
    image_spans: bool = False


@dataclass
class BaseAttState(ABC):
    """Attention metadata with a weak reference to its owning InferStateInfo."""

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
    def copy_for_decode_cuda_graph(self, new_state: "BaseDecodeAttState"):
        """Copy tensor metadata into the buffers owned by the captured graph."""
        for attr_name, attr_value in vars(new_state).items():
            if isinstance(attr_value, torch.Tensor):
                attr_ = getattr(self, attr_name, None)
                if attr_ is not None and attr_.data_ptr() != attr_value.data_ptr():
                    attr_.copy_(attr_value, non_blocking=True)

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
