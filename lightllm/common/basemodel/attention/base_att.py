import threading

import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING, Tuple, Union, Dict

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

    def create_att_prefill_state(self) -> "BasePrefillAttState":
        raise NotImplementedError("not impl")

    def create_att_decode_state(self) -> "BaseDecodeAttState":
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
        is_parallel_block_draft = self.model.is_mtp_draft_model and args.mtp_mode in ("dspark", "dflash")
        return not is_parallel_block_draft

    def _find_layer_index(
        self, k: torch.Tensor, v: torch.Tensor, att_state: Union["BasePrefillAttState", "BaseDecodeAttState"]
    ) -> int:
        kv_buffer = att_state.infer_state.mem_manager.kv_buffer
        layer_count = len(kv_buffer)
        find_dict = {kv_buffer[i].data_ptr(): i for i in range(layer_count)}
        key = min(k.data_ptr(), v.data_ptr())
        assert key in find_dict
        return find_dict[key]


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


@dataclass
class BasePrefillAttState(ABC):

    backend: BaseAttBackend = None
    infer_state: "InferStateInfo" = None

    @abstractmethod
    def init_state(self):
        pass

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
class BaseDecodeAttState(ABC):
    backend: BaseAttBackend = None
    infer_state: "InferStateInfo" = None

    @abstractmethod
    def init_state(self):
        pass

    def copy_for_decode_cuda_graph(self, new_state: "BaseDecodeAttState"):
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
