import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING, Tuple, Union, Dict

if TYPE_CHECKING:
    from lightllm.common.basemodel.basemodel import TpPartBaseModel
    from lightllm.common.basemodel.infer_struct import InferStateInfo


class BaseAttBackend:
    """
    用于创建支持各种不同的AttBackend, 如 fa3, flashinfer, triton 实现等。
    每个 model 复用一个 backend 实例。
    """

    _instances = {}

    def __new__(cls, *args, **kwargs):
        """
        Main 和 speculative draft model 可能使用不同的 CUDA graph 上限
        和缓存布局，不能只按 backend class 共享实例。
        """
        model = kwargs.get("model", args[0] if args else None)
        instance_key = (cls, id(model))
        if instance_key not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[instance_key] = instance
        return cls._instances[instance_key]

    def __init__(self, model: "TpPartBaseModel"):
        self.model = model

    def create_att_prefill_state(self) -> "BasePrefillAttState":
        raise NotImplementedError("not impl")

    def create_att_decode_state(self) -> "BaseDecodeAttState":
        raise NotImplementedError("not impl")

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
