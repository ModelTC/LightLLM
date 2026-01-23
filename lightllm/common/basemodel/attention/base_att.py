import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING, Tuple, Union, Dict

if TYPE_CHECKING:
    from lightllm.common.basemodel.basemodel import TpPartBaseModel
    from lightllm.common.basemodel.infer_struct import InferStateInfo


class BaseAttBackend:
    """
    用于创建支持各种不同的AttBackend, 如 fa3, flashinfer, triton 实现等，
    这个是单列模式, 每种backend只有一个实例
    """

    _instances = {}

    def __new__(cls, *args, **kwargs):
        """
        重写__new__方法实现单例模式
        """
        # 检查是否已经有该类的实例
        if cls not in cls._instances:
            # 创建新实例并存储
            instance = super().__new__(cls)
            cls._instances[cls] = instance
        # 返回已有的实例
        return cls._instances[cls]

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
