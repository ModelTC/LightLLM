from abc import ABC, abstractmethod
from lightllm.platform.base.ops.infer_ops import InferOps
from lightllm.platform.base.ops.weight_ops import WeightOps


class BackendOps(ABC):

    @property
    @abstractmethod
    def weight(self) -> WeightOps:
        pass

    @property
    @abstractmethod
    def infer(self) -> InferOps:
        pass
