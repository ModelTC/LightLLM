from lightllm.platform.base.ops import BackendOps, InferOps, WeightOps
from lightllm.platform.ops.weight.cuda import CudaWeightOps
from lightllm.platform.ops.infer.cuda import CudaInferOps


class CudaOps(BackendOps):
    def __init__(self) -> None:
        self._weight = CudaWeightOps()
        self._infer = CudaInferOps()

    @property
    def weight(self) -> WeightOps:
        return self._weight

    @property
    def infer(self) -> InferOps:
        return self._infer
