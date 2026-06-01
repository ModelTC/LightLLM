from lightllm.platform.base.ops import BackendOps, InferOps, WeightOps
from lightllm.platform.ops.weight.ascend import AscendWeightOps
from lightllm.platform.ops.infer.ascend import AscendInferOps


class AscendOps(BackendOps):
    def __init__(self) -> None:
        self._weight = AscendWeightOps()
        self._infer = AscendInferOps()

    @property
    def weight(self) -> WeightOps:
        return self._weight

    @property
    def infer(self) -> InferOps:
        return self._infer
