from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from lightllm.server.router.model_infer.infer_batch import InferReq
    from ..base_backend import ModeBackend


class PrefillQueueStrategy(ABC):
    """在请求分类和 token 预算分配前调整就绪请求的排队顺序。

    输入中可能包含 decode、暂停或已结束的请求。基类先把 decode 请求稳定地放到最左侧，
    子类只负责排序右侧的 prefill 请求。实现必须返回原请求集合的一个排列；除策略自身的惰性
    调度元数据外，不能修改请求状态、丢弃请求或原地修改输入列表。各 TP rank 的排序结果必须
    一致；除非策略明确提供次级排序条件，否则应保持相同排序键请求的原始顺序。
    策略可通过 ``self.backend`` 读取已完成初始化的模型、缓存、rank 等调度状态。
    """

    def __init__(self, backend: "ModeBackend") -> None:
        self.backend = backend

    def reorder(
        self,
        ready_reqs: List["InferReq"],
        no_decode: bool = False,
        strict_prefill: bool = False,
    ) -> List["InferReq"]:
        decode_reqs = []
        prefill_reqs = []
        for req in ready_reqs:
            if self.backend._is_decode_req(req, no_decode=no_decode, strict_prefill=strict_prefill):
                decode_reqs.append(req)
            else:
                prefill_reqs.append(req)

        return decode_reqs + self.reorder_prefill(prefill_reqs)

    @abstractmethod
    def reorder_prefill(self, prefill_reqs: List["InferReq"]) -> List["InferReq"]:
        raise NotImplementedError
