from typing import TYPE_CHECKING, List

from .base import PrefillQueueStrategy

if TYPE_CHECKING:
    from lightllm.server.router.model_infer.infer_batch import InferReq


def remaining_prefill_tokens(req: "InferReq") -> int:
    return max(0, req.shm_req.input_len - req.cur_kv_len)


def infer_priority(req: "InferReq") -> int:
    return req.shm_req.sample_params.infer_high_priority


class FCFSStrategy(PrefillQueueStrategy):
    """按 prefill 推理优先级排队；优先级相同时保持 FCFS 顺序。"""

    def reorder_prefill(self, prefill_reqs: List["InferReq"]) -> List["InferReq"]:
        # 这里依赖 Python 的 sorted 为稳定排序，确保相同优先级的请求保持 FCFS 顺序。
        return sorted(prefill_reqs, key=infer_priority)


class PromoteShortestPrefillStrategy(PrefillQueueStrategy):
    """在普通 prefill 请求中，只将剩余 token 最少的一个请求提升到普通队列头部。

    负优先级请求按原始相对顺序排在前面，非负优先级请求视为普通请求。
    除被提升的最短请求外，其他普通请求保持原始相对顺序。
    prompt 已处理完成的请求按零个剩余 token 计算。
    """

    def reorder_prefill(self, prefill_reqs: List["InferReq"]) -> List["InferReq"]:
        high_priority_reqs = [req for req in prefill_reqs if infer_priority(req) < 0]
        normal_reqs = [req for req in prefill_reqs if infer_priority(req) >= 0]
        if not normal_reqs:
            return high_priority_reqs

        shortest_req = min(normal_reqs, key=remaining_prefill_tokens)
        normal_reqs.remove(shortest_req)
        high_priority_reqs.append(shortest_req)
        high_priority_reqs.extend(normal_reqs)
        return high_priority_reqs


class HRRNPrefillStrategy(PrefillQueueStrategy):
    """使用基于 token aging 的最高响应比优先策略排列普通 prefill 请求。

    负优先级请求仍按原始相对顺序排在最前面。普通请求按
    ``waited_prefill_tokens / uncached_prefill_tokens`` 从大到小排列；等待越久或
    首次参与 HRRN 排序时的未缓存工作量越小，越早获得调度。请求 ID 用作相同比率下的
    确定性次级排序键。
    """

    def reorder_prefill(self, prefill_reqs: List["InferReq"]) -> List["InferReq"]:
        high_priority_reqs = [req for req in prefill_reqs if infer_priority(req) < 0]
        normal_reqs = [req for req in prefill_reqs if infer_priority(req) >= 0]
        processed_prefill_tokens = self.backend.processed_prefill_tokens

        def hrrn_key(req: "InferReq"):
            # 请求第一次进入 HRRN 排序时才绑定服务成本，使此前完成的 Cache 初始化
            # 能反映在分母中；后续 chunked prefill 不再改变该请求的服务成本。
            if not hasattr(req, "uncached_prefill_tokens"):
                req.uncached_prefill_tokens = max(1, remaining_prefill_tokens(req))
            uncached_prefill_tokens = req.uncached_prefill_tokens
            waited_tokens = max(0, processed_prefill_tokens - req.arrival_processed_prefill_tokens)
            return (-waited_tokens / uncached_prefill_tokens, req.req_id)

        normal_reqs.sort(key=hrrn_key)
        high_priority_reqs.extend(normal_reqs)
        return high_priority_reqs
