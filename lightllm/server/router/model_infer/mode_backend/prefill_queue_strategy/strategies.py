from typing import TYPE_CHECKING, List

from .base import PrefillQueueStrategy

if TYPE_CHECKING:
    from lightllm.server.router.model_infer.infer_batch import InferReq


def remaining_prefill_tokens(req: "InferReq") -> int:
    return max(0, req.shm_req.input_len - req.cur_kv_len)


def infer_priority(req: "InferReq") -> int:
    # 默认为 0；PD Master 在 Decode 容量不足、重新发起第二段及后续生成时设为 -1。
    # 这是生成分段的内部优先级，普通 chunked prefill 续算不会因此自动变为 -1。
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
    首次参与 HRRN 排序时需要计算的 token 越少，越早获得调度。请求 ID 用作相同比率下的
    确定性次级排序键。PD Prefill 已发出 KV 页时，还按发页后的等待量和下一段成本排序。
    """

    def reorder_prefill(self, prefill_reqs: List["InferReq"]) -> List["InferReq"]:
        high_priority_reqs = [req for req in prefill_reqs if infer_priority(req) < 0]
        normal_reqs = [req for req in prefill_reqs if infer_priority(req) >= 0]
        processed_prefill_tokens = self.backend.processed_prefill_tokens

        def hrrn_key(req: "InferReq"):
            # 首次参与 HRRN 普通请求排序时记录 aging 起点，后续轮次保留该快照。
            if not hasattr(req, "arrival_processed_prefill_tokens"):
                req.arrival_processed_prefill_tokens = processed_prefill_tokens
            # 首次排序时按模型实际计算起点估算成本，计入短续段需要重算的缓存窗口。
            # 后续 chunked prefill 不再改变该请求的服务成本。
            if not hasattr(req, "uncached_prefill_tokens"):
                prompt_len = req.shm_req.input_len
                prefill_start = req.get_prefill_input_start(prompt_len) if req.cur_kv_len < prompt_len else prompt_len
                req.uncached_prefill_tokens = max(1, prompt_len - prefill_start)
            uncached_prefill_tokens = req.uncached_prefill_tokens
            waited_tokens = max(0, processed_prefill_tokens - req.arrival_processed_prefill_tokens)
            score = waited_tokens / uncached_prefill_tokens
            last_page_tokens = req.pd_last_page_processed_tokens
            if last_page_tokens is not None:
                remaining = req.shm_req.input_len - req.cur_kv_len
                if remaining > 0:
                    chunk_cost = min(self.backend.args.chunked_prefill_size, remaining)
                    score = max(score, (processed_prefill_tokens - last_page_tokens) / chunk_cost)
            return (-score, req.req_id)

        normal_reqs.sort(key=hrrn_key)
        high_priority_reqs.extend(normal_reqs)
        return high_priority_reqs


class HRRNResumePrefillStrategy(HRRNPrefillStrategy):
    """在 HRRN 基础上，周期性将一个过久未发 KV 页的 PD prefill 续算请求移到普通队列头部。

    只调整排队顺序，不修改 infer_high_priority 或 chunk 大小；实际执行仍受后续预算和容量检查约束。
    """

    def __init__(self, backend):
        super().__init__(backend)
        # 下次允许额外提权的累计 prefill token 门槛；初始为 0，首次检查无需等待冷却。
        # 提权后会更新为当前累计值加间隔，并不是一直为 0。
        self.next_resume_priority_tokens = 0

    def reorder_prefill(self, prefill_reqs: List["InferReq"]) -> List["InferReq"]:
        # 先保留父类的高优先级分组及 HRRN 排序，再尝试额外提升一个普通请求。
        ordered = super().reorder_prefill(prefill_reqs)
        # 本后端累计提交执行的 prefill token 数，用作工作量时钟，不是墙上时间。
        processed = self.backend.processed_prefill_tokens
        if processed < self.next_resume_priority_tokens:
            return ordered

        # 提权最多到普通队列头部，不能越过 infer_high_priority < 0 的内部高优先级请求。
        first_normal = next((index for index, req in enumerate(ordered) if infer_priority(req) >= 0), len(ordered))
        # 只考虑发过 KV 页且 prefill 尚未完成的普通请求；仅命中缓存不代表发过页。
        # pd_last_page_processed_tokens 是最近发布 KV 页任务时的累计 token 数，不是传输完成时间。
        # 该值越小，距上次发页越久；相同时按请求 ID、队列位置确定顺序。
        oldest = min(
            (
                (req.pd_last_page_processed_tokens, req.req_id, index)
                for index, req in enumerate(ordered)
                if index >= first_normal
                and req.pd_last_page_processed_tokens is not None
                and 0 < req.cur_kv_len < req.shm_req.input_len
            ),
            default=None,
        )
        if oldest is not None:
            last_page_tokens, _, index = oldest
            # 用四个 batch token 上限的工作量同时限制请求等待门槛和相邻两次提权的间隔。
            # 这不等于固定经过四轮调度：每轮实际提交的 token 数可能不足 batch_max_tokens。
            interval = 4 * self.backend.batch_max_tokens
            if processed - last_page_tokens >= interval:
                ordered.insert(first_normal, ordered.pop(index))
                # 本次调整顺序后就进入冷却，不等待请求实际执行；没有合格候选时不更新门槛。
                self.next_resume_priority_tokens = processed + interval
        return ordered
