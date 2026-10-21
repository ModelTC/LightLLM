from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

import torch

from .base import ReqManager


if TYPE_CHECKING:
    from lightllm.server.router.model_infer.infer_batch import InferReq


class HybridAttentionReqManager(ReqManager, ABC):
    """混合 attention 的请求运行态与大小页 checkpoint 管理接口。

    大小页沿同一虚拟 token 索引空间匹配前缀，full attention KV 保持 token 粒度存储。
    linear/sliding-window 状态在大页边界及请求可缓存尾部的小页边界保存 checkpoint，
    缓存命中后，再将相应 checkpoint 恢复到请求运行态。

    请求的 GPU 计算状态由具体实现管理；small_page_buffers 持有 CPU 小页快照；
    big_page_buffers 引用 mem_manager 的 CPU 大页快照，与 full KV 容量一起创建、调整。
    大小页不包含额外的 GPU 运行态，命中后恢复到请求状态，不重新分配 checkpoint 池。
    公共缓存流程负责 checkpoint 槽位分配、边界、匹配与淘汰；本接口负责运行态与 checkpoint 保存/恢复。
    """

    def __init__(self, max_request_num, max_sequence_length, mem_manager):
        super().__init__(max_request_num, max_sequence_length, mem_manager)
        self.small_page_buffers = None

    @property
    def big_page_buffers(self):
        return self.mem_manager.big_page_buffers

    @abstractmethod
    def create_small_page_cache_manager(self, size: int):
        """创建并持有 CPU 小页池，返回同一池供 radix 使用；size 是槽位数，不是 token 数。"""

    @abstractmethod
    def init_hybrid_attention_state(self, req: "InferReq"):
        """无前缀缓存命中时，初始化已分配请求槽位的 GPU 运行态。"""

    def restore_big_page_state(self, big_page_buffer_idx: int, req: "InferReq"):
        """将指定大页槽位的 CPU checkpoint 恢复到请求 GPU 运行态。"""
        self.restore_state(req, self.big_page_buffers, big_page_buffer_idx)

    def restore_small_page_state(self, req: "InferReq"):
        """将 req.shared_kv_node 对应的小页 checkpoint 恢复到请求 GPU 运行态。"""
        self.restore_state(req, self.small_page_buffers, req.shared_kv_node.small_page_buffer_idx)

    @abstractmethod
    def restore_state(self, req: "InferReq", state_cache_manager, buffer_idx: int):
        """CPU checkpoint → 请求 GPU 运行态；大小页共用，不负责前缀匹配或 full KV 索引恢复。"""

    def save_big_page_states(self, b_req_idx: torch.Tensor, req_indexes: List[int], buffer_indexes: List[int]):
        """批量保存请求 GPU 运行态到已分配的大页槽位，buffer_indexes 中的 -1 表示跳过。

        b_req_idx 与 req_indexes 分别为同一批请求的 GPU 索引张量和 CPU 索引列表。
        默认逐请求保存，模型可覆盖为批量拷贝算子。
        """
        for req_idx, buffer_idx in zip(req_indexes, buffer_indexes):
            if buffer_idx != -1:
                self.save_state(req_idx, buffer_idx, self.big_page_buffers)

    @abstractmethod
    def save_state(self, req_idx: int, buffer_idx: int, state_cache_manager):
        """请求 GPU 运行态 → 指定 CPU checkpoint 槽位；大小页共用，调用方负责分配槽位。"""

    def update_mtp_state(self, b_req_mtp_start_loc, b_req_idx, b_mtp_index, accepted_index, verify_width):
        """接受推测 token 后更新运行态位置；需要调整状态索引的模型覆写。"""
        return
