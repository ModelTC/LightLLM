from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING, List, Optional

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

    ##### 管理器初始化与检查点缓存池 #####
    # 大页池由 mem_manager 持有，小页池由本管理器创建并提供给前缀缓存使用。

    def __init__(self, max_request_num, max_sequence_length, mem_manager):
        super().__init__(max_request_num, max_sequence_length, mem_manager)
        self.small_page_buffers = None

    @property
    def big_page_buffers(self):
        return self.mem_manager.big_page_buffers

    @abstractmethod
    def create_small_page_cache_manager(self, size: int):
        """创建并持有 CPU 小页池，返回同一池供 radix 使用；size 是槽位数，不是 token 数。"""

    ##### 请求运行态初始化 #####
    # 未命中前缀缓存时，初始化已分配请求槽位的 GPU 状态。

    @abstractmethod
    def init_hybrid_attention_state(self, req: "InferReq"):
        """无前缀缓存命中时，初始化已分配请求槽位的 GPU 运行态。"""

    ##### 检查点恢复：CPU 缓存 → 请求 GPU 运行态 #####
    # 大小页分别提供恢复入口，实际状态拷贝统一交给 restore_state 实现。

    def restore_big_page_state(self, big_page_buffer_idx: int, req: "InferReq", checkpoint_len: Optional[int] = None):
        """将指定大页槽位的 CPU checkpoint 恢复到请求 GPU 运行态。"""
        self.restore_state(req, self.big_page_buffers, big_page_buffer_idx, checkpoint_len)

    def restore_small_page_state(self, req: "InferReq", checkpoint_len: Optional[int] = None):
        """将 req.shared_kv_node 对应的小页 checkpoint 恢复到请求 GPU 运行态。"""
        self.restore_state(req, self.small_page_buffers, req.shared_kv_node.small_page_buffer_idx, checkpoint_len)

    @abstractmethod
    def restore_state(self, req: "InferReq", state_cache_manager, buffer_idx: int, checkpoint_len: Optional[int] = None):
        """CPU checkpoint → 请求 GPU 运行态；大小页共用，不负责前缀匹配或 full KV 索引恢复。"""

    ##### 检查点保存：请求 GPU 运行态 → CPU 缓存 #####
    # 大页批量保存、小页逐请求保存；两类入口共用 save_state，检查点槽位由调用方分配。

    def save_big_page_states(self, b_req_idx: torch.Tensor, req_indexes: List[int], buffer_indexes: List[int], checkpoint_lens: Optional[List[int]] = None):
        """批量保存请求 GPU 运行态到已分配的大页槽位，buffer_indexes 中的 -1 表示跳过。

        b_req_idx 与 req_indexes 分别为同一批请求的 GPU 索引张量和 CPU 索引列表。
        默认逐请求保存，模型可覆盖为批量拷贝算子。
        """
        lengths = checkpoint_lens if checkpoint_lens is not None else [None] * len(req_indexes)
        for req_idx, buffer_idx, length in zip(req_indexes, buffer_indexes, lengths):
            if buffer_idx != -1:
                self.save_state(req_idx, buffer_idx, self.big_page_buffers, length)

    def save_small_page_state(self, req_idx: int, buffer_idx: int, checkpoint_len: Optional[int] = None):
        """将请求 GPU 运行态保存到已分配的小页槽位。"""
        self.save_state(req_idx, buffer_idx, self.small_page_buffers, checkpoint_len)

    @abstractmethod
    def save_state(self, req_idx: int, buffer_idx: int, state_cache_manager, checkpoint_len: Optional[int] = None):
        """请求 GPU 运行态 → 指定 CPU checkpoint 槽位；大小页共用，调用方负责分配槽位。"""

    ##### MTP 验证后的运行态更新 #####
    # 根据本轮接受的位置更新请求状态，供后续解码及检查点保存使用。

    def update_mtp_state(
        self, b_req_mtp_start_loc, b_req_idx, b_mtp_index, accepted_index, verify_width, linear_state_batches=None
    ):
        """接受推测 token 后更新运行态位置；需要调整状态索引的模型覆写。"""
        return

    ##### PD 分离：传输块与请求运行态的读写 #####
    # P 节点将最终接受的请求状态写入传输块，D 节点从收到的传输块恢复请求运行态。
    # 模型状态布局、TP 分片及实际拷贝由接口内部处理，调用方只传递块 tensor 和 rank。

    @contextmanager
    def pd_sharing_context(self):
        """序列化前临时移除推理进程专用资源，退出时恢复，异常时也保持原对象完整。

        去掉内存管理器回指、CPU 小页池、请求分配器和采样参数管理器，
        避免将 pinned CPU 缓冲区或无关资源带入 PD 传输进程。
        """
        removed = {}
        try:
            for name in ("mem_manager", "small_page_buffers", "req_list", "req_sampling_params_manager"):
                if name in self.__dict__:
                    removed[name] = self.__dict__.pop(name)
            yield
        finally:
            self.__dict__.update(removed)

    def get_pd_state_size(self) -> int:
        """Return the linear-state transfer size; other hybrid backends use their own protocol."""
        raise NotImplementedError

    def get_pd_state(self, req_idx: int, state_block: torch.Tensor, tp_rank: int):
        """将请求最终接受的状态拷贝到传输块，内部完成模型状态打包和本 rank 分片。

        传输块由调用方提供；异步拷贝完成前，不得推进或释放请求，也不得复用传输块。
        """
        raise NotImplementedError

    def load_pd_state(self, req_idx: int, state_block: torch.Tensor, tp_rank: int):
        """从传输块恢复请求运行态，内部完成解析、TP 分片、拷贝和旧状态跟踪重置。

        拷贝在当前 CUDA stream 上执行，调用方沿用 PD 完成事件协调后续推理和块复用。
        """
        raise NotImplementedError
