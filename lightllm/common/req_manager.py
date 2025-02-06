import torch
from lightllm.utils.log_utils import init_logger
from .mem_manager import MemoryManager
from typing import List

logger = init_logger(__name__)


class _ReqNode:
    def __init__(self, index):
        self.index = index
        self.next: "_ReqNode" = None


class _ReqLinkedList:
    def __init__(self, max_request_num):
        self.nodes = [_ReqNode(i) for i in range(max_request_num)]
        self.marks = [0 for _ in range(max_request_num)]
        self.root_node = _ReqNode(-1)
        for i in range(0, max_request_num - 1):
            self.nodes[i].next = self.nodes[i + 1]
        self.root_node.next = self.nodes[0]
        self.can_alloc_size = max_request_num
        return

    def alloc(self):
        if self.root_node.next is None:
            logger.warning("alloc req index fail")
            return None
        get_node = self.root_node.next
        self.root_node.next = self.root_node.next.next
        assert self.marks[get_node.index] == 0
        self.marks[get_node.index] = 1
        self.can_alloc_size -= 1
        return get_node.index

    def free(self, index):
        assert self.marks[index] == 1
        node = self.nodes[index]
        node.next = self.root_node.next
        self.root_node.next = node
        self.marks[index] = 0
        self.can_alloc_size += 1
        return

    def is_all_free(self):
        return self.can_alloc_size == len(self.marks)


class ReqManager:
    def __init__(self, max_request_num, max_sequence_length, mem_manager: MemoryManager):
        # 这里对最大请求数量的管理在默认上多申请了一个，主要是 index 为 max_request_num 代表
        # 的这个请求管理 id， 主要是为了兼容 DP 运行模式下，让各个 DP 能 padding 到 DP 中最大
        # 的那个batch size 进行运行，所有 padding 的请求都会使用预留的这个请求管理 id 进行处理
        # 这样让 DP 的实现更为简化一些。
        self.req_list = _ReqLinkedList(max_request_num)
        self.req_to_token_indexs = torch.zeros(
            (max_request_num + 1, max_sequence_length), dtype=torch.int32, device="cuda"
        )
        self.mem_manager = mem_manager
        self.max_request_num = max_request_num
        self.HOLD_REQUEST_ID = max_request_num

    def alloc(self):
        return self.req_list.alloc()

    def free(self, free_req_indexes: List[int], free_token_index):
        for req_index in free_req_indexes:
            self.req_list.free(req_index)

        if self.req_list.is_all_free():
            logger.debug(f"freed all request size {self.req_list.can_alloc_size}")
        self.mem_manager.free(free_token_index)

    def free_req(self, free_req_index: int):
        self.req_list.free(free_req_index)
        if self.req_list.is_all_free():
            logger.debug(f"freed all request size {self.req_list.can_alloc_size}")
        return

    def free_token(self, free_token_index):
        self.mem_manager.free(free_token_index)
        return

    def free_all(self):
        self.req_list = _ReqLinkedList(self.max_request_num)
        return
