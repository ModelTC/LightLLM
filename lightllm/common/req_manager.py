import torch
from lightllm.utils.log_utils import init_logger
from .mem_manager import MemoryManager

logger = init_logger(__name__)


class ReqManager:
    def __init__(self, max_request_num, max_sequence_length, mem_manager: MemoryManager):
        # 这里对最大请求数量的管理在默认上多申请了一个，主要是 index 为 max_request_num 代表
        # 的这个请求管理 id， 主要是为了兼容 DP 运行模式下，让各个 DP 能 padding 到 DP 中最大
        # 的那个batch size 进行运行，所有 padding 的请求都会使用预留的这个请求管理 id 进行处理
        # 这样让 DP 的实现更为简化一些。
        self.req_state = torch.zeros((max_request_num + 1,), dtype=torch.bool, device="cuda")
        self.req_to_token_indexs = torch.zeros(
            (max_request_num + 1, max_sequence_length), dtype=torch.int32, device="cuda"
        )
        self.req_state[-1] = 1
        self.can_use_req_size = max_request_num
        self.mem_manager = mem_manager
        self.HOLD_REQUEST_ID = max_request_num

    def alloc(self, need_size):
        if need_size > self.can_use_req_size:
            logger.error(f"Insufficient requested capacity, remaining {self.can_use_req_size}")
            return None
        select_index = torch.nonzero(self.req_state == 0).reshape(-1)[:need_size]
        self.req_state[select_index] = 1
        self.can_use_req_size -= len(select_index)
        return select_index

    def free(self, free_req_index, free_token_index):
        self.can_use_req_size += len(free_req_index)
        self.req_state[free_req_index] = 0
        if self.can_use_req_size + 1 == len(self.req_state):
            logger.debug(f"freed all request size {self.can_use_req_size}")
        self.mem_manager.free(free_token_index)

    def free_req(self, free_req_index):
        self.can_use_req_size += 1
        self.req_state[free_req_index] = 0
        return

    def free_token(self, free_token_index):
        self.mem_manager.free(free_token_index)

    def free_all(self):
        self.can_use_req_size = len(self.req_state) - 1
        self.req_state[:-1] = 0
