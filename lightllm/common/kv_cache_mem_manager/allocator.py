import torch
from lightllm.server.router.dynamic_prompt.shared_arr import SharedInt
from lightllm.utils.dist_utils import get_current_rank_in_node
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.utils.log_utils import init_logger
from typing import Union, List

logger = init_logger(__name__)


class KvCacheAllocator:
    def __init__(self, size: int, publish_usage: bool = True) -> None:
        self.size = size
        self.mem_state = torch.arange(
            0, self.size, dtype=torch.int32, device="cpu", requires_grad=False, pin_memory=True
        )
        self.mark_start = 0
        self.mark_end = self.size

        self._mem_state_return = torch.arange(
            0, self.size * 3, dtype=torch.int32, device="cpu", requires_grad=False, pin_memory=True
        )
        self._return_start = 0

        self.can_use_mem_size = self.size

        # Only the full KV pool publishes scheduler capacity; attention-state pools are private.
        self.shared_can_use_token_num = None
        if publish_usage:
            rank_in_node = get_current_rank_in_node()
            self.shared_can_use_token_num = SharedInt(
                f"{get_unique_server_name()}_mem_manger_can_use_token_num_{rank_in_node}"
            )
            self.shared_can_use_token_num.set_value(self.can_use_mem_size)
        return

    def alloc(self, need_size) -> torch.Tensor:
        if need_size > self.mark_end - self.mark_start:
            logger.error(f"warn no enough cache need_size {need_size} left_size {self.can_use_mem_size}")
            assert False, "error alloc state"

        start = self.mark_start
        end = self.mark_start + need_size
        self.mark_start += need_size

        self.can_use_mem_size -= need_size
        if self.shared_can_use_token_num is not None:
            self.shared_can_use_token_num.set_value(self.can_use_mem_size)

        # 利用缓冲区返回，避免异步情况下的内存竞争
        if self._return_start + need_size > self._mem_state_return.shape[0]:
            self._return_start = 0
        ans = self._mem_state_return[self._return_start : self._return_start + need_size]
        ans.copy_(self.mem_state[start:end])
        self._return_start += need_size
        return ans

    def free(self, free_index: Union[torch.Tensor, List[int]]):
        """_summary_

        Args:
            free_index (torch.Tensor): _description_
        """

        end = self.mark_start
        start = self.mark_start - len(free_index)
        assert start >= 0, f"error free state start: {self.mark_start} free len {len(free_index)}"

        if isinstance(free_index, list):
            self.mem_state.numpy()[start:end] = free_index
        else:
            # 从 gpu 到 cpu 的拷贝操作是流内阻塞操作
            self.mem_state[start:end] = free_index

        self.mark_start -= len(free_index)

        self.can_use_mem_size += len(free_index)
        if self.shared_can_use_token_num is not None:
            self.shared_can_use_token_num.set_value(self.can_use_mem_size)

        if self.can_use_mem_size == len(self.mem_state):
            logger.debug(f"freed all gpu mem size {self.can_use_mem_size}")
        return

    def free_all(self):
        self.mem_state.numpy()[:] = list(range(0, len(self.mem_state)))
        self.mark_start = 0
        self.mark_end = len(self.mem_state)
        self.can_use_mem_size = len(self.mem_state)
        if self.shared_can_use_token_num is not None:
            self.shared_can_use_token_num.set_value(self.can_use_mem_size)
        return

    def resize(self, new_size: int) -> None:
        """
        just for test code
        """
        self.size = new_size
        self.mem_state = torch.arange(
            0, self.size, dtype=torch.int32, device="cpu", requires_grad=False, pin_memory=True
        )
        self.mark_start = 0
        self.mark_end = self.size

        self._mem_state_return = torch.arange(
            0, self.size * 3, dtype=torch.int32, device="cpu", requires_grad=False, pin_memory=True
        )
        self._return_start = 0

        self.can_use_mem_size = self.size
        if self.shared_can_use_token_num is not None:
            self.shared_can_use_token_num.set_value(self.can_use_mem_size)
