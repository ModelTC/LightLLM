import torch
    

class MemoryManager:
    def __init__(self, size, dtype, head_num, head_dim, layer_num, always_copy=False):
        self.size = size        
        self.dtype = dtype
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.always_copy = always_copy
        
        # mem_state 修改为使用计数方式，方便后期实现token共享机制，实现beam search 等
        self.mem_state = torch.zeros((size,), dtype=torch.int32, device="cuda")
        self.indexes = torch.arange(0, size, dtype=torch.long, device="cuda")
        self.can_use_mem_size = size
        self._init_buffers(size, dtype, head_num, head_dim, layer_num)
    
    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        self.key_buffer = [torch.empty((size, head_num, head_dim), dtype=dtype, device="cuda") for _ in range(layer_num)]
        self.value_buffer = [torch.empty((size, head_num, head_dim), dtype=dtype, device="cuda") for _ in range(layer_num)]
    
    def _free_buffers(self):
        self.key_buffer = None
        self.value_buffer = None
    
    @torch.no_grad()
    def alloc(self, need_size):
        if need_size > self.can_use_mem_size:
            print(f'warn no enough cache need_size {need_size} left_size {self.can_use_mem_size}')
            return None
        can_use_index = torch.nonzero(self.mem_state == 0).view(-1)
        select_index = can_use_index[0 : need_size]
        self.add_refs(select_index)
        return select_index
    
    @torch.no_grad()
    def alloc_contiguous(self, need_size):
        if self.always_copy:
            return None
        if need_size > self.can_use_mem_size:
            print(f'warn no enough cache need_size {need_size} left_size {self.can_use_mem_size}')
            return None
        
        can_use_index = torch.nonzero(self.mem_state == 0).view(-1)
        can_use_index_size = len(can_use_index)
        can_use_index = can_use_index[0 : can_use_index_size - need_size + 1][(can_use_index[need_size - 1: ] - can_use_index[0 : can_use_index_size - need_size + 1]) == need_size - 1]
        if can_use_index.shape[0] == 0:
            # print(f'warn no enough cache to contiguous need_size {need_size} left_size {self.can_use_mem_size}')
            return None
        start = can_use_index[0].item()
        end = start + need_size
        select_index = self.indexes[start : end]
        self.add_refs(select_index)
        return select_index, start, end
    
    @torch.no_grad()
    def free(self, free_index):
        """_summary_

        Args:
            free_index (torch.Tensor): _description_
        """
        free_index = free_index.long()
        self.decrease_refs(free_index)
        if self.can_use_mem_size == len(self.mem_state):
            print(f"freed all gpu mem size {self.can_use_mem_size}")
        return
    
    @torch.no_grad()
    def add_refs(self, token_index: torch.Tensor):
        state = self.mem_state[token_index]
        has_used_tokens = torch.count_nonzero(state).item()
        all_tokens = len(state)
        self.can_use_mem_size -= all_tokens - has_used_tokens
        self.mem_state[token_index] += 1
        return
    
    @torch.no_grad()
    def decrease_refs(self, token_index: torch.Tensor):
        self.mem_state[token_index] -= 1
        state = self.mem_state[token_index]
        used_tokens = torch.count_nonzero(state).item()
        all_tokens = len(state)
        self.can_use_mem_size += all_tokens - used_tokens
        return

    
    @torch.no_grad()
    def free_all(self):
        self.can_use_mem_size = len(self.mem_state)
        self.mem_state[:] = 0
    
    @torch.no_grad()
    def resize_mem(self, new_size):
        """
        just for test code
        """
        size = new_size        
        dtype = self.dtype
        head_num = self.head_num
        head_dim = self.head_dim
        layer_num = self.layer_num
        always_copy = self.always_copy

        self.mem_state = torch.zeros((size,), dtype=torch.int32, device="cuda")
        self.indexes = torch.arange(0, size, dtype=torch.long, device="cuda")
        self.can_use_mem_size = size
        self._free_buffers()
        self._init_buffers(size, dtype, head_num, head_dim, layer_num)
        return
