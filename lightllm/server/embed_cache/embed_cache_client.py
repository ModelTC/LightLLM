import torch
from typing import Optional
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger
from lightllm.utils.embed_utils import calcu_embed_cache_meta
from lightllm.common.cpu_cache import CpuCacheCreator, CpuCacheTensorSpec
from .allocator import MemoryBlock, MemoryManager
from .copy_to_cache import offload_embed_tensor_to_cache

logger = init_logger(__name__)


class CpuEmbedCacheClient(object):
    """
    This class is responsible for handling cpu kv cache meta data.
    """

    def __init__(self, create_meta_data: bool, init_shm_data: bool):
        self.args = get_env_start_args()
        # to do here need calcu from from settings.
        self.embed_cache_tensor_meta = calcu_embed_cache_meta()
        self.token_num: int = self.embed_cache_tensor_meta.token_num

        if create_meta_data:
            self.token_index_manager = MemoryManager(total_size=self.token_num)

        cache_tensor_spec = CpuCacheTensorSpec(
            shm_key=self.args.multi_modal_cache_shm_id,
            shape=(
                self.embed_cache_tensor_meta.token_num,
                self.embed_cache_tensor_meta.layer_num,
                self.embed_cache_tensor_meta.hidden_size,
            ),
            dtype=self.embed_cache_tensor_meta.data_type,
            size_bytes=self.embed_cache_tensor_meta.calcu_size(),
        )
        cache_tensor_creator = CpuCacheCreator(tensor_spec=cache_tensor_spec)
        self.cpu_embed_cache_tensor, _ = cache_tensor_creator.create_or_attach(
            init_shm_data=init_shm_data,
            pin=not init_shm_data,
            pin_no_blocking=False,
        )
        return

    def alloc_indexes(self, token_num: int) -> Optional["MemoryBlock"]:
        return self.token_index_manager.alloc(need_size=token_num)

    def release_indexes(self, block: "MemoryBlock"):
        self.token_index_manager.release(block)
        return

    def copy_to_cache(self, embed_tensor: torch.Tensor, start_index_in_cache: int):
        offload_embed_tensor_to_cache(
            embed_tensor=embed_tensor,
            cache_tensor=self.cpu_embed_cache_tensor,
            start_index_in_cache=start_index_in_cache,
        )
        return

    def copy_vision_to_cache(self, embed_tensor: torch.Tensor, start_index_in_cache: int):
        if embed_tensor.ndim == 3:
            # check for qwen3 vision embed tensor shape, use apply deepstack
            assert embed_tensor.shape[1] == self.cpu_embed_cache_tensor.shape[1]

        offload_embed_tensor_to_cache(
            embed_tensor=embed_tensor,
            cache_tensor=self.cpu_embed_cache_tensor,
            start_index_in_cache=start_index_in_cache,
        )
        return

        return


if __name__ == "__main__":
    mem = MemoryManager(total_size=2000)

    import random

    alloced_list = []
    for i in range(20000):
        if random.randint(0, 3) > 1:
            o = mem.alloc(need_size=random.randint(1, 100))
            if o is not None:
                alloced_list.append(o)
        else:
            if len(alloced_list) > 0:
                index = random.randint(0, len(alloced_list) - 1)
                mem.release(alloced_list[index])
                del alloced_list[index]

    for e in alloced_list:
        mem.release(e)

    print(mem.mem_set_by_start)
    assert len(mem.mem_set_by_size) == len(mem.mem_set_by_start)
    assert len(mem.mem_set_by_size) == 1
    print(mem.mem_set_by_size[0])
