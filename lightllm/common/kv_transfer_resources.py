from dataclasses import dataclass
from multiprocessing.reduction import ForkingPickler
from typing import TYPE_CHECKING, Optional

import torch
import torch.multiprocessing as mp
from filelock import FileLock

from lightllm.utils.device_utils import kv_trans_use_p2p
from lightllm.utils.dist_utils import get_current_rank_in_node, get_node_world_size
from lightllm.utils.log_utils import init_logger
from lightllm.utils.shm_utils import create_or_link_shm, get_service_shm_name

if TYPE_CHECKING:
    from lightllm.common.linear_state_transfer import LinearAttentionStateTransfer
    from lightllm.common.deepseek4_cache_transfer import DeepseekV4KVTransferView, DeepseekV4StateTransferView

logger = init_logger(__name__)


@dataclass(frozen=True)
class KVTransferView:
    kv_buffer: torch.Tensor
    page_shape: Optional[tuple]
    page_dtype: torch.dtype
    replicated: bool

    @classmethod
    def from_mem_manager(cls, mem_manager, *, include_page_layout=False):
        return cls(
            mem_manager.kv_buffer,
            mem_manager.get_paged_kv_move_buffer_shape(1, 1)[2:] if include_page_layout else None,
            mem_manager.dtype,
            mem_manager.kv_is_replicated,
        )

    def create_page_io(self, kv_views, state_transfers, device_id, dp_world_size):
        from lightllm.server.router.model_infer.mode_backend.pd.cache_page_io import KVCachePageIO

        return KVCachePageIO(kv_views, state_transfers, device_id, dp_world_size)


@dataclass
class KVTransferResources:
    # Export GPU handles only; checkpoint pools must stay pinned in the inference process.
    kv: "KVTransferView | DeepseekV4KVTransferView"
    req_to_token_indexes: torch.Tensor
    state_transfer: Optional["LinearAttentionStateTransfer | DeepseekV4StateTransferView"] = None

    def write_to_shm(self):
        if kv_trans_use_p2p():
            from lightllm.server.router.model_infer.mode_backend.pd.p2p_fix import reduce_tensor

            mp.reductions.reduce_tensor.__code__ = reduce_tensor.__code__

        lock = FileLock(f"/tmp/{get_service_shm_name('mem_manager_lock')}")
        with lock:
            node_world_size = get_node_world_size()
            shm_name = f"mem_manager_{get_current_rank_in_node()}"
            obj_bytes_array = [ForkingPickler.dumps(self).tobytes() for _ in range(node_world_size * 2)]
            # IPC ref-counter offsets can change the serialized size between consumers.
            obj_size = max(map(len, obj_bytes_array))
            shm = create_or_link_shm(
                name=shm_name, expected_size=obj_size * (node_world_size * 2) + 4 + 4, force_mode="create"
            )
            logger.info(f"create shm {shm.name} size {shm.size} for mem manger shared buffer")
            shm.buf[0:4] = (node_world_size * 2).to_bytes(4, "little")
            shm.buf[4:8] = obj_size.to_bytes(4, "little")
            start_index = 8
            for obj_bytes in obj_bytes_array:
                shm.buf[start_index : start_index + obj_size] = obj_bytes.ljust(obj_size, b"\0")
                start_index += obj_size

    @staticmethod
    def loads_from_shm(rank_in_node: int):
        shm_name = f"mem_manager_{rank_in_node}"
        lock = FileLock(f"/tmp/{get_service_shm_name('mem_manager_lock')}")
        logger.info(f"get memmanager from shm {shm_name}")
        with lock:
            shm = create_or_link_shm(name=shm_name, expected_size=-1, force_mode="link")
            left_num = int.from_bytes(shm.buf[0:4], "little")
            obj_size = int.from_bytes(shm.buf[4:8], "little")
            assert left_num > 0
            end_index = 8 + left_num * obj_size
            start_index = 8 + (left_num - 1) * obj_size
            obj_bytes = shm.buf[start_index:end_index].tobytes()
            shm.buf[0:4] = (left_num - 1).to_bytes(4, byteorder="little")
            shm.close()
            return ForkingPickler.loads(obj_bytes)
