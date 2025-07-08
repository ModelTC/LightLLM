import os
import pickle
from multiprocessing import shared_memory
from lightllm.server.core.objs.atomic_lock import AtomicShmLock
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.utils.log_utils import init_logger

LIGHTLLM_REQS_BUFFER_BYTE_SIZE = int(os.getenv("LIGHTLLM_REQS_BUFFER_BYTE_SIZE", 64 * 1024 * 1024))  # 默认64M buf

logger = init_logger(__name__)


class ShmReqsBuffer:
    def __init__(self):
        self.args = get_env_start_args()
        self.name = f"{get_unique_server_name()}_ShmReqsBufferParams"
        self.lock = AtomicShmLock(lock_name=f"{get_unique_server_name()}_ShmReqsBufferParams_atomlock")
        self._create_or_link_shm()
        self.node_world_size = self.args.tp // self.args.nnodes

    def set_ready(self):
        with self.lock:
            assert self.int_view[0] == 0
            self.int_view[0] = self.node_world_size
        return

    def sub_state(self):
        with self.lock:
            cur_value = self.int_view[0]
            assert cur_value > 0
            self.int_view[0] = cur_value - 1
        return

    def is_empty(self):
        with self.lock:
            return self.int_view[0] == 0

    def is_ready(self):
        with self.lock:
            return self.int_view[0] == self.node_world_size

    def write_obj(self, obj):
        obj_bytes = pickle.dumps(obj)
        self.int_view[1] = len(obj_bytes)
        self.shm.buf[8 : 8 + len(obj_bytes)] = obj_bytes
        return

    def read_obj(self):
        bytes_len = self.int_view[1]
        obj = pickle.loads(self.shm.buf[8 : 8 + bytes_len])
        return obj

    def _create_or_link_shm(self):
        try:
            shm = shared_memory.SharedMemory(name=self.name, create=True, size=LIGHTLLM_REQS_BUFFER_BYTE_SIZE)
        except:
            shm = shared_memory.SharedMemory(name=self.name, create=False, size=LIGHTLLM_REQS_BUFFER_BYTE_SIZE)

        if shm.size != LIGHTLLM_REQS_BUFFER_BYTE_SIZE:
            logger.warning(f"size not same, unlink shm {self.name} and create again")
            shm.close()
            shm.unlink()
            try:
                shm = shared_memory.SharedMemory(name=self.name, create=True, size=LIGHTLLM_REQS_BUFFER_BYTE_SIZE)
                logger.info(f"create shm {self.name}")
            except:
                shm = shared_memory.SharedMemory(name=self.name, create=False, size=LIGHTLLM_REQS_BUFFER_BYTE_SIZE)
                logger.info(f"link shm {self.name}")

        self.shm = shm
        self.int_view = self.shm.buf.cast("i")
        # 前4个字节是特殊的计数用途，router写入后，被各个推理进程在拿去所有数据后，减1后归0
        self.int_view[0] = 0
        return
