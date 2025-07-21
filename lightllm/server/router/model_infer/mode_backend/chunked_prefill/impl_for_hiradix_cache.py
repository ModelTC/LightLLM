import torch
from typing import List, Tuple
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
from lightllm.utils.log_utils import init_logger
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.server.router.model_infer.mode_backend.generic_post_process import sample
from lightllm.server.router.model_infer.mode_backend.pre import (
    prepare_prefill_inputs,
    prepare_decode_inputs,
)
import torch.multiprocessing as mp
from .impl import ChunkedPrefillBackend


logger = init_logger(__name__)

class ChunkedPrefillBackendHiCache(ChunkedPrefillBackend):

    def __init__(self, radix_mem_queue: mp.Queue) -> None:
        super().__init__()
        self.radix_mem_queue = radix_mem_queue

    def init_custom(self):
        self.radix_mem_queue.put((self.model.mem_propties, self.model.shared_mem_data))