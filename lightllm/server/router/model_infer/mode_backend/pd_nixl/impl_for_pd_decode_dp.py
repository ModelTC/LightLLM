from types import MethodType
import torch.multiprocessing as mp
from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import get_env_start_args

from .impl_for_pd_base import PDNIXLBackendBaseDPChunked
from .impl_for_pd_decode import PDNIXLBackendForDecodeNode

logger = init_logger(__name__)


class PDNIXLDPBackendForDecodeNode(PDNIXLBackendBaseDPChunked):
    def __init__(self, prefill_task_queue: mp.Queue, prefill_done_queue: mp.Queue, nix_meta_queue: mp.Queue) -> None:
        self.init_custom = MethodType(PDNIXLBackendForDecodeNode.init_custom, self)
        super().__init__(prefill_task_queue, prefill_done_queue, nix_meta_queue)
        self.classed_req_strict_prefill = False
        self.support_overlap = True

        self._build_remote_prefill_task = MethodType(PDNIXLBackendForDecodeNode._build_remote_prefill_task, self)
        self._trigger_remote_prefill = MethodType(PDNIXLBackendForDecodeNode._trigger_remote_prefill, self)
        self._pre_handle_finished_reqs = MethodType(PDNIXLBackendForDecodeNode._pre_handle_finished_reqs, self)
        self._get_classed_reqs = MethodType(PDNIXLBackendForDecodeNode._get_classed_reqs, self)

