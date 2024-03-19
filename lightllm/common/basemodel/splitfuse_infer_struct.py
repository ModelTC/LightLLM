import torch
from .infer_struct import InferStateInfo
from lightllm.common.mem_manager import MemoryManager
from lightllm.common.req_manager import ReqManager


class SplitFuseInferStateInfo:
    """
    推理时用的信息结构体
    """

    inner_decode_infer_state_class = InferStateInfo

    def __init__(self):
        self.use_dynamic_prompt_cache = False

        self.batch_size = None

        self.decode_req_num = None
        self.decode_total_token_num = None
        self.decode_b_req_idx: torch.Tensor = None
        self.decode_b_start_loc: torch.Tensor = None
        self.decode_b_seq_len: torch.Tensor = None
        self.decode_max_len_in_batch = None

        self.prefill_req_num = None
        self.prefill_b_req_idx: torch.Tensor = None
        self.prefill_b_split_start_loc: torch.Tensor = None
        self.prefill_b_split_ready_cache_len: torch.Tensor = None
        self.prefill_max_split_seq_len_in_batch = None
        self.prefill_b_seq_len: torch.Tensor = None

        self.mem_manager: MemoryManager = None
        self.req_manager: ReqManager = None

        self.mem_is_contiguous = None
        self.mem_start = None
        self.mem_end = None
        self.mem_index = None
        self.kv_buffer = None

        self.parrall_stream = torch.cuda.Stream()
        self.start_event = torch.cuda.Event()
        self.end_event = torch.cuda.Event()

        self.is_splitfuse = True
        self.inner_decode_infer_status = None
        return

    def create_inner_decode_infer_status(self):
        infer_status = self.inner_decode_infer_state_class()
        infer_status.batch_size = self.decode_req_num
        infer_status.total_token_num = self.decode_total_token_num
        infer_status.b_req_idx = self.decode_b_req_idx
        infer_status.b_start_loc = self.decode_b_start_loc
        infer_status.b_seq_len = self.decode_b_seq_len
        infer_status.max_len_in_batch = self.decode_max_len_in_batch

        infer_status.mem_manager = self.mem_manager
        infer_status.req_manager = self.req_manager
        infer_status.is_prefill = False

        self.inner_decode_infer_status = infer_status
        return infer_status

    def init_some_extra_state(self, model, input_ids: torch.Tensor):
        pass
