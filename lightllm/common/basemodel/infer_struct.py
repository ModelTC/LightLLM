import torch
from lightllm.common.mem_manager import MemoryManager
from lightllm.common.req_manager import ReqManager
from lightllm.distributed import CustomProcessGroup


class InferStateInfo:
    """
    推理时用的信息结构体
    """

    def __init__(self):
        self.batch_size = None
        self.total_token_num = None
        self.b_req_idx = None
        self.b_start_loc = None
        self.b_ready_cache_len = None  # only for prefill prompt cache used.
        self.b_seq_len = None
        # max_len_in_batch prefill 和 decode 阶段含义不同
        # prefill 阶段指每个req 输入token的长度（不包括已经cache的部分）最大值
        # decode 阶段指的是每个req的总长 最大值
        self.max_len_in_batch = None
        self.is_prefill = None

        self.mem_manager: MemoryManager = None
        self.req_manager: ReqManager = None

        self.mem_index = None
        self.kv_buffer_shapedtype = None

        self.is_token_healing = False
        self.return_all_prompt_logics = False
        self.use_dynamic_prompt_cache = False
        self.multimodal_params = None
        self.is_cuda_graph = False  # 标记是否是cuda graph的捕获推理
        self.dist_group: CustomProcessGroup = None

        # 在microbatch overlap的运行模式下，用于标记当前 microbatch 的 index 序号
        # 在一些细节场景下需要有该信息区分一些资源的申请和管理。
        self.microbatch_index: int = 0

    def init_some_extra_state(self, model, input_ids: torch.Tensor):
        pass

    def copy_for_cuda_graph(self, new_infer_state):
        for attr_name, attr_value in vars(new_infer_state).items():
            if isinstance(attr_value, torch.Tensor):
                attr_ = getattr(self, attr_name, None)
                if attr_ is not None and attr_.data_ptr() != attr_value.data_ptr():
                    attr_.copy_(attr_value, non_blocking=True)
        return
