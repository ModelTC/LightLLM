import torch
from lightllm.common.mem_manager import MemoryManager
from lightllm.common.req_manager import ReqManager
from lightllm.distributed import CustomProcessGroup
from typing import Tuple, Any
from .triton_kernel.gen_prefill_params import gen_prefill_params
from .triton_kernel.gen_decode_params import gen_decode_params


class InferStateInfo:
    """
    推理时用的信息结构体
    """

    def __init__(self):
        self.batch_size: int = None
        self.total_token_num: int = None
        self.b_req_idx: torch.Tensor = None
        self.b_start_loc: torch.Tensor = None
        self.b_ready_cache_len: torch.Tensor = None  # only for prefill prompt cache used.
        self.b_seq_len: torch.Tensor = None
        # max_len_in_batch prefill 和 decode 阶段含义不同
        # prefill 阶段指每个req 输入token的长度（不包括已经cache的部分）最大值
        # decode 阶段指的是每个req的总长 最大值
        self.max_len_in_batch: int = None
        self.is_prefill: bool = None

        self.mem_manager: MemoryManager = None
        self.req_manager: ReqManager = None

        self.mem_index: torch.Tensor = None
        self.kv_buffer_shapedtype: Tuple[Any, Any] = None

        self.is_token_healing: bool = False
        self.return_all_prompt_logics: bool = False
        self.use_dynamic_prompt_cache: bool = False
        self.multimodal_params: dict = None
        self.is_cuda_graph: bool = False  # 标记是否是cuda graph的捕获推理
        self.dist_group: CustomProcessGroup = None

        # 在microbatch overlap的运行模式下，用于标记当前 microbatch 的 index 序号
        # 在一些细节场景下需要有该信息区分一些资源的申请和管理。
        self.microbatch_index: int = 0

        # 衍生使用的管理变量，为了方便扩展接入其他的高性能attention推理算子，在
        # inferstate 基类上添加下面的标记变量，用于扩展。
        # b 开头的tensor变量其shape为[batch_size,]
        # b1 开头的tensor变量其shape为[batch_size + 1,]
        self.b_q_seq_len: torch.Tensor = None
        self.b1_cu_q_seq_len: torch.Tensor = None
        self.b_kv_seq_len: torch.Tensor = None
        self.b1_cu_kv_seq_len: torch.Tensor = None
        self.position_ids: torch.Tensor = None
        self.max_q_seq_len: int = None
        self.max_kv_seq_len: int = None

    def init_some_extra_state(self, model, input_ids: torch.Tensor):
        if self.is_prefill:
            (
                self.b_q_seq_len,
                self.b1_cu_q_seq_len,
                self.b_kv_seq_len,
                self.b1_cu_kv_seq_len,
                self.position_ids,
                self.max_q_seq_len,
                self.max_kv_seq_len,
            ) = gen_prefill_params(
                input_token_num=input_ids.shape[0],
                b_ready_cache_len=self.b_ready_cache_len,
                b_seq_len=self.b_seq_len,
            )
            self.b_start_loc = self.b1_cu_q_seq_len[0:-1]
        else:
            (
                self.b_q_seq_len,
                self.b1_cu_q_seq_len,
                self.b_kv_seq_len,
                self.b1_cu_kv_seq_len,
                self.position_ids,
                self.max_q_seq_len,
                self.max_kv_seq_len,
            ) = gen_decode_params(b_seq_len=self.b_seq_len)
            self.b_start_loc = self.b1_cu_kv_seq_len[0:-1]

    def copy_for_cuda_graph(self, new_infer_state):
        for attr_name, attr_value in vars(new_infer_state).items():
            if isinstance(attr_value, torch.Tensor):
                attr_ = getattr(self, attr_name, None)
                if attr_ is not None and attr_.data_ptr() != attr_value.data_ptr():
                    attr_.copy_(attr_value, non_blocking=True)
        return
