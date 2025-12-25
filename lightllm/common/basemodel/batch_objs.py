import torch
from dataclasses import dataclass, field
from typing import Optional
from typing import List
from lightllm.utils.envs_utils import enable_diverse_mode_gqa_decode_fast_kernel
from lightllm.utils.tensor_utils import tensor_to_no_ref_tensor


@dataclass
class ModelInput:
    # 通用变量
    batch_size: int
    total_token_num: int
    max_len_in_batch: int
    # 在 decode 阶段， 常规模式下， max_q_seq_len 必定是 1，
    # 在 mtp 模式下，max_q_seq_len 统计的是一个请求考虑了 mtp 步数的
    # 最大长度，实际值是 max([(1 + req.mtp_step) for req in reqs])
    max_q_seq_len: int
    max_kv_seq_len: int
    max_cache_len: int = None
    prefix_total_token_num: int = None
    input_ids: torch.Tensor = None
    b_req_idx: torch.Tensor = None
    b_mtp_index: torch.Tensor = None
    b_seq_len: torch.Tensor = None
    # 只会在 diverse_mode 下的 decode 阶段真正被使用的参数, 用于记录共享的radix cache中的长度
    b_shared_seq_len: torch.Tensor = None
    # 只会在 diverse_mode 下的 decode 阶段真正被使用的参数, 用于记录请求间的共享关系。
    # 举列说明:
    # b_shared_seq_len : [10, 10, 10, 11, 11, 11, 11]
    # b_mark_shared_group: [0, 0, 3, 0, 0, 0, 4]
    # b_mark_shared_group 中每一个不为0的位置都代表其与前面多少个请求形成一个共享前缀组。属于
    # 同一个共享前缀组的请求, 其在对应的 b_shared_seq_len 中的内容必然相同。
    b_mark_shared_group: torch.Tensor = None
    mem_indexes: torch.Tensor = None
    is_prefill: bool = False
    b_ready_cache_len: torch.Tensor = None
    b_prefill_start_loc: torch.Tensor = None
    multimodal_params: list = None
    # cpu 变量
    mem_indexes_cpu: torch.Tensor = None
    # prefill 阶段使用的参数，但是不是推理过程使用的参数，是推理外部进行资源管理
    # 的一些变量
    b_prefill_has_output_cpu: List[bool] = None  # 标记进行prefill的请求是否具有输出

    # 专有变量，用于一些特殊的模型，特殊的模式下, 传递一些特殊
    # 的输入变量。只在特殊的模型模式下才会具体使用和生效。

    # deepseekv3_mtp_draft_input_hiddens 用于 deepseekv3 模型 mtp 模式下
    # 的 draft 模型的输入
    deepseekv3_mtp_draft_input_hiddens: Optional[torch.Tensor] = None

    def to_cuda(self):
        if self.input_ids is not None:
            self.input_ids = self.input_ids.cuda(non_blocking=True)
        if self.mem_indexes is None:
            self.mem_indexes = self.mem_indexes_cpu.cuda(non_blocking=True)
        self.b_req_idx = self.b_req_idx.cuda(non_blocking=True)
        self.b_seq_len = self.b_seq_len.cuda(non_blocking=True)
        self.b_mtp_index = self.b_mtp_index.cuda(non_blocking=True)
        if self.b_ready_cache_len is not None:
            self.b_ready_cache_len = self.b_ready_cache_len.cuda(non_blocking=True)
        if self.b_prefill_start_loc is not None:
            self.b_prefill_start_loc = self.b_prefill_start_loc.cuda(non_blocking=True)
        if not self.is_prefill and enable_diverse_mode_gqa_decode_fast_kernel():
            batch_size = len(self.b_req_idx)
            if self.b_mark_shared_group is None:
                self.b_mark_shared_group = torch.ones(size=(batch_size,), dtype=torch.int32, device="cuda")
            else:
                self.b_mark_shared_group = self.b_mark_shared_group.cuda(non_blocking=True)
            if self.b_shared_seq_len is None:
                self.b_shared_seq_len = torch.zeros(size=(batch_size,), dtype=torch.int32, device="cuda")
            else:
                self.b_shared_seq_len = self.b_shared_seq_len.cuda(non_blocking=True)

    def __post_init__(self):
        self.check_input()

    def check_input(self):
        assert len(self.multimodal_params) == self.batch_size


@dataclass
class ModelOutput:
    # 通用变量
    logits: torch.Tensor
    # 用于判断 mem_indexes 是否成功写入 req manager 中的事件对象。
    prefill_mem_indexes_ready_event: torch.Event = None

    # 专有变量，用于一些特殊的模型，特殊的模式下, 传递一些特殊
    # 的输出变量。只在特殊的模型模式下才会具体使用和生效。

    # deepseekv3_mtp_main_output_hiddens 用于在mtp模式下，llm main model
    # 输出最后一层的hidden state 状态用于 draft 模型的 deepseekv3_mtp_draft_input_hiddens
    # 输入
    deepseekv3_mtp_main_output_hiddens: Optional[torch.Tensor] = None

    def to_no_ref_tensor(self):
        self.logits = tensor_to_no_ref_tensor(self.logits)
        if self.deepseekv3_mtp_main_output_hiddens is not None:
            self.deepseekv3_mtp_main_output_hiddens = tensor_to_no_ref_tensor(self.deepseekv3_mtp_main_output_hiddens)
