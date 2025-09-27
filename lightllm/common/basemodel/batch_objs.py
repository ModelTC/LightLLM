import torch
from dataclasses import dataclass, field
from typing import Optional
from typing import List


@dataclass
class ModelInput:
    # 通用变量
    batch_size: int
    total_token_num: int
    max_len_in_batch: int
    input_ids: torch.Tensor = None
    b_req_idx: torch.Tensor = None
    b_mtp_index: torch.Tensor = None
    b_seq_len: torch.Tensor = None
    mem_indexes: torch.Tensor = None
    is_prefill: bool = False
    b_ready_cache_len: torch.Tensor = None
    multimodal_params: list = field(default_factory=list)

    # cpu 变量
    input_ids_cpu: torch.Tensor = None
    b_req_idx_cpu: torch.Tensor = None
    b_mtp_index_cpu: torch.Tensor = None
    mem_indexes_cpu: torch.Tensor = None
    b_seq_len_cpu: torch.Tensor = None
    b_ready_cache_len_cpu: torch.Tensor = None
    # prefill 阶段使用的参数，但是不是推理过程使用的参数，是推理外部进行资源管理
    # 的一些变量
    b_prefill_has_output_cpu: List[bool] = None  # 标记进行prefill的请求是否具有输出

    # 专有变量，用于一些特殊的模型，特殊的模式下, 传递一些特殊
    # 的输入变量。只在特殊的模型模式下才会具体使用和生效。

    # deepseekv3_mtp_draft_input_hiddens 用于 deepseekv3 模型 mtp 模式下
    # 的 draft 模型的输入
    deepseekv3_mtp_draft_input_hiddens: Optional[torch.Tensor] = None

    def to_cuda(self):
        # input_ids 可能不存在，通过req_to_token_indexs来获取
        if self.input_ids is None and self.input_ids_cpu is not None:
            self.input_ids = self.input_ids_cpu.cuda(non_blocking=True)
        if self.mem_indexes is None:
            self.mem_indexes = self.mem_indexes_cpu.cuda(non_blocking=True)
        if self.b_req_idx is None:
            self.b_req_idx = self.b_req_idx_cpu.cuda(non_blocking=True)
        if self.b_seq_len is None:
            self.b_seq_len = self.b_seq_len_cpu.cuda(non_blocking=True)
        # b_ready_cache_len 只在 prefill 阶段生效
        if self.b_ready_cache_len_cpu is not None:
            self.b_ready_cache_len = self.b_ready_cache_len_cpu.cuda(non_blocking=True)
        if self.b_mtp_index is None:
            self.b_mtp_index = self.b_mtp_index_cpu.cuda(non_blocking=True)


@dataclass
class ModelOutput:
    # 通用变量
    logits: torch.Tensor

    # 专有变量，用于一些特殊的模型，特殊的模式下, 传递一些特殊
    # 的输出变量。只在特殊的模型模式下才会具体使用和生效。

    # deepseekv3_mtp_main_output_hiddens 用于在mtp模式下，llm main model
    # 输出最后一层的hidden state 状态用于 draft 模型的 deepseekv3_mtp_draft_input_hiddens
    # 输入
    deepseekv3_mtp_main_output_hiddens: Optional[torch.Tensor] = None
