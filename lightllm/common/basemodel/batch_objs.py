import torch
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelInput:
    # 通用变量
    batch_size: int
    total_token_num: int
    max_len_in_batch: int
    input_ids: torch.Tensor
    mem_indexes: torch.Tensor
    b_req_idx: torch.Tensor
    b_seq_len: torch.Tensor
    is_prefill: bool = False
    b_ready_cache_len: torch.Tensor = None
    multimodal_params: list = field(default_factory=list)

    # 专有变量，用于一些特殊的模型，特殊的模式下, 传递一些特殊
    # 的输入变量。只在特殊的模型模式下才会具体使用和生效。

    # deepseekv3_mtp_draft_input_hiddens 用于 deepseekv3 模型 mtp 模式下
    # 的 draft 模型的输入
    deepseekv3_mtp_draft_input_hiddens: Optional[torch.Tensor] = None


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
