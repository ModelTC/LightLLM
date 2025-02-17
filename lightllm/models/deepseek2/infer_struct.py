import os
import torch
import numpy as np
import torch.distributed as dist
from lightllm.models.llama.infer_struct import LlamaInferStateInfo


class Deepseek2InferStateInfo(LlamaInferStateInfo):
    def __init__(self):
        super().__init__()
        self.kv_starts = None
        self.enable_dp = os.getenv("ENABLE_DP", "0").upper() in ["ON", "TRUE", "1"]

    def init_some_extra_state(self, model, input_ids: torch.Tensor):
        super().init_some_extra_state(model, input_ids)
        # 只有 decode 阶段使用 ppl 的优化算子才会有这个管理变量
        if not self.is_prefill:
            self.kv_starts = torch.cat([self.b_start_loc, self.b_start_loc[-1:] + self.b_seq_len[-1:]], dim=0)
            self.total_token_num_tensor = torch.sum(self.b_seq_len)

        if self.is_prefill:
            self.b_kv_start_loc = self.b_seq_len.cumsum(dim=0) - self.b_seq_len

        if self.enable_dp:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            local_token_num = input_ids.size(0)
            all_token_num = [torch.zeros(1, dtype=torch.int32).to(input_ids.device) for _ in range(world_size)]
            dist.all_gather(all_token_num, torch.tensor([local_token_num], dtype=torch.int32).to(input_ids.device))
            all_token_num = torch.cat(all_token_num, dim=0)
            self.all_token_num = all_token_num.sum().cpu().numpy()
            cumsum_token_num = torch.cumsum(all_token_num, dim=0).cpu().numpy()
            self.all_start_idx = cumsum_token_num - all_token_num.cpu().numpy()
            self.all_end_idx = cumsum_token_num
            self.start_idx = self.all_start_idx[rank]
            self.end_idx = self.all_end_idx[rank]

        return
