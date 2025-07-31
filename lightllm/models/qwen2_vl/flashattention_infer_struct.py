import os
import torch
import numpy as np
import torch.distributed as dist
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.models.llama.flashattention_infer_struct import FlashAttentionStateInfo
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.models.deepseek2.triton_kernel.repack_kv_index import repack_kv_index
from lightllm.common.basemodel.batch_objs import ModelInput


class Qwen2VLFlashAttentionStateInfo(FlashAttentionStateInfo):
    def init_some_extra_state(self, model, input_ids: torch.Tensor):
        InferStateInfo.init_some_extra_state(self, model, input_ids)
        if self.is_prefill:
            self.max_seq_len = self.max_kv_seq_len
            self.q_max_seq_len = self.max_q_seq_len
            position_ids = self.position_ids
            self.position_sin = model._sin_cached[:, position_ids, :].unsqueeze(1)
            self.position_cos = model._cos_cached[:, position_ids, :].unsqueeze(1)
            position_ids = None
        else:
            position_ids = self.position_ids
            self.position_sin = model._sin_cached[:, position_ids, :].unsqueeze(1)
            self.position_cos = model._cos_cached[:, position_ids, :].unsqueeze(1)

        # init flash attention state
        self._init_flash_attention_state(model, input_ids)
        return
