import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np

from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight
from einops import rearrange
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama.triton_kernel.rmsnorm import rmsnorm_forward
from lightllm.common.basemodel import PostLayerInferTpl

class LlamaPostLayerInfer(PostLayerInferTpl):
    """
    """

    def __init__(self, tp_rank, world_size, network_config, mode):
        super().__init__(tp_rank, world_size, network_config, mode)
        self.eps_ = network_config["rms_norm_eps"]
        self.vocab_size_ = network_config["vocab_size"]
        self.embed_dim_ = network_config["n_embed"]
        return
    
    def _norm(self, input, infer_state, layer_weight:LlamaPreAndPostLayerWeight) -> torch.Tensor:
        return rmsnorm_forward(input, layer_weight.final_norm_weight_, eps=self.eps_)

    def soft_max(self, data):
        return torch.softmax(data.permute(1, 0).float(), dim=-1)

    def token_forward(self, input_embdings, infer_state: LlamaInferStateInfo, layer_weight: LlamaPreAndPostLayerWeight, return_logics=False):
        batch_size = infer_state.batch_size
        last_input = torch.empty((batch_size, self.embed_dim_), device=input_embdings.device, dtype=torch.float16)
        if infer_state.is_prefill:
            last_index = torch.cumsum(infer_state.b_seq_len, dim=0, dtype=torch.long) - 1
            last_input[:, :] = input_embdings[last_index, :]
        else:
            last_input[:, :] = input_embdings[-batch_size:, :]
        input_embdings = None
        last_input = self._norm(last_input, infer_state, layer_weight)
        last_input = rearrange(last_input, "batch embed_dim -> embed_dim batch").contiguous().reshape(-1, batch_size)
        logic_batch = torch.mm(layer_weight.lm_head_weight_, last_input)
        last_input = None
        if self.world_size_ == 1:
            gather_data = logic_batch
        else:
            gather_data = torch.empty((self.vocab_size_, batch_size), device=logic_batch.device, dtype=torch.float16)
            split_size = self.vocab_size_ // self.world_size_
            dist.all_gather([gather_data[i * split_size: (i + 1) * split_size, :]
                            for i in range(self.world_size_)], logic_batch, group=None, async_op=False)
        logic_batch = None

        if not return_logics:
            prob_out = self.soft_max(gather_data)
            gather_data = None
            return prob_out
        else:
            ans_logics = gather_data.permute(1, 0).float()
            gather_data = None
            return ans_logics
