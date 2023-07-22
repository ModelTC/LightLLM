import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np

from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import PreAndPostLayerWeight
from einops import rearrange
from lightllm.models.llama.layer_infer.infer_struct import InferStateInfo
from lightllm.models.llama.triton_kernel.rmsnorm import rmsnorm_forward

torch.backends.cudnn.enabled = True


class PostLayerInfer:
    """
    """

    def __init__(self, tp_rank, world_size, network_config):
        self.tp_rank_ = tp_rank
        self.world_size_ = world_size
        self.network_config_ = network_config
        assert (network_config["vocab_size"] % self.world_size_ == 0)
        self.vocab_size_ = network_config["vocab_size"]
        self.tp_vocab_size_ = network_config["vocab_size"] // self.world_size_
        self.embed_dim_ = network_config["hidden_size"]
        self.layer_norm_eps_ = network_config["rms_norm_eps"]
        self.vob_start_id_ = self.tp_vocab_size_ * self.tp_rank_
        self.vob_end_id_ = self.tp_vocab_size_ * (self.tp_rank_ + 1)

    def soft_max(self, data):
        return torch.softmax(data.permute(1, 0).float(), dim=-1)

    def token_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight: PreAndPostLayerWeight, return_logics=False):
        batch_size = infer_state.batch_size
        last_input = torch.empty((batch_size, self.embed_dim_), device=input_embdings.device, dtype=torch.float16)
        if infer_state.is_prefill:
            last_index = torch.cumsum(infer_state.b_seq_len, dim=0, dtype=torch.long) - 1
            last_input[:, :] = input_embdings[last_index, :]
        else:
            last_input[:, :] = input_embdings[-batch_size:, :]
        input_embdings = None
        last_input = rmsnorm_forward(last_input, layer_weight.final_layernorm_weight_, eps=self.layer_norm_eps_)
        last_input = rearrange(last_input, "batch embed_dim -> embed_dim batch").contiguous().reshape(-1, batch_size)
        logic_batch = torch.mm(layer_weight.lm_head_weight, last_input)
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
