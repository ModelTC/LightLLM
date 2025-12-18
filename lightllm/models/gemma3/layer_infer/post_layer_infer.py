import numpy as np
import torch

from lightllm.distributed.communication_op import all_gather
from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer
from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight


class Gemma3PostLayerInfer(LlamaPostLayerInfer):
    """ """

    def __init__(self, network_config, mode):
        super().__init__(network_config, mode)
        self.eps_ = 1e-6
        return

    def gemma3_rmsnorm(self, input, weight, eps: float = 1e-6, out=None):
        def _inner_norm(x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

        output = _inner_norm(input.float())
        output = output * (1.0 + weight.float())
        if out is not None:
            out = output.to(out.dtype)
        return output

    def _norm(self, input, infer_state, layer_weight) -> torch.Tensor:
        return self.gemma3_rmsnorm(input, layer_weight.final_norm_weight_, eps=self.eps_)

    def token_forward(self, input_embdings, infer_state, layer_weight):
        last_input, token_num = self._slice_get_last_input(input_embdings, infer_state)
        input_embdings_dtype = input_embdings.dtype
        last_input = self._norm(last_input.float(), infer_state, layer_weight).to(torch.bfloat16)
        last_input = last_input.permute(1, 0).view(-1, token_num)
        logic_batch = self.alloc_tensor(
            (layer_weight.lm_head_weight_.shape[0], last_input.shape[1]), dtype=last_input.dtype
        )
        torch.mm(layer_weight.lm_head_weight_.to(last_input.dtype), last_input, out=logic_batch)
        last_input = None
        if self.tp_world_size_ == 1:
            gather_data = logic_batch
        else:
            gather_data = self.alloc_tensor((self.vocab_size_, token_num), dtype=input_embdings_dtype)
            split_indexes = np.linspace(0, self.vocab_size_, self.tp_world_size_ + 1, dtype=np.int64)
            all_gather(
                [gather_data[split_indexes[i] : split_indexes[i + 1], :] for i in range(self.tp_world_size_)],
                logic_batch,
                group=infer_state.dist_group,
                async_op=False,
            )
        logic_batch = None
        ans_logics = self.alloc_tensor(
            (token_num, self.vocab_size_),
            dtype=torch.float32,
        )
        ans_logics[:, :] = gather_data.permute(1, 0)
        gather_data = None
        return ans_logics
