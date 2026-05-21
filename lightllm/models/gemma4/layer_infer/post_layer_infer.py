import torch
import numpy as np
from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer
from lightllm.distributed.communication_op import all_gather


class Gemma4PostLayerInfer(LlamaPostLayerInfer):
    """
    Same final RMSNorm + tied lm_head path as Llama, with an extra tanh-based
    logit softcap at the end: logits = softcap * tanh(logits / softcap).
    """

    def __init__(self, network_config):
        super().__init__(network_config)
        self.final_logit_softcapping = float(network_config.get("final_logit_softcapping"))

    def token_forward(self, input_embdings, infer_state, layer_weight):
        last_hidden, token_num = self._slice_get_last_input(input_embdings, infer_state)
        input_embdings_dtype = input_embdings.dtype
        last_hidden = self._norm(last_hidden, infer_state, layer_weight)
        lm_input = last_hidden.permute(1, 0).view(-1, token_num)
        logic_batch = layer_weight.lm_head_weight_(input=lm_input, alloc_func=self.alloc_tensor)
        vocab_size = layer_weight.lm_head_weight_.vocab_size
        if self.tp_world_size_ == 1:
            gather_data = logic_batch
        else:
            gather_data = self.alloc_tensor((vocab_size, token_num), dtype=input_embdings_dtype)
            split_indexes = np.linspace(0, vocab_size, self.tp_world_size_ + 1, dtype=np.int64)
            all_gather(
                [gather_data[split_indexes[i] : split_indexes[i + 1], :] for i in range(self.tp_world_size_)],
                logic_batch,
                group=infer_state.dist_group,
                async_op=False,
            )
        logic_batch = None
        logits = self.alloc_tensor((token_num, vocab_size), dtype=torch.float32)
        logits[:, :] = gather_data.permute(1, 0)
        gather_data = None
        if self.final_logit_softcapping is not None and self.final_logit_softcapping > 0:
            cap = self.final_logit_softcapping
            logits = torch.tanh(logits / cap) * cap
        return logits, last_hidden
