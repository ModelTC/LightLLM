import numpy as np
import torch
from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer
from lightllm.common.basemodel.batch_objs import ModelOutput
from lightllm.distributed.communication_op import all_gather


class Gemma4PostLayerInfer(LlamaPostLayerInfer):
    """
    Same final RMSNorm + tied lm_head path as Llama, with an extra tanh-based
    logit softcap at the end: logits = softcap * tanh(logits / softcap).
    """

    def __init__(self, network_config):
        super().__init__(network_config)
        cap = network_config.get("final_logit_softcapping")
        self.final_logit_softcapping = float(cap) if cap is not None else None

    def _get_normed_last_hidden(self, input_embdings, infer_state, layer_weight):
        last_input, token_num = self._slice_get_last_input(input_embdings, infer_state)
        input_embdings = None
        last_input = self._norm(last_input, infer_state, layer_weight)
        return last_input, token_num

    def _dense_logits_from_normed(self, normed, token_num, infer_state, layer_weight):
        input_embdings_dtype = normed.dtype
        lm_input = normed.permute(1, 0).view(-1, token_num)
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
        return logits

    def _apply_logit_softcapping(self, logits: torch.Tensor) -> torch.Tensor:
        if self.final_logit_softcapping is not None and self.final_logit_softcapping > 0:
            cap = self.final_logit_softcapping
            logits = torch.tanh(logits / cap) * cap
        return logits

    def _logits_from_normed(self, normed, token_num, infer_state, layer_weight):
        return self._dense_logits_from_normed(normed, token_num, infer_state, layer_weight)

    def _mtp_hiddens_from_normed(self, normed):
        return normed

    def token_forward(self, input_embdings, infer_state, layer_weight):
        normed, token_num = self._get_normed_last_hidden(input_embdings, infer_state, layer_weight)
        logits = self._logits_from_normed(normed, token_num, infer_state, layer_weight)
        logits = self._apply_logit_softcapping(logits)
        return ModelOutput(
            logits=logits,
            mtp_main_output_hiddens=self._mtp_hiddens_from_normed(normed),
        )
