import torch
import numpy as np
from lightllm.models.cohere.infer_struct import CohereInferStateInfo
from lightllm.models.cohere.layer_weights.pre_and_post_layer_weight import CoherePreAndPostLayerWeight
from lightllm.models.cohere.triton_kernels.layernorm import layernorm_forward
from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer
from lightllm.common.build_utils import repair_config
from lightllm.distributed.communication_op import all_gather


class CoherePostLayerInfer(LlamaPostLayerInfer):
    def __init__(self, network_config, mode):
        repair_config(config=network_config, same_names=["layer_norm_eps", "rms_norm_eps"])
        super().__init__(network_config, mode)
        self.eps_ = network_config["layer_norm_eps"]
        self.logits_scale = network_config["logit_scale"]
        return

    def _norm(
        self, input: torch.Tensor, infer_state: CohereInferStateInfo, layer_weight: CoherePreAndPostLayerWeight
    ) -> torch.Tensor:
        return layernorm_forward(
            input.unsqueeze(1), layer_weight.final_norm_weight_.weight.unsqueeze(0), eps=self.eps_
        ).squeeze(1)

    def token_forward(
        self, input_embdings: torch.Tensor, infer_state: CohereInferStateInfo, layer_weight: CoherePreAndPostLayerWeight
    ):
        last_input, token_num = self._slice_get_last_input(input_embdings, infer_state)
        input_embdings_dtype = input_embdings.dtype
        input_embdings = None
        last_input = self._norm(last_input, infer_state, layer_weight)
        last_input = last_input.permute(1, 0).view(-1, token_num)
        logic_batch = layer_weight.lm_head_weight_.lm_head(input=last_input, alloc_func=self.alloc_tensor)
        last_input = None
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
        gather_data = gather_data * self.logits_scale
        logic_batch = None
        ans_logics = self.alloc_tensor(
            (token_num, vocab_size),
            dtype=torch.float32,
        )
        ans_logics[:, :] = gather_data.permute(1, 0)
        gather_data = None
        return ans_logics

    def tpsp_token_forward(
        self, input_embdings: torch.Tensor, infer_state: CohereInferStateInfo, layer_weight: CoherePreAndPostLayerWeight
    ):
        raise NotImplementedError("not impl")

    def overlap_tpsp_token_forward(
        self,
        input_embdings: torch.Tensor,
        input_embdings1: torch.Tensor,
        infer_state: CohereInferStateInfo,
        infer_state1: CohereInferStateInfo,
        layer_weight: CoherePreAndPostLayerWeight,
    ):
        raise NotImplementedError("not impl")
