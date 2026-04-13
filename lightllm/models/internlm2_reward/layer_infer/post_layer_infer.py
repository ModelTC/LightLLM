import torch
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer
from ..layer_weights.pre_and_post_layer_weight import Internlm2RewardPreAndPostLayerWeight


class Internlm2RewardPostLayerInfer(LlamaPostLayerInfer):
    def token_forward(
        self, input_embdings, infer_state: LlamaInferStateInfo, layer_weight: Internlm2RewardPreAndPostLayerWeight
    ):
        input_embdings = self._tpsp_allgather(input=input_embdings, infer_state=infer_state)
        if infer_state.need_dp_prefill_balance:
            input_embdings = infer_state._all_to_all_unbalance_get(data=input_embdings)

        last_input, token_num = self._slice_get_last_input(input_embdings, infer_state)

        input_embdings = None
        last_input = self._norm(last_input, infer_state, layer_weight)
        score = layer_weight.score_head_.mm(last_input)

        return score
