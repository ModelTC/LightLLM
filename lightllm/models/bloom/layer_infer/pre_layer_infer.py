import torch
import torch.distributed as dist
from lightllm.common.basemodel import PreLayerInferTpl
from lightllm.common.basemodel import InferStateInfo
from lightllm.models.bloom.layer_weights.pre_and_post_layer_weight import BloomPreAndPostLayerWeight
from lightllm.utils.infer_utils import mark_cost_time
from lightllm.models.bloom.triton_kernel.layernorm import layernorm_forward

class BloomPreLayerInfer(PreLayerInferTpl):
    """
    """
    def __init__(self, tp_rank, world_size, network_config, mode):
        super().__init__(tp_rank, world_size, network_config, mode)
        self.eps_ = network_config["layer_norm_epsilon"]
        tp_vocab_size_ = network_config["vocab_size"] // self.world_size_
        self.vob_start_id_ = tp_vocab_size_ * self.tp_rank_
        self.vob_end_id_ = tp_vocab_size_ * (self.tp_rank_ + 1)
        return
    
    def _norm(self, input, infer_state, layer_weight : BloomPreAndPostLayerWeight) -> torch.Tensor:
        return layernorm_forward(input, layer_weight.pre_norm_weight_, layer_weight.pre_norm_bias_, eps=self.eps_)

    @mark_cost_time("pre context forward")  # dont to remove this, will make performence down, did not know why
    def context_forward(self, input_ids, infer_state: InferStateInfo, layer_weight: BloomPreAndPostLayerWeight):
        total_token_num = infer_state.total_token_num
        input_ids = input_ids[0:total_token_num]

        input_mask = torch.logical_or(self.vob_start_id_ > input_ids, input_ids >= self.vob_end_id_)
        tmp_input_ids = (input_ids - self.vob_start_id_)
        tmp_input_ids[input_mask] = 0
        input_embdings = torch.embedding(layer_weight.wte_weight_, tmp_input_ids, padding_idx=-1)
        input_embdings[input_mask] = 0.0
        if self.world_size_ > 1:
            dist.all_reduce(input_embdings, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings = self._norm(input_embdings, infer_state, layer_weight)
        return input_embdings

    def token_forward(self, input_ids, infer_state: InferStateInfo, layer_weight: BloomPreAndPostLayerWeight):
        input_mask = torch.logical_or(self.vob_start_id_ > input_ids, input_ids >= self.vob_end_id_)
        tmp_input_ids = (input_ids - self.vob_start_id_)
        tmp_input_ids[input_mask] = 0
        input_embdings = torch.embedding(layer_weight.wte_weight_, tmp_input_ids, padding_idx=-1)
        input_embdings[input_mask] = 0.0
        if self.world_size_ > 1:
            dist.all_reduce(input_embdings, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings = self._norm(input_embdings, infer_state, layer_weight)
        return input_embdings
