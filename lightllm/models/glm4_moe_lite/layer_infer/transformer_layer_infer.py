import os
import torch
import torch.distributed as dist
import triton
from functools import partial
from lightllm.models.deepseek2.layer_infer.transformer_layer_infer import Deepseek2TransformerLayerInfer
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.distributed.communication_op import reduce_scatter_tensor


class Glm4MoeLiteTransformerLayerInfer(Deepseek2TransformerLayerInfer):
    def __init__(self, layer_num, network_config):
        self._glm4_layer_num = layer_num
        self._glm4_first_k_dense = network_config.get("first_k_dense_replace", 0)
        self._glm4_has_routed_experts = network_config.get("n_routed_experts") is not None
        super().__init__(layer_num, network_config)

    @property
    def is_moe(self):
        return self._glm4_has_routed_experts and self._glm4_layer_num >= self._glm4_first_k_dense

    @is_moe.setter
    def is_moe(self, value):
        pass

    def _get_o(self, input: torch.Tensor, infer_state, layer_weight) -> torch.Tensor:
        if input.shape[2] == self.kv_lora_rank:
            input = layer_weight.v_b_proj_.bmm(input.transpose(0, 1)).transpose(0, 1)
        o_tensor = layer_weight.o_weight_.mm(input.reshape(-1, self.tp_q_head_num_ * self.v_head_dim))
        return o_tensor

    def _tpsp_get_o(self, input, infer_state, layer_weight) -> torch.Tensor:
        if infer_state.need_dp_prefill_balance:
            input = infer_state._all_to_all_balance_get(data=input)

        if input.shape[2] == self.kv_lora_rank:
            input = layer_weight.v_b_proj_.bmm(input.transpose(0, 1)).transpose(0, 1)

        input = input.reshape(-1, self.tp_q_head_num_ * self.v_head_dim)
        dest_size = triton.cdiv(input.shape[0], self.tp_world_size_) * self.tp_world_size_
        o_tensor = self.alloc_tensor((dest_size, self.embed_dim_), dtype=input.dtype, device=input.device)
        layer_weight.o_weight_.mm(input, out=o_tensor[0 : len(infer_state.input_ids), :])
        e_o_tensor = o_tensor[len(infer_state.input_ids) :, :]
        if e_o_tensor.shape[0] > 0:
            e_o_tensor.fill_(0)

        if self.tp_world_size_ > 1:
            sp_token_num = o_tensor.shape[0] // self.tp_world_size_
            reduce_o_tensor = self.alloc_tensor((sp_token_num, self.embed_dim_), dtype=input.dtype, device=input.device)
            reduce_scatter_tensor(
                output=reduce_o_tensor,
                input=o_tensor,
                op=dist.ReduceOp.SUM,
                group=infer_state.dist_group,
                async_op=False,
            )
            o_tensor = reduce_o_tensor

        return o_tensor

    def _moe_ffn(self, input, infer_state, layer_weight):
        hidden_states = input.view(-1, self.embed_dim_)
        num_tokens, hidden_dim = hidden_states.shape

        if self.n_shared_experts is not None and layer_weight.num_fused_shared_experts == 0:
            shared_output = LlamaTransformerLayerInfer._ffn(self, hidden_states, infer_state, layer_weight)

        router_logits = layer_weight.moe_gate.mm(hidden_states.to(torch.float32))

        layer_weight.experts.experts(
            hidden_states,
            router_logits=router_logits,
            top_k=self.num_experts_per_tok,
            renormalize=self.norm_topk_prob,
            use_grouped_topk=self.n_group,
            topk_group=self.topk_group,
            num_expert_group=self.n_group,
        )

        if self.n_shared_experts is not None and layer_weight.num_fused_shared_experts == 0:
            hidden_states.add_(shared_output)

        return hidden_states.view(num_tokens, hidden_dim)
