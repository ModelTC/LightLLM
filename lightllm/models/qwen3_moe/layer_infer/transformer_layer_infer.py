import os
import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
import triton
from typing import Tuple
from lightllm.models.qwen3_moe.layer_weights.transformer_layer_weight import Qwen3MOETransformerLayerWeight
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama.triton_kernel.rmsnorm import rmsnorm_forward
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.llama.triton_kernel.silu_and_mul import silu_and_mul_fwd
from functools import partial
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class Qwen3MOETransformerLayerInfer(LlamaTransformerLayerInfer):
    def __init__(self, layer_num, network_config, mode=[]):
        self.n_routed_experts = network_config["num_experts"]
        self.is_moe = (
            network_config["num_experts"] > 0
            and layer_num not in network_config["mlp_only_layers"]
            and (layer_num + 1) % network_config["decoder_sparse_step"] == 0
        )
        self.num_experts_per_tok = network_config["num_experts_per_tok"]
        self.norm_topk_prob = network_config["norm_topk_prob"]
        super().__init__(layer_num, network_config, mode)
        self.head_dim_ = network_config["head_dim"]
        self.tp_k_head_num_ = max(self.tp_k_head_num_, 1)
        self.tp_v_head_num_ = max(self.tp_v_head_num_, 1)
        return

    def _bind_func(self):
        super()._bind_func()
        self._bind_ffn()
        return

    def _bind_ffn(self):
        if self.is_moe:
            moe_mode = os.environ.get("MOE_MODE", "TP")
            if moe_mode == "EP":
                self._ffn = partial(Qwen3MOETransformerLayerInfer._moe_ffn_edp, self)
            else:
                self._ffn = partial(Qwen3MOETransformerLayerInfer._moe_ffn, self)
        else:
            self._ffn = partial(LlamaTransformerLayerInfer._ffn, self)

    def _get_qkv(
        self,
        input: torch.Tensor,
        cache_kv,
        infer_state: LlamaInferStateInfo,
        layer_weight: Qwen3MOETransformerLayerWeight,
    ) -> torch.Tensor:
        input = input.view(-1, self.embed_dim_)
        q = layer_weight.q_proj.mm(input)
        cache_kv = layer_weight.kv_proj.mm(
            input, out=cache_kv.view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_) * self.head_dim_)
        ).view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_), self.head_dim_)
        rmsnorm_forward(
            q.view(-1, self.head_dim_),
            weight=layer_weight.q_norm_weight_.weight,
            eps=self.eps_,
            out=q.view(-1, self.head_dim_),
        )

        cache_kv[:, : self.tp_k_head_num_, :] = rmsnorm_forward(
            cache_kv[:, : self.tp_k_head_num_, :].reshape(-1, cache_kv.shape[-1]),
            weight=layer_weight.k_norm_weight_.weight,
            eps=self.eps_,
        ).view(-1, self.tp_k_head_num_, cache_kv.shape[-1])

        rotary_emb_fwd(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            cache_kv[:, : self.tp_k_head_num_, :],
            infer_state.position_cos,
            infer_state.position_sin,
        )
        return q, cache_kv

    def _moe_ffn(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: Qwen3MOETransformerLayerWeight
    ) -> torch.Tensor:

        hidden_states = input.view(-1, self.embed_dim_)
        num_tokens, hidden_dim = hidden_states.shape
        router_logits = layer_weight.moe_gate.mm(hidden_states)
        layer_weight.experts.experts(
            hidden_states,
            router_logits=router_logits,
            top_k=self.num_experts_per_tok,
            renormalize=self.norm_topk_prob,
            use_grouped_topk=False,
            topk_group=None,
            num_expert_group=None,
        )
        return hidden_states.view(num_tokens, hidden_dim)

    def _moe_ffn_edp(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: Qwen3MOETransformerLayerWeight
    ) -> torch.Tensor:

        hidden_states = input
        token_num, hidden_dim = hidden_states.shape
        if self.n_shared_experts is not None:
            shared_output = LlamaTransformerLayerInfer._ffn(self, hidden_states, infer_state, layer_weight)

        router_logits = layer_weight.moe_gate.mm(hidden_states)
        ep_output = layer_weight.experts.experts(
            hidden_states,
            router_logits=router_logits,
            top_k=self.num_experts_per_tok,
            renormalize=self.norm_topk_prob,
            use_grouped_topk=self.n_group,
            topk_group=self.topk_group,
            num_expert_group=self.n_group,
            is_prefill=infer_state.is_prefill,
        )
        if self.n_shared_experts is not None:
            ep_output.add_(shared_output)

        ep_output = ep_output.view(token_num, hidden_dim)
        return ep_output
