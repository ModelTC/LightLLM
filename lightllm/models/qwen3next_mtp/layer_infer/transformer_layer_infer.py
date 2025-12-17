import torch
import torch.nn.functional as F
import torch.distributed as dist
from typing import Tuple
from functools import partial
from lightllm.models.qwen3next_mtp.layer_weights.transformer_layer_weight import Qwen3NextMTPTransformerLayerWeight
from lightllm.models.qwen3_moe.layer_infer.transformer_layer_infer import Qwen3MOETransformerLayerInfer
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.common.fused_moe.moe_silu_and_mul import silu_and_mul_fwd
from lightllm.models.qwen3next.triton_kernel.gemma_rmsnorm import gemma_rmsnorm_forward
from lightllm.distributed import all_reduce
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class Qwen3NextMTPTransformerLayerInfer(Qwen3MOETransformerLayerInfer):
    """
    Qwen3Next MTP Transformer Layer Inference.
    MTP layers use full attention (not linear attention) with MoE FFN and shared expert.
    Similar to Qwen3NextTransformerLayerInfer but for MTP layers.
    """

    def __init__(self, layer_num, network_config, mode=[]):
        # MTP is always MoE
        self.n_routed_experts = network_config["num_experts"]
        self.is_moe = True
        self.num_experts_per_tok = network_config["num_experts_per_tok"]
        self.norm_topk_prob = network_config["norm_topk_prob"]
        super(Qwen3MOETransformerLayerInfer, self).__init__(layer_num, network_config, mode)
        self.head_dim_ = network_config["head_dim"]
        self.tp_k_head_num_ = max(self.tp_k_head_num_, 1)
        self.tp_v_head_num_ = max(self.tp_v_head_num_, 1)
        self.partial_rotary_factor = network_config.get("partial_rotary_factor", 1.0)
        return

    def _bind_norm(self):
        pass

    def _att_norm(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: Qwen3NextMTPTransformerLayerWeight
    ) -> torch.Tensor:
        out = self.alloc_tensor(input.shape, input.dtype)
        gemma_rmsnorm_forward(input, layer_weight.att_norm_weight_.weight, self.eps_, out=out)
        return out

    def _ffn_norm(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: Qwen3NextMTPTransformerLayerWeight
    ) -> torch.Tensor:
        out = self.alloc_tensor(input.shape, input.dtype)
        gemma_rmsnorm_forward(input, layer_weight.ffn_norm_weight_.weight, self.eps_, out=out)
        return out

    def _get_qkv(
        self,
        input: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        layer_weight: Qwen3NextMTPTransformerLayerWeight,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input = input.view(-1, self.embed_dim_)
        q = layer_weight.q_proj.mm(input)
        cache_kv = layer_weight.kv_proj.mm(
            input.view(-1, self.embed_dim_),
        ).view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_), self.head_dim_)

        gemma_rmsnorm_forward(
            q.view(-1, self.head_dim_),
            layer_weight.q_norm_weight_.weight,
            eps=self.eps_,
            out=q.view(-1, self.head_dim_),
        )

        cache_kv[:, : self.tp_k_head_num_, :] = gemma_rmsnorm_forward(
            cache_kv[:, : self.tp_k_head_num_, :].reshape(-1, cache_kv.shape[-1]),
            layer_weight.k_norm_weight_.weight,
            eps=self.eps_,
        ).view(-1, self.tp_k_head_num_, cache_kv.shape[-1])

        rotary_emb_fwd(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            cache_kv[:, : self.tp_k_head_num_, :],
            infer_state.position_cos,
            infer_state.position_sin,
            partial_rotary_factor=self.partial_rotary_factor,
        )
        return q, cache_kv

    def _ffn_with_shared_expert(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: Qwen3NextMTPTransformerLayerWeight
    ) -> torch.Tensor:
        """MoE FFN with shared expert."""
        input = input.view(-1, self.embed_dim_)
        # Shared expert forward
        up_gate_out = layer_weight.shared_expert_gate_up_proj.mm(input)
        ffn1_out = self.alloc_tensor((input.size(0), up_gate_out.size(1) // 2), input.dtype)
        silu_and_mul_fwd(up_gate_out, ffn1_out)
        ffn2_out = layer_weight.shared_expert_down_proj.mm(ffn1_out)
        shared_expert_out = F.sigmoid(layer_weight.shared_expert_gate.mm(input)) * ffn2_out
        # MoE forward
        moe_out = self._ffn(input, infer_state, layer_weight)
        return shared_expert_out + moe_out

    def context_forward(
        self, input_embdings, infer_state: LlamaInferStateInfo, layer_weight: Qwen3NextMTPTransformerLayerWeight
    ):
        input1 = self._att_norm(input_embdings, infer_state, layer_weight)
        q, cache_kv = self._get_qkv(input1, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_kv, infer_state, layer_weight)
        o = self._context_attention_kernel(q, cache_kv, infer_state, layer_weight)
        q = None
        o = self._get_o(o, infer_state, layer_weight)
        if self.tp_world_size_ > 1:
            all_reduce(o, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        input_embdings.add_(o.view(-1, self.embed_dim_))
        o = None

        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn_with_shared_expert(input1, infer_state, layer_weight)
        input1 = None
        if self.tp_world_size_ > 1:
            all_reduce(ffn_out, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return input_embdings

    def token_forward(
        self, input_embdings, infer_state: LlamaInferStateInfo, layer_weight: Qwen3NextMTPTransformerLayerWeight
    ):
        input1 = self._att_norm(input_embdings, infer_state, layer_weight)
        q, cache_kv = self._get_qkv(input1, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_kv, infer_state, layer_weight)
        o = self._token_attention_kernel(q, infer_state, layer_weight)
        q = None
        o = self._get_o(o, infer_state, layer_weight)
        if self.tp_world_size_ > 1:
            all_reduce(o, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        input_embdings.add_(o.view(-1, self.embed_dim_))
        o = None

        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn_with_shared_expert(input1, infer_state, layer_weight)
        input1 = None
        if self.tp_world_size_ > 1:
            all_reduce(ffn_out, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return input_embdings
