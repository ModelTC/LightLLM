import torch
import torch.nn.functional as F
import torch.distributed as dist
from lightllm.models.qwen3next.layer_weights.transformer_layer_weight import Qwen3NextTransformerLayerWeight
from lightllm.models.qwen3_moe.layer_infer.transformer_layer_infer import Qwen3MOETransformerLayerInfer
from functools import partial
from lightllm.utils.log_utils import init_logger
from lightllm.common.fused_moe.moe_silu_and_mul import silu_and_mul_fwd
from lightllm.models.qwen3next.mem_manager import Qwen3NextMemoryManager
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from typing import Tuple
from typing_extensions import override
from einops import rearrange
from lightllm.models.qwen3next.triton_kernel.gated_rmsnorm import gated_rmsnorm_forward
from lightllm.models.qwen3next.triton_kernel.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from lightllm.models.qwen3next.triton_kernel.fused_gdn_gating import fused_gdn_gating
from lightllm.models.qwen3next.triton_kernel.fla.ops.chunk import chunk_gated_delta_rule
from lightllm.models.qwen3next.triton_kernel.fla.ops.fused_recurrent import fused_recurrent_gated_delta_rule
from lightllm.distributed import all_reduce
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.qwen3next.triton_kernel.gemma_rmsnorm import gemma_rmsnorm_forward
from lightllm.models.qwen3next.layer_infer.gdn_layer_infer import Qwen3NextGatedDeltaNetInfer

logger = init_logger(__name__)


class Qwen3NextTransformerLayerInfer(Qwen3MOETransformerLayerInfer):
    def __init__(self, layer_num, network_config, mode=[]):
        super().__init__(layer_num, network_config, mode)
        self.is_gdn = (layer_num + 1) % network_config["full_attention_interval"] != 0
        self.partial_rotary_factor = network_config.get("partial_rotary_factor", 1.0)

        if self.is_gdn:
            self.gdn_infer = Qwen3NextGatedDeltaNetInfer(layer_num, network_config)
        return

    @override
    def _bind_norm(self):
        pass

    def _ffn_with_shared_expert(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: Qwen3NextTransformerLayerWeight
    ) -> torch.Tensor:
        input = input.view(-1, self.embed_dim_)
        up_gate_out = layer_weight.shared_expert_gate_up_proj.mm(input)
        ffn1_out = self.alloc_tensor((input.size(0), up_gate_out.size(1) // 2), input.dtype)
        silu_and_mul_fwd(up_gate_out, ffn1_out)
        ffn2_out = layer_weight.shared_expert_down_proj.mm(ffn1_out)
        shared_expert_out = F.sigmoid(layer_weight.shared_expert_gate.mm(input)) * ffn2_out
        moe_out = self._ffn(input, infer_state, layer_weight)
        return shared_expert_out + moe_out

    @override
    def _att_norm(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: Qwen3NextTransformerLayerWeight
    ) -> torch.Tensor:
        out = self.alloc_tensor(input.shape, input.dtype)
        gemma_rmsnorm_forward(input, layer_weight.att_norm_weight_.weight, self.eps_, out=out)
        return out

    @override
    def _ffn_norm(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: Qwen3NextTransformerLayerWeight
    ) -> torch.Tensor:
        out = self.alloc_tensor(input.shape, input.dtype)
        gemma_rmsnorm_forward(input, layer_weight.ffn_norm_weight_.weight, self.eps_, out=out)
        return out

    @override
    def _get_qkv(
        self,
        input: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        layer_weight: Qwen3NextTransformerLayerWeight,
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

    @override
    def _get_o(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: Qwen3NextTransformerLayerWeight
    ) -> torch.Tensor:
        input = input * layer_weight._gate
        layer_weight._gate = None
        o_tensor = layer_weight.o_proj.mm(input)
        return o_tensor

    def _context_full_attn(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: Qwen3NextTransformerLayerWeight
    ):
        q, cache_kv = self._get_qkv(input, infer_state, layer_weight)
        input = None
        self._post_cache_kv(cache_kv, infer_state, layer_weight)
        o = self._context_attention_kernel(q, cache_kv, infer_state, layer_weight)
        q = None
        o = self._get_o(o, infer_state, layer_weight)
        if self.tp_world_size_ > 1:
            all_reduce(o, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        return o

    def context_forward(
        self, input_embdings, infer_state: LlamaInferStateInfo, layer_weight: Qwen3NextTransformerLayerWeight
    ):
        input1 = self._att_norm(input_embdings, infer_state, layer_weight)
        if self.is_gdn:
            o = self.gdn_infer.forward(input1, infer_state, layer_weight.gdn_layer_weight)
        else:
            layer_weight._gate = torch.sigmoid(layer_weight.o_gate_proj.mm(input1))
            o = self._context_full_attn(input1, infer_state, layer_weight)
        input_embdings.add_(o.view(-1, self.embed_dim_))
        o = None

        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn_with_shared_expert(input1, infer_state, layer_weight)
        input1 = None
        if self.tp_world_size_ > 1:
            all_reduce(ffn_out, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return input_embdings

    def _token_full_attn(self, input, infer_state: LlamaInferStateInfo, layer_weight: Qwen3NextTransformerLayerWeight):
        q, cache_kv = self._get_qkv(input, infer_state, layer_weight)
        input = None
        self._post_cache_kv(cache_kv, infer_state, layer_weight)
        o = self._token_attention_kernel(q, infer_state, layer_weight)
        q = None
        o = self._get_o(o, infer_state, layer_weight)
        if self.tp_world_size_ > 1:
            all_reduce(o, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        return o

    def token_forward(
        self, input_embdings, infer_state: LlamaInferStateInfo, layer_weight: Qwen3NextTransformerLayerWeight
    ):
        input1 = self._att_norm(input_embdings, infer_state, layer_weight)
        if self.is_gdn:
            o = self.gdn_infer.forward(input1, infer_state, layer_weight.gdn_layer_weight)
        else:
            layer_weight._gate = torch.sigmoid(layer_weight.o_gate_proj.mm(input1))
            o = self._token_full_attn(input1, infer_state, layer_weight)
        input_embdings.add_(o.view(-1, self.embed_dim_))
        o = None

        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn_with_shared_expert(input1, infer_state, layer_weight)
        input1 = None
        if self.tp_world_size_ > 1:
            all_reduce(ffn_out, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return input_embdings
