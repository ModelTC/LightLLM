from typing_extensions import Self
import torch
import torch.distributed as dist
from einops import rearrange

from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.qwen3next.mem_manager import Qwen3NextMemoryManager
from lightllm.models.qwen3next.triton_kernel.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from lightllm.models.qwen3next.triton_kernel.fused_gdn_gating import fused_gdn_gating
from lightllm.models.qwen3next.triton_kernel.fla.ops.chunk import chunk_gated_delta_rule
from lightllm.models.qwen3next.triton_kernel.fla.ops.fused_recurrent import fused_recurrent_gated_delta_rule
from lightllm.distributed import all_reduce
from lightllm.models.qwen3next.triton_kernel.gated_rmsnorm import gated_rmsnorm_forward
from lightllm.common.basemodel.layer_infer.base_layer_infer import BaseLayerInfer
from lightllm.models.qwen3next.layer_weights.gdn_layer_weight import Qwen3NextGatedDeltaNetWeight


class Qwen3NextGatedDeltaNetInfer(BaseLayerInfer):
    def __init__(self, layer_idx, network_config):
        super().__init__()
        self.network_config_ = network_config
        self.layer_idx_ = layer_idx
        self.hidden_size = self.network_config_["hidden_size"]
        self.num_v_heads = self.network_config_["linear_num_value_heads"]
        self.num_k_heads = self.network_config_["linear_num_key_heads"]
        self.head_k_dim = self.network_config_["linear_key_head_dim"]
        self.head_v_dim = self.network_config_["linear_value_head_dim"]
        self.eps_ = self.network_config_["rms_norm_eps"]
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_dim = self.network_config_["linear_conv_kernel_dim"]
        self.activation = self.network_config_["hidden_act"]
        self.tp_qkvz_dim = (self.key_dim * 2 + self.value_dim * 2) // self.tp_world_size_
        self.tp_ba_dim = (self.num_v_heads * 2) // self.tp_world_size_
        self.tp_num_k_heads = self.num_k_heads // self.tp_world_size_
        self.tp_num_v_heads = self.num_v_heads // self.tp_world_size_
        self.tp_key_dim = self.key_dim // self.tp_world_size_
        self.tp_value_dim = self.value_dim // self.tp_world_size_
        assert self.num_v_heads % self.num_k_heads == 0, "num_v_heads must be divisible by num_k_heads"
        self.num_v_heads_per_k_head = self.num_v_heads // self.num_k_heads

    def _fix_query_key_value_ba_ordering(self, mixed_qkvzba):
        """
        Derives `query`, `key` and `value` tensors from `mixed_qkvzba`.
        """
        mixed_qkvz, mixed_ba = torch.split(mixed_qkvzba, [self.tp_qkvz_dim, self.tp_ba_dim], dim=-1)

        mixed_qkvz = mixed_qkvz.view(
            -1,
            self.tp_num_k_heads,
            self.head_k_dim + self.head_k_dim + (self.head_v_dim + self.head_v_dim) * self.num_v_heads_per_k_head,
        )
        mixed_ba = mixed_ba.view(-1, self.tp_num_k_heads, 2 * self.num_v_heads_per_k_head)

        qkvz_split_list = [
            self.head_k_dim,
            self.head_k_dim,
            (self.num_v_heads_per_k_head * self.head_v_dim),
            (self.num_v_heads_per_k_head * self.head_v_dim),
        ]
        (query, key, value, z) = torch.split(mixed_qkvz, qkvz_split_list, dim=2)
        (b, a) = torch.split(mixed_ba, [self.num_v_heads_per_k_head, self.num_v_heads_per_k_head], dim=2)

        query = query.reshape(-1, self.tp_num_k_heads * self.head_k_dim)
        key = key.reshape(-1, self.tp_num_k_heads * self.head_k_dim)
        value = value.reshape(-1, self.tp_num_v_heads * self.head_v_dim)
        z = z.reshape(-1, self.tp_num_v_heads, self.head_v_dim)
        b = b.reshape(-1, self.tp_num_v_heads)
        a = a.reshape(-1, self.tp_num_v_heads)

        return query, key, value, z, b, a

    def _rearrange_mixed_qkv(self, mixed_qkv):
        if mixed_qkv is None:
            return None, None, None
        query, key, value = torch.split(
            mixed_qkv,
            [self.tp_key_dim, self.tp_key_dim, self.tp_value_dim],
            dim=-1,
        )
        query, key = map(lambda x: rearrange(x, "l (h d) -> 1 l h d", d=self.head_k_dim), (query, key))
        value = rearrange(value, "l (h d) -> 1 l h d", d=self.head_v_dim)
        return query, key, value

    def forward(
        self,
        input: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        layer_weight: Qwen3NextGatedDeltaNetWeight,
    ):
        assert isinstance(infer_state.mem_manager, Qwen3NextMemoryManager)
        input = input.view(-1, self.hidden_size)

        conv_states, ssm_states = infer_state.mem_manager.get_mamba_state_buffer(self.layer_idx_)
        cache_indices = infer_state.b_req_idx

        mixed_qkvzba = layer_weight.in_proj.mm(input)
        q, k, v, z, b, a = self._fix_query_key_value_ba_ordering(mixed_qkvzba)
        mixed_qkv = torch.cat([q, k, v], dim=-1)

        if infer_state.is_prefill:
            mixed_qkv = mixed_qkv.transpose(0, 1)
            out_tensor = self.alloc_tensor(mixed_qkv.shape, mixed_qkv.dtype, device=mixed_qkv.device)
            causal_conv1d_fn(
                mixed_qkv,
                layer_weight.conv1d.weight.transpose(0, 1),
                layer_weight.conv1d.bias,
                conv_states.transpose(1, 2),
                infer_state.b1_cu_q_seq_len,
                out=out_tensor,
                cache_indices=cache_indices,
                activation=self.activation,
            )
            mixed_qkv = out_tensor.transpose(0, 1)
        else:
            mixed_qkv = causal_conv1d_update(
                mixed_qkv,
                conv_states.transpose(1, 2),
                layer_weight.conv1d.weight.transpose(0, 1),
                layer_weight.conv1d.bias,
                self.activation,
                conv_state_indices=cache_indices,
                validate_data=True,
            )

        # Rearrange mixed_qkv to query, key, value
        query, key, value = self._rearrange_mixed_qkv(mixed_qkv)

        # Compute beta and g
        beta = b.sigmoid()
        g = fused_gdn_gating(layer_weight.A_log.weight, a, layer_weight.dt_bias.weight)
        g, beta = map(lambda x: rearrange(x, "l d -> 1 l d"), (g, beta))

        if infer_state.is_prefill:
            initial_state = ssm_states[cache_indices].contiguous()
            (core_attn_out, last_recurrent_state,) = chunk_gated_delta_rule(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=infer_state.b1_cu_q_seq_len,
                head_first=False,
                use_qk_l2norm_in_kernel=True,
            )
            # Update SSM state with final state
            ssm_states[cache_indices, ...] = last_recurrent_state.to(ssm_states.dtype)
        else:
            batch_size = input.shape[0]
            cu_seqlens = torch.arange(0, batch_size + 1, dtype=torch.int32, device=input.device)
            (core_attn_out, last_recurrent_state,) = fused_recurrent_gated_delta_rule(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                initial_state=ssm_states,
                inplace_final_state=True,
                cu_seqlens=cu_seqlens,
                ssm_state_indices=cache_indices,
                use_qk_l2norm_in_kernel=True,
            )

        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        norm_out = self.alloc_tensor(core_attn_out.shape, core_attn_out.dtype, device=core_attn_out.device)
        gated_rmsnorm_forward(
            core_attn_out,
            layer_weight.norm.weight,
            layer_weight.norm.bias,
            self.eps_,
            z,
            out=norm_out,
        )
        core_attn_out = norm_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")

        output = layer_weight.out_proj.mm(core_attn_out)
        if self.tp_world_size_ > 1:
            all_reduce(output, group=infer_state.dist_group, op=dist.ReduceOp.SUM, async_op=False)
        return output
