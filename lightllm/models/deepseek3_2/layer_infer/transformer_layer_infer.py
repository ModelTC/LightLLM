from functools import partial
from typing import override

import torch

from lightllm.models.deepseek2.layer_infer.transformer_layer_infer import Deepseek2TransformerLayerInfer
from lightllm.models.deepseek3_2.layer_infer.nsa_indexer_layer_inder import NSAIndexerInfer
from lightllm.models.deepseek3_2.layer_weights.transformer_layer_weight import Deepseek3_2TransformerLayerWeight
from lightllm.models.deepseek3_2.infer_struct import Deepseek3_2FlashAttentionStateInfo
from lightllm.models.deepseek3_2.triton_kernel.token_group_quant import per_token_group_quant_mla_deep_gemm_masked_fp8
from lightllm.models.llama.triton_kernel.rmsnorm import rmsnorm_forward
from lightllm.models.deepseek2.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.common.basemodel.attention.base_att import AttControl
from lightllm.common.basemodel.attention.create_utils import get_nsa_prefill_att_backend_class


class Deepseek3_2TransformerLayerInfer(Deepseek2TransformerLayerInfer):
    def __init__(self, layer_num, network_config, mode=[]):
        self.index_topk = network_config["index_topk"]
        super().__init__(layer_num, network_config, mode)

        self.indexer = NSAIndexerInfer(layer_idx=self.layer_num_, network_config=self.network_config_, mode=mode)
        self.topk_indices = None

        # Initialize NSA attention backend (singleton, lazy initialization)
        self._nsa_backend_class = get_nsa_prefill_att_backend_class()
        self._nsa_backend = None
        return

    def _get_nsa_backend(self):
        """Get or create the NSA backend (lazy initialization)."""
        if self._nsa_backend is None:
            # NSA backend doesn't require model reference for basic operations
            self._nsa_backend = self._nsa_backend_class(model=None)
        return self._nsa_backend

    @override
    def _get_qkv(
        self,
        input: torch.Tensor,
        infer_state: Deepseek3_2FlashAttentionStateInfo,
        layer_weight: Deepseek3_2TransformerLayerWeight,
    ) -> torch.Tensor:
        input = input.view(-1, self.embed_dim_)

        q, cache_kv = layer_weight.qkv_a_proj_with_mqa_.mm(input).split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
        )
        q = rmsnorm_forward(q, weight=layer_weight.q_a_layernorm_.weight, eps=self.eps_)

        # Process all tokens for indexer
        # Note: Prefix cache slicing optimization is disabled due to batch structure
        # mismatch issues with fast_topk_transform_fused kernel
        self.topk_indices = self.indexer.get_indices(input, q, infer_state, layer_weight.indexer_layer_weight)

        q = layer_weight.q_b_proj_.mm(q)
        cache_kv = cache_kv.view(-1, 1, self.kv_lora_rank + self.qk_rope_head_dim)
        q = q.view(-1, self.tp_q_head_num_, self.qk_nope_head_dim + self.qk_rope_head_dim)
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        rmsnorm_forward(
            cache_kv[:, :, : self.kv_lora_rank],
            weight=layer_weight.kv_a_layernorm_.weight,
            eps=self.eps_,
            out=cache_kv[:, :, : self.kv_lora_rank],
        )

        rotary_emb_fwd(
            q_rope,
            cache_kv[:, :, self.kv_lora_rank :],
            infer_state.position_cos,
            infer_state.position_sin,
        )
        return q, cache_kv

    @override
    def _bind_attention(self):
        if "triton_fp8kv" in self.mode:
            self._copy_kv_to_mem_cache = partial(Deepseek2TransformerLayerInfer._copy_kv_to_mem_cache_fp8, self)
        else:
            self._copy_kv_to_mem_cache = partial(Deepseek2TransformerLayerInfer._copy_kv_to_mem_cache_normal, self)

        self._context_attention_kernel = partial(Deepseek3_2TransformerLayerInfer._nsa_context_attention_kernel, self)
        self._token_attention_kernel = partial(Deepseek3_2TransformerLayerInfer._nsa_token_attention_kernel, self)
        pass

    def _nsa_context_attention_kernel(
        self,
        q: torch.Tensor,
        kv,
        infer_state: Deepseek3_2FlashAttentionStateInfo,
        layer_weight: Deepseek3_2TransformerLayerWeight,
        out=None,
    ) -> torch.Tensor:
        # Model-specific q projection (uses layer weights)
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_nope = layer_weight.k_b_proj_.bmm(q_nope.transpose(0, 1)).transpose(0, 1)
        q_all = torch.cat([q_nope, q_rope], dim=-1)

        # Use NSA backend for attention computation
        att_control = AttControl(
            nsa_prefill=True,
            nsa_prefill_dict={
                "topk_indices": self.topk_indices,
                "softmax_scale": self.softmax_scale,
                "kv_lora_rank": self.kv_lora_rank,
            },
        )

        # Create prefill state and execute attention
        nsa_backend = self._get_nsa_backend()
        prefill_state = nsa_backend.create_att_prefill_state(infer_state)
        prefill_state.init_state()
        mla_out = prefill_state.prefill_att(
            q=q_all,
            k=infer_state.mem_manager.kv_buffer[self.layer_num_],
            v=None,
            att_control=att_control,
        )
        return mla_out

    def _nsa_token_attention_kernel(
        self,
        q,
        infer_state: Deepseek3_2FlashAttentionStateInfo,
        layer_weight: Deepseek3_2TransformerLayerWeight,
        out=None,
    ):
        # Model-specific q projection (uses layer weights)
        q_nope, q_rope = q[:, :, : -self.qk_rope_head_dim], q[:, :, -self.qk_rope_head_dim :]
        q_nope = layer_weight.k_b_proj_.bmm(q_nope.transpose(0, 1)).transpose(0, 1)

        # Use NSA backend for attention computation
        att_control = AttControl(
            nsa_decode=True,
            nsa_decode_dict={
                "topk_indices": self.topk_indices,
                "nsa_cache_seqlens": infer_state.nsa_cache_seqlens,
                "nsa_cu_seqlens_k": infer_state.nsa_cu_seqlens_k,
                "softmax_scale": self.softmax_scale,
                "kv_lora_rank": self.kv_lora_rank,
                "qk_rope_head_dim": self.qk_rope_head_dim,
            },
        )

        # Create decode state and execute attention
        nsa_backend = self._get_nsa_backend()
        decode_state = nsa_backend.create_att_decode_state(infer_state)
        decode_state.init_state()
        o_tensor = decode_state.decode_att(
            q=(q_nope, q_rope),
            k=infer_state.mem_manager.kv_buffer[self.layer_num_],
            v=None,
            att_control=att_control,
        )
        return o_tensor
