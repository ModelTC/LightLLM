import torch
from typing import Union
from lightllm.models.deepseek2.infer_struct import Deepseek2InferStateInfo
from lightllm.models.deepseek2.layer_infer.transformer_layer_infer import Deepseek2TransformerLayerInfer
from lightllm.models.deepseek3_2.layer_weights.transformer_layer_weight import Deepseek3_2TransformerLayerWeight
from lightllm.common.basemodel.triton_kernel.norm.rmsnorm import rmsnorm_forward
from lightllm.models.deepseek2.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.common.basemodel.attention.base_att import AttControl
from lightllm.common.basemodel.attention.nsa import NsaFlashMlaSparsePrefillAttState, NsaFlashMlaSparseDecodeAttState
from lightllm.models.deepseek3_2.triton_kernel.act_quant import act_quant
from lightllm.models.deepseek3_2.triton_kernel.destindex_copy_indexer_ks import destindex_copy_indexer_ks
from lightllm.models.deepseek3_2.triton_kernel.extract_indexer_ks import extract_indexer_ks
from lightllm.utils.envs_utils import get_env_start_args


class Deepseek3_2TransformerLayerInfer(Deepseek2TransformerLayerInfer):
    def __init__(self, layer_num, network_config):
        self.index_topk = network_config["index_topk"]
        super().__init__(layer_num, network_config)

        self.indexer = NsaInfer(layer_idx=self.layer_num_, network_config=self.network_config_)
        return

    def _get_qkv(
        self,
        input: torch.Tensor,
        infer_state: Deepseek2InferStateInfo,
        layer_weight: Deepseek3_2TransformerLayerWeight,
    ) -> torch.Tensor:
        input = input.view(-1, self.embed_dim_)

        q, cache_kv = layer_weight.qkv_a_proj_with_mqa_.mm(input).split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
        )
        q = rmsnorm_forward(q, weight=layer_weight.q_a_layernorm_.weight, eps=self.eps_)

        infer_state.topk_indices = self.indexer.get_indices(input, q, infer_state, layer_weight)

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

    def _context_attention_kernel(
        self,
        q: torch.Tensor,
        kv,
        infer_state: Deepseek2InferStateInfo,
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
                "topk_indices": infer_state.topk_indices,
                "softmax_scale": self.softmax_scale,
                "kv_lora_rank": self.kv_lora_rank,
            },
        )

        del infer_state.topk_indices

        mla_out = infer_state.prefill_att_state.prefill_att(
            q=q_all,
            k=infer_state.mem_manager.kv_buffer[self.layer_num_],
            v=None,
            att_control=att_control,
        )
        return mla_out

    def _token_attention_kernel(
        self,
        q,
        infer_state: Deepseek2InferStateInfo,
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
                "topk_indices": infer_state.topk_indices,
                "softmax_scale": self.softmax_scale,
                "kv_lora_rank": self.kv_lora_rank,
                "qk_rope_head_dim": self.qk_rope_head_dim,
            },
        )

        del infer_state.topk_indices

        o_tensor = infer_state.decode_att_state.decode_att(
            q=(q_nope, q_rope),
            k=infer_state.mem_manager.kv_buffer[self.layer_num_],
            v=None,
            att_control=att_control,
        )
        return o_tensor


class NsaInfer:
    def __init__(self, layer_idx: int, network_config: dict):
        super().__init__()
        self.layer_idx_ = layer_idx
        self.network_config_ = network_config
        self.index_topk = network_config["index_topk"]
        self.qk_nope_head_dim = network_config["qk_nope_head_dim"]
        self.qk_rope_head_dim = network_config["qk_rope_head_dim"]
        self.index_head_dim = network_config["index_head_dim"]
        self.eps = network_config["rms_norm_eps"]
        self.block_size = network_config["quantization_config"]["weight_block_size"][0]
        self.scale_fmt = network_config["quantization_config"]["scale_fmt"]
        self.softmax_scale = (self.qk_nope_head_dim + self.qk_rope_head_dim) ** (-0.5)
        self.index_n_heads = network_config["index_n_heads"]
        self.index_n_heads_scale = (self.index_n_heads ** -0.5) * self.softmax_scale

    def get_indices(
        self,
        hidden_states: torch.Tensor,
        q_lora: torch.Tensor,
        infer_state: Deepseek2InferStateInfo,
        att_state: Union[NsaFlashMlaSparsePrefillAttState, NsaFlashMlaSparseDecodeAttState],
        layer_weight: Deepseek3_2TransformerLayerWeight,
    ) -> torch.Tensor:

        q, k = self._get_q_k_bf16(hidden_states, q_lora, infer_state, layer_weight)
        q_fp8, q_scale = act_quant(q, self.block_size, self.scale_fmt)
        k_fp8, k_scale = act_quant(k, self.block_size, self.scale_fmt)

        destindex_copy_indexer_ks(
            k_fp8, k_scale, infer_state.mem_index, infer_state.mem_manager.indexer_ks_buffer.kv_buffer[self.layer_idx_]
        )

        weights = layer_weight.weights_proj_.mm(hidden_states) * self.index_n_heads_scale
        weights = weights.unsqueeze(-1) * q_scale

        ks = att_state.ks
        ke = att_state.ke
        lengths = att_state.lengths

        if infer_state.is_prefill:
            mtp_step = 0
        else:
            mtp_step = get_env_start_args().mtp_step
        # Use efficient Triton kernel to extract FP8 keys and scales from buffer
        k_fp8_, k_scale_ = extract_indexer_ks(
            I_buffer=infer_state.mem_manager.indexer_ks_buffer.kv_buffer[self.layer_idx_],
            b_seq_len=infer_state.b_seq_len,
            b_req_idx=infer_state.b_req_idx,
            req_to_token_indexs=infer_state.req_manager.req_to_token_indexs,
            out_token_num=infer_state.b_seq_len.shape[0] * infer_state.max_kv_seq_len,
            max_kv_seq_len=infer_state.max_kv_seq_len,
            mtp_step=mtp_step,
        )

        import deep_gemm

        logits = deep_gemm.fp8_mqa_logits(q_fp8, (k_fp8_, k_scale_), weights.squeeze(-1), ks, ke)

        from sgl_kernel import fast_topk_v2

        b_topk_index = fast_topk_v2(
            score=logits,
            lengths=lengths,
            topk=self.index_topk,
            row_starts=ks,
        )
        # 将 topk index 转化为 mem index

        from ..triton_kernel.topk_index_to_mem_index import trans_topk_index_to_mem_index

        b_topk_index = trans_topk_index_to_mem_index(
            topk_index=b_topk_index,
            ragged_mem_index=att_state.ragged_mem_index,
        )

        return b_topk_index

    @staticmethod
    def _rotate_activation(x: torch.Tensor) -> torch.Tensor:
        assert x.dtype == torch.bfloat16
        from sgl_kernel import hadamard_transform

        hidden_size = x.size(-1)
        assert (hidden_size & (hidden_size - 1)) == 0, "Hidden size must be a power of 2 for Hadamard transform."
        return hadamard_transform(x, scale=hidden_size ** -0.5)

    def _get_q_k_bf16(
        self,
        hidden_states: torch.Tensor,
        q_lora: torch.Tensor,
        infer_state: Deepseek2InferStateInfo,
        layer_weight: Deepseek3_2TransformerLayerWeight,
    ):
        q = layer_weight.wq_b_proj_.mm(q_lora).view(-1, self.index_n_heads, self.index_head_dim)
        k = layer_weight.wk_proj_.mm(hidden_states)

        k = layer_weight.k_norm_(k, eps=self.eps)

        rotary_emb_fwd(
            q[:, :, : self.qk_rope_head_dim],
            k[:, None, : self.qk_rope_head_dim],
            infer_state.position_cos,
            infer_state.position_sin,
        )

        q = self._rotate_activation(q)
        k = self._rotate_activation(k)
        return q, k
