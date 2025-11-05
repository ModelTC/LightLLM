from sgl_kernel import fast_topk_transform_fused
import deep_gemm
import torch
import torch.nn.functional as F

from lightllm.common.basemodel.layer_infer.base_layer_infer import BaseLayerInfer
from lightllm.models.deepseek3_2.layer_weights.nsa_indexer_layer_weight import NSAIndexerWeight
from lightllm.models.deepseek3_2.infer_struct import Deepseek3_2FlashAttentionInferStateInfo
from lightllm.models.deepseek2.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.deepseek3_2.triton_kernel.act_quant import act_quant


class NSAIndexerInfer(BaseLayerInfer):
    def __init__(self, layer_idx, network_config, mode=[]):
        super().__init__()
        self.layer_idx_ = layer_idx
        self.network_config_ = network_config
        self.mode = mode
        self.index_topk = network_config["index_topk"]
        self.tp_q_head_num_ = network_config["num_attention_heads"] // self.tp_world_size_
        self.tp_k_head_num_ = 1
        self.tp_v_head_num_ = 1
        self.qk_nope_head_dim = network_config["qk_nope_head_dim"]
        self.qk_rope_head_dim = network_config["qk_rope_head_dim"]
        self.index_head_dim = network_config["index_head_dim"]
        self.eps = network_config["rms_norm_eps"]
        self.block_size = network_config["quantization_config"]["weight_block_size"][0]
        self.scale_fmt = network_config["quantization_config"]["scale_fmt"]
        self.softmax_scale = (self.qk_nope_head_dim + self.qk_rope_head_dim) ** (-0.5)
        self.index_n_heads = network_config["index_n_heads"]
        self.index_n_heads_scale = self.index_n_heads ** -0.5

        self.q_lora = None
        self.hidden_states = None
        return

    def ref_fp8_mqa_logits(self, q: torch.Tensor, kv: torch.Tensor, weights: torch.Tensor,
                        cu_seqlen_ks: torch.Tensor, cu_seqlen_ke: torch.Tensor, cost_only: bool = False):
        seq_len_kv = kv.shape[0]

        if cost_only:
            start = cu_seqlen_ks.clamp(min=0, max=seq_len_kv)
            end   = cu_seqlen_ke.clamp(min=0, max=seq_len_kv)
            count_ones_per_row = (end - start).clamp(min=0)
            return count_ones_per_row.sum()

        k = kv
        q = q.float()
        k = k.float()

        mask_lo = torch.arange(0, seq_len_kv, device='cuda')[None, :] >= cu_seqlen_ks[:, None]
        mask_hi = torch.arange(0, seq_len_kv, device='cuda')[None, :] < cu_seqlen_ke[:, None]
        mask = mask_lo & mask_hi

        score = torch.einsum('mhd,nd->hmn', q, k)
        logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
        logits = logits.masked_fill(~mask, float('-inf'))

        cost = mask.sum()
        return logits, cost

    def get_indices(self, infer_state: Deepseek3_2FlashAttentionInferStateInfo, layer_weight: NSAIndexerWeight) -> torch.Tensor:
        assert self.hidden_states is not None
        assert self.q_lora is not None

        q, k = self._get_q_k_bf16(infer_state, layer_weight)
        q_fp8, q_scale = act_quant(q, self.block_size, self.scale_fmt)
        k_fp8, k_scale = act_quant(k, self.block_size, self.scale_fmt)

        weights = layer_weight.weights_proj_.mm(self.hidden_states) * self.index_n_heads_scale
        weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale

        logits = fp8_paged_mqa_logits_torch(
            q_fp8, k_fp8, weights, 
            infer_state.lengths, 
            infer_state.page_table, 
            infer_state.max_model_len
        )

        return fast_topk_transform_fused(
            score=logits, 
            lengths=infer_state.lengths,
            page_table_size_1=infer_state.page_table,
            cu_seqlens_q=infer_state.b1_cu_q_seq_len,
            topk=self.index_topk
        )
    
    @staticmethod
    def _rotate_activation(x: torch.Tensor) -> torch.Tensor:
        assert x.dtype == torch.bfloat16
        from sgl_kernel import hadamard_transform

        hidden_size = x.size(-1)
        assert (
            hidden_size & (hidden_size - 1)
        ) == 0, "Hidden size must be a power of 2 for Hadamard transform."
        return hadamard_transform(x, scale=hidden_size**-0.5)

    def _get_q_k_bf16(self, infer_state: Deepseek3_2FlashAttentionInferStateInfo, layer_weight: NSAIndexerWeight):
        q = layer_weight.wq_b_proj_.mm(self.q_lora).view(-1, self.index_n_heads, self.index_head_dim)
        self.q_lora = None

        k = layer_weight.wk_proj_.mm(self.hidden_states)
        self.hidden_states = None
        k = F.layer_norm(
            k.float(), (self.index_head_dim,), layer_weight.k_norm_.weight, layer_weight.k_norm_.bias, self.eps
        ).type_as(k)
        
        rotary_emb_fwd(
            q[:, :, : self.qk_rope_head_dim],
            k[:, None, : self.qk_rope_head_dim],
            infer_state.position_cos,
            infer_state.position_sin,
        )

        q = self._rotate_activation(q)
        k = self._rotate_activation(k)
        return q, k


# TODO
def fp8_paged_mqa_logits_torch(q: torch.Tensor, kv_cache: torch.Tensor,
                             weights: torch.Tensor, context_lens: torch.Tensor, block_tables: torch.Tensor,
                             max_model_len: int):
    batch_size, next_n, heads, dim = q.size()
    num_block, block_size, _, dim = kv_cache.size()
    logits = torch.full([batch_size * next_n, max_model_len], float('-inf'), device=q.device, dtype=torch.float32)
    context_lens = context_lens.tolist()
    for i in range(batch_size):
        context_len = context_lens[i]
        q_offsets = torch.arange(context_len - next_n, context_len, device='cuda')
        weight_slice = weights[i * next_n:(i + 1) * next_n, :].transpose(0, 1).contiguous()
        for block_rk in range((context_len + block_size - 1) // block_size):
            block_idx = block_tables[i][block_rk]
            qx, kx = q[i], kv_cache[block_idx]
            k_offsets = torch.arange(block_rk * block_size, (block_rk + 1) * block_size, device='cuda')
            mask = (k_offsets[None, :] < context_len) & (k_offsets[None, :] <= q_offsets[:, None])
            s = torch.where(mask[None, :, :], (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(logits.dtype), float('-inf'))
            s = torch.relu(s) * weight_slice[..., None]
            s = s.sum(dim=0)
            logits[i * next_n:(i + 1) * next_n, block_rk * block_size: (block_rk + 1) * block_size] = torch.where(k_offsets[None, :] <= q_offsets[:, None], s, float('-inf'))
    return logits