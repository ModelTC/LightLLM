from sgl_kernel import fast_topk_transform_fused
import deep_gemm
import torch
import torch.nn.functional as F

from lightllm.common.basemodel.layer_infer.base_layer_infer import BaseLayerInfer
from lightllm.models.deepseek3_2.layer_weights.nsa_indexer_layer_weight import NSAIndexerWeight
from lightllm.models.deepseek3_2.infer_struct import Deepseek3_2FlashAttentionStateInfo
from lightllm.models.deepseek2.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.deepseek3_2.triton_kernel.act_quant import act_quant
from lightllm.models.deepseek3_2.triton_kernel.destindex_copy_indexer_ks import destindex_copy_indexer_ks
from lightllm.models.deepseek3_2.triton_kernel.extract_indexer_ks import extract_indexer_ks
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


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
        self.index_n_heads_scale = (self.index_n_heads ** -0.5) * self.softmax_scale

        return

    def ref_fp8_mqa_logits(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        weights: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        cost_only: bool = False,
    ):
        seq_len_kv = kv.shape[0]

        if cost_only:
            start = cu_seqlen_ks.clamp(min=0, max=seq_len_kv)
            end = cu_seqlen_ke.clamp(min=0, max=seq_len_kv)
            count_ones_per_row = (end - start).clamp(min=0)
            return count_ones_per_row.sum()

        k = kv
        q = q.float()
        k = k.float()

        mask_lo = torch.arange(0, seq_len_kv, device="cuda")[None, :] >= cu_seqlen_ks[:, None]
        mask_hi = torch.arange(0, seq_len_kv, device="cuda")[None, :] < cu_seqlen_ke[:, None]
        mask = mask_lo & mask_hi

        score = torch.einsum("mhd,nd->hmn", q, k)
        logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
        logits = logits.masked_fill(~mask, float("-inf"))

        cost = mask.sum()
        return logits, cost

    def get_indices(
        self,
        hidden_states: torch.Tensor,
        q_lora: torch.Tensor,
        infer_state: Deepseek3_2FlashAttentionStateInfo,
        layer_weight: NSAIndexerWeight,
    ) -> torch.Tensor:

        q, k = self._get_q_k_bf16(hidden_states, q_lora, infer_state, layer_weight)
        q_fp8, q_scale = act_quant(q, self.block_size, self.scale_fmt)
        k_fp8, k_scale = act_quant(k, self.block_size, self.scale_fmt)

        destindex_copy_indexer_ks(
            k_fp8, k_scale, infer_state.mem_index, infer_state.indexer_ks_buffer.kv_buffer[self.layer_idx_]
        )

        weights = layer_weight.weights_proj_.mm(hidden_states) * self.index_n_heads_scale
        weights = weights.unsqueeze(-1) * q_scale

        ks = infer_state.ks
        ke = infer_state.ke
        lengths = infer_state.lengths
        page_table_1 = infer_state.page_table_size_1

        # Use efficient Triton kernel to extract FP8 keys and scales from buffer
        k_fp8_, k_scale_ = extract_indexer_ks(
            infer_state.indexer_ks_buffer.kv_buffer[self.layer_idx_], infer_state.req_all_mem_index
        )

        # Get actual sequence length from q (which comes from q_lora)
        # This may differ from ks.shape[0] during certain operations
        actual_seq_len = q.shape[0]

        # ks, ke, lengths, and weights should all match actual_seq_len
        # Slice them if they don't match
        if ks.shape[0] != actual_seq_len:
            ks = ks[:actual_seq_len]
            ke = ke[:actual_seq_len]
            lengths = lengths[:actual_seq_len]
            weights = weights[:actual_seq_len]

        logits = deep_gemm.fp8_mqa_logits(q_fp8, (k_fp8_, k_scale_), weights.squeeze(-1), ks, ke)

        return fast_topk_transform_fused(
            score=logits,
            lengths=lengths,
            page_table_size_1=page_table_1,
            cu_seqlens_q=infer_state.cu_seqlens_q,
            topk=self.index_topk,
        )

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
        infer_state: Deepseek3_2FlashAttentionStateInfo,
        layer_weight: NSAIndexerWeight,
    ):
        q = layer_weight.wq_b_proj_.mm(q_lora).view(-1, self.index_n_heads, self.index_head_dim)
        k = layer_weight.wk_proj_.mm(hidden_states)

        # TODO
        k = F.layer_norm(
            k.float(), (self.index_head_dim,), layer_weight.k_norm_.weight, layer_weight.k_norm_.bias, self.eps
        ).type_as(k)

        # Slice position_cos and position_sin to match actual token length
        actual_seq_len = q.shape[0]
        rotary_emb_fwd(
            q[:, :, : self.qk_rope_head_dim],
            k[:, None, : self.qk_rope_head_dim],
            infer_state.position_cos[:actual_seq_len],
            infer_state.position_sin[:actual_seq_len],
        )

        q = self._rotate_activation(q)
        k = self._rotate_activation(k)
        return q, k
