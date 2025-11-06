from sgl_kernel import fast_topk_transform_fused
import deep_gemm
import torch
import torch.nn.functional as F

from lightllm.common.basemodel.layer_infer.base_layer_infer import BaseLayerInfer
from lightllm.models.deepseek3_2.layer_weights.nsa_indexer_layer_weight import NSAIndexerWeight
from lightllm.models.deepseek3_2.infer_struct import Deepseek3_2FlashAttentionStateInfo
from lightllm.models.deepseek2.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.deepseek3_2.triton_kernel.act_quant import act_quant
from lightllm.models.deepseek3_2.mem_manager import Deepseek3_2MemoryManager
from lightllm.models.deepseek3_2.triton_kernel.destindex_copy_indexer_ks import destindex_copy_indexer_ks
# from lightllm.models.deepseek3_2.triton_kernel.fp8_mqa_logits import fp8_mqa_logits

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

    def get_indices(self, infer_state: Deepseek3_2FlashAttentionStateInfo, layer_weight: NSAIndexerWeight) -> torch.Tensor:
        assert self.hidden_states is not None
        assert self.q_lora is not None

        q, k = self._get_q_k_bf16(infer_state, layer_weight)
        q_fp8, q_scale = act_quant(q, self.block_size, self.scale_fmt)
        k_fp8, k_scale = act_quant(k, self.block_size, self.scale_fmt)

        self._copy_ks_to_mem_cache(k_fp8, k_scale, infer_state.mem_index, infer_state.mem_manager)

        weights = layer_weight.weights_proj_.mm(self.hidden_states) * self.index_n_heads_scale
        weights = weights.unsqueeze(-1) * q_scale

        ks_buffer = infer_state.mem_manager.indexer_ks_mem_manager.kv_buffer[self.layer_idx_]

        k_fp8_list = []
        k_scale_list = []
        ks_list = []
        ke_list = []
        offset = 0
        for i in range(infer_state.batch_size):
            q_len = infer_state.b_q_seq_len[i]
            cache_len = infer_state.b_ready_cache_len[i]
            mem_indexes = infer_state.req_manager.req_to_token_indexs[infer_state.b_req_idx[i], :cache_len+q_len]
            k_fp8 = ks_buffer[mem_indexes, 0, :128].view(torch.float8_e4m3fn).contiguous()
            k_scale = ks_buffer[mem_indexes, 0, 128:].view(torch.float32).contiguous()
            ks = torch.full((q_len,), offset, dtype=torch.int32, device="cuda")
            ke = ks + torch.arange(q_len, dtype=torch.int32, device="cuda") + 1
            k_fp8_list.append(k_fp8)
            k_scale_list.append(k_scale)
            ks_list.append(ks)
            ke_list.append(ke)
            offset += q_len 

        k_fp8 = torch.cat(k_fp8_list, dim=0).view(torch.float8_e4m3fn)
        k_scale = torch.cat(k_scale_list, dim=0).view(torch.float32).squeeze(-1)
        kv_fp8 = (k_fp8, k_scale)
        ks = torch.cat(ks_list, dim=0)
        ke = torch.cat(ke_list, dim=0)

        logits = deep_gemm.fp8_mqa_logits(
            q_fp8,
            kv_fp8,
            weights.squeeze(-1),
            ks,
            ke,
            clean_logits=False,
        )

        return self.get_topk(logits, infer_state)

    def get_topk(self, logits, infer_state: Deepseek3_2FlashAttentionStateInfo):
        topk_indices_list = []
        offset = 0

        for i in range(infer_state.batch_size):
            q_len = infer_state.b_q_seq_len[i]
            cache_len = infer_state.b_ready_cache_len[i]
            end_pos = q_len + cache_len
            # Slice logits for this batch (both query and sequence dimensions)
            batch_logits = logits[offset:offset + q_len, :end_pos]
            topk_indices = batch_logits.topk(min(self.index_topk, end_pos), dim=-1)[1]
            mem_indexes = infer_state.req_manager.req_to_token_indexs[infer_state.b_req_idx[i], :cache_len+q_len]
            indices = torch.full((q_len, self.index_topk), -1, dtype=torch.int32, device="cuda")
            for j in range(q_len):
                indices[j, :topk_indices[j].shape[0]] = mem_indexes[topk_indices[j]]
            topk_indices_list.append(indices)
            offset += q_len

        topk_indices_ = torch.cat(topk_indices_list, dim=0)

        return topk_indices_


    def get_k_float32_from_buffer(self, buffer: torch.Tensor):
        k_fp8 = buffer[:, :, :128].view(torch.float8_e4m3fn)
        k_scale = buffer[:, :, 128:].view(torch.float32)[:, :, :1]
        k_float32 = k_fp8.float() * k_scale
        return k_float32

    @staticmethod
    def _rotate_activation(x: torch.Tensor) -> torch.Tensor:
        assert x.dtype == torch.bfloat16
        from sgl_kernel import hadamard_transform

        hidden_size = x.size(-1)
        assert (
            hidden_size & (hidden_size - 1)
        ) == 0, "Hidden size must be a power of 2 for Hadamard transform."
        return hadamard_transform(x, scale=hidden_size**-0.5)

    def _get_q_k_bf16(self, infer_state: Deepseek3_2FlashAttentionStateInfo, layer_weight: NSAIndexerWeight):
        q = layer_weight.wq_b_proj_.mm(self.q_lora).view(-1, self.index_n_heads, self.index_head_dim)
        self.q_lora = None

        k = layer_weight.wk_proj_.mm(self.hidden_states)
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

    def _copy_ks_to_mem_cache(self, k_fp8, k_scale, mem_index, mem_manager: Deepseek3_2MemoryManager):
        # k_fp8 : [seq_len, 128] torch.fp8_e4m3
        # k_scale : [seq_len, 1] torch.float32
        # mem_index : [seq_len] torch.int32
        # buffer : [10000000, 1, 132] torch.uint8
        buffer = mem_manager.indexer_ks_mem_manager.kv_buffer[self.layer_idx_]
        destindex_copy_indexer_ks(
            k_fp8.unsqueeze(1),  # Add head dimension: [seq_len, 1, 128]
            k_scale.unsqueeze(1),  # Add head dimension: [seq_len, 1, 1]
            mem_index,
            buffer
        )
        return