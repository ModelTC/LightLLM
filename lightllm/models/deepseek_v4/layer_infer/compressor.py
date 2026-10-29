import torch
import triton

from lightllm.common.basemodel import BaseLayerInfer
from ..infer_struct import DeepseekV4InferStateInfo
from ..layer_weights.transformer_layer_weight import DeepseekV4TransformerLayerWeight
from ..triton_kernel.compressor import apply_ape, _fused_compress_norm_rope_kernel, _save_compressor_state_kernel


class CompressorInfer(BaseLayerInfer):
    """Compress latent KV into packed KV, or emit BF16 C4 indexer keys.

    Both paths share the compression kernel and keep unfinished groups in request state.
    """

    def __init__(self, layer_idx: int, network_config: dict, tp_world_size: int, is_in_indexer: bool = False):
        super().__init__()
        self.layer_idx_ = layer_idx
        self.tp_world_size_ = tp_world_size
        self.is_in_indexer = is_in_indexer
        self.compress_ratio = network_config["compress_ratios"][layer_idx]
        self.head_dim = network_config["index_head_dim" if is_in_indexer else "head_dim"]
        self.qk_rope_head_dim = network_config["qk_rope_head_dim"]
        self.eps = network_config["rms_norm_eps"]

    def compress(
        self,
        x: torch.Tensor,
        infer_state: DeepseekV4InferStateInfo,
        layer_weight: DeepseekV4TransformerLayerWeight,
        cos_table: torch.Tensor,
        sin_table: torch.Tensor,
        use_custom_tensor_manager: bool = True,
        kv_score_out: torch.Tensor = None,
        out_buffer: torch.Tensor = None,
    ):
        if self.compress_ratio == 0:
            return None
        if self.is_in_indexer:
            projection = layer_weight.idx_cmp_wkv_gate_
            norm_weight = layer_weight.idx_cmp_norm_.weight
            ape = layer_weight.idx_cmp_ape_.weight
        else:
            projection = layer_weight.compressor_wkv_gate_
            norm_weight = layer_weight.compressor_norm_.weight
            ape = layer_weight.compressor_ape_.weight
        if kv_score_out is None:
            shape = (x.shape[0], self.head_dim * (4 if self.compress_ratio == 4 else 2))
            allocate = self.alloc_tensor if use_custom_tensor_manager else torch.empty
            kv_score_out = allocate(shape, dtype=torch.float32, device=x.device)
        kv_score = projection.mm(x, out=kv_score_out, use_custom_tensor_mananger=use_custom_tensor_manager)
        apply_ape(
            kv_score=kv_score,
            position_ids=infer_state.position_ids,
            ape=ape,
            compress_ratio=self.compress_ratio,
        )
        if self.is_in_indexer and use_custom_tensor_manager and out_buffer is None:
            out_buffer = self.alloc_tensor(
                (infer_state.mem_index.numel(), self.head_dim),
                torch.bfloat16,
                device=infer_state.mem_index.device,
            )
        return fused_compress(
            kv_score=kv_score,
            infer_state=infer_state,
            layer_idx=self.layer_idx_,
            norm_weight=norm_weight,
            eps=self.eps,
            head_dim=self.head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            compress_ratio=self.compress_ratio,
            cos_table=cos_table,
            sin_table=sin_table,
            is_in_indexer=self.is_in_indexer,
            out_buffer=out_buffer,
        )


def fused_compress(
    *,
    kv_score: torch.Tensor,
    infer_state,
    layer_idx: int,
    norm_weight: torch.Tensor,
    eps: float,
    head_dim: int,
    qk_rope_head_dim: int,
    compress_ratio: int,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    is_in_indexer: bool = False,
    out_buffer: torch.Tensor = None,
):
    mem_manager = infer_state.mem_manager
    req_manager = infer_state.req_manager
    compressed_kv_slots = torch.where(
        (infer_state.position_ids + 1) % compress_ratio == 0,
        infer_state.mem_index.reshape(-1) // compress_ratio,
        -1,
    )
    if is_in_indexer:
        req_to_compressor_state = req_manager.req_to_c4_indexer_state[
            req_manager.cache_config.layer_to_c4_idx[layer_idx]
        ]
        if out_buffer is None:
            out_buffer = torch.empty(
                (infer_state.mem_index.numel(), req_manager.cache_config.indexer_head_dim),
                dtype=torch.bfloat16,
                device=infer_state.mem_index.device,
            )
        out_page_size = 1
    else:
        if compress_ratio == 4:
            req_to_compressor_state = req_manager.req_to_c4_state[req_manager.cache_config.layer_to_c4_idx[layer_idx]]
        else:
            req_to_compressor_state = req_manager.req_to_c128_state[
                req_manager.cache_config.layer_to_c128_idx[layer_idx]
            ]
        out_page_size = req_manager.cache_config.compressed_kv_token_alignment // compress_ratio
        from lightllm.models.deepseek_v4.triton_kernel.cache_io import get_compressed_kv_buffer

        out_buffer = get_compressed_kv_buffer(mem_manager, layer_idx)

    if kv_score.shape[0] == 0:
        return out_buffer

    # 三种 compressor 共用算子，分别对应 req_to_c4_state / req_to_c4_indexer_state / req_to_c128_state。
    state_capacity_tokens = req_to_compressor_state.shape[1]
    req_to_compressor_state = req_to_compressor_state.flatten(0, 1)
    candidate_dim = kv_score.shape[-1] // 2
    state_dim = req_to_compressor_state.shape[-1]
    is_c4 = compress_ratio == 4
    block_state = triton.next_power_of_2(candidate_dim)
    block_head = triton.next_power_of_2(head_dim)

    _fused_compress_norm_rope_kernel[(kv_score.shape[0],)](
        kv_score,
        kv_score.stride(0),
        kv_score.stride(1),
        req_to_compressor_state,
        infer_state.position_ids,
        infer_state._dsv4_token_to_batch_idx,
        infer_state.b_req_idx,
        infer_state.b_mtp_index,
        infer_state.b_seq_len,
        infer_state.b_ready_cache_len,
        infer_state.b_q_start_loc,
        compressed_kv_slots,
        norm_weight,
        eps,
        cos_table,
        cos_table.stride(0),
        cos_table.stride(1),
        sin_table,
        sin_table.stride(0),
        sin_table.stride(1),
        out_buffer,
        HEAD_DIM=head_dim,
        CANDIDATE_DIM=candidate_dim,
        STATE_DIM=state_dim,
        COMPRESS_RATIO=compress_ratio,
        WINDOW_SIZE=compress_ratio * (2 if is_c4 else 1),
        IS_C4=is_c4,
        IS_PREFILL=infer_state.is_prefill,
        STATE_CAPACITY_TOKENS=state_capacity_tokens,
        ROPE_HEAD_DIM=qk_rope_head_dim,
        FP8_MAX=torch.finfo(torch.float8_e4m3fn).max,
        AMAX_MIN=1e-4,
        NOPE_DIM=head_dim - qk_rope_head_dim,
        QUANT_BLOCK=64,
        SCALE_BYTES=(head_dim - qk_rope_head_dim) // 64 + 1,
        PAGE_SIZE=out_page_size,
        BYTES_PER_PAGE=out_buffer.stride(0),
        BLOCK=block_head,
        IS_IN_INDEXER=is_in_indexer,
        num_warps=4,
    )

    _save_compressor_state_kernel[(kv_score.shape[0],)](
        kv_score,
        kv_score.stride(0),
        kv_score.stride(1),
        infer_state.position_ids,
        infer_state._dsv4_token_to_batch_idx,
        infer_state.b_req_idx,
        infer_state.b_seq_len,
        req_to_compressor_state,
        CANDIDATE_DIM=candidate_dim,
        STATE_DIM=state_dim,
        IS_PREFILL=infer_state.is_prefill,
        STATE_CAPACITY_TOKENS=state_capacity_tokens,
        BLOCK=block_state,
        num_warps=4,
    )
    return out_buffer
