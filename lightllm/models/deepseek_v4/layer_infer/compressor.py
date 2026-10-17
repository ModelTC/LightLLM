import torch
import triton

from lightllm.common.basemodel import BaseLayerInfer
from ..infer_struct import DeepseekV4InferStateInfo
from ..layer_weights.transformer_layer_weight import DeepseekV4TransformerLayerWeight
from ..triton_kernel.compressor import apply_ape, _fused_compress_norm_rope_insert_kernel, _save_partial_states_kernel


class CompressorInfer(BaseLayerInfer):
    """Compress latent KV into packed history, or emit BF16 C4 indexer keys.

    Both paths share the compression kernel and keep unfinished groups in request state.
    """

    def __init__(self, layer_idx: int, network_config: dict, tp_world_size: int, is_in_indexer: bool = False):
        super().__init__()
        self.layer_idx_ = layer_idx
        self.network_config_ = network_config
        self.tp_world_size_ = tp_world_size
        self.is_in_indexer = is_in_indexer
        self.compress_ratio = network_config["compress_ratios"][layer_idx]
        self.head_dim = network_config["head_dim"]
        self.index_head_dim = network_config["index_head_dim"]
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
            head_dim = self.index_head_dim
        else:
            projection = layer_weight.compressor_wkv_gate_
            norm_weight = layer_weight.compressor_norm_.weight
            ape = layer_weight.compressor_ape_.weight
            head_dim = self.head_dim
        if kv_score_out is None:
            shape = (x.shape[0], head_dim * (4 if self.compress_ratio == 4 else 2))
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
                (infer_state.mem_index.numel(), self.index_head_dim),
                torch.bfloat16,
                device=infer_state.mem_index.device,
            )
        return fused_compress(
            kv_score=kv_score,
            infer_state=infer_state,
            layer_idx=self.layer_idx_,
            norm_weight=norm_weight,
            eps=self.eps,
            head_dim=head_dim,
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
    requests = infer_state.req_manager
    out_slots = torch.where(
        (infer_state.position_ids + 1) % compress_ratio == 0,
        infer_state.mem_index.reshape(-1) // compress_ratio,
        -1,
    )
    if is_in_indexer:
        assert compress_ratio == 4, "只有 c4(CSA) 层有 indexer-K"

        state_buffer = requests.req_to_c4_indexer_state[requests.cache_config.layer_to_c4_idx[layer_idx]]
        checkpoint_buffer = (
            requests.prefill_indexer_checkpoints[requests.cache_config.layer_to_c4_idx[layer_idx]]
            if requests.prefill_indexer_checkpoints is not None
            else None
        )
        if out_buffer is None:
            out_buffer = torch.empty(
                (infer_state.mem_index.numel(), requests.cache_config.indexer_head_dim),
                dtype=torch.bfloat16,
                device=infer_state.mem_index.device,
            )
        out_page_size = 1
    else:
        if compress_ratio == 4:
            state_buffer = requests.req_to_c4_state[requests.cache_config.layer_to_c4_idx[layer_idx]]
            checkpoint_buffer = (
                requests.prefill_c4_checkpoints[requests.cache_config.layer_to_c4_idx[layer_idx]]
                if requests.prefill_c4_checkpoints is not None
                else None
            )
            out_page_size = requests.cache_config.history_token_alignment // compress_ratio
        elif compress_ratio == 128:
            state_buffer = requests.req_to_c128_state[requests.cache_config.layer_to_c128_idx[layer_idx]]
            checkpoint_buffer = None
            out_page_size = requests.cache_config.history_token_alignment // compress_ratio
        from lightllm.models.deepseek_v4.triton_kernel.cache_io import get_compressed_kv_buffer

        out_buffer = get_compressed_kv_buffer(mem_manager, layer_idx)

    if kv_score.shape[0] == 0:
        return out_buffer

    state_tokens = state_buffer.shape[1]
    state_buffer = state_buffer.flatten(0, 1)
    checkpoint_buffer = state_buffer if checkpoint_buffer is None else checkpoint_buffer.flatten(0, 1)
    candidate_dim = kv_score.shape[-1] // 2
    state_dim = state_buffer.shape[-1]
    is_c4 = compress_ratio == 4
    block_state = triton.next_power_of_2(candidate_dim)
    block_head = triton.next_power_of_2(head_dim)
    token_to_batch_idx = infer_state._dsv4_token_to_batch_idx if infer_state.is_prefill else infer_state.b_req_idx
    ready_cache_len = (
        infer_state.b_ready_cache_len if infer_state.b_ready_cache_len is not None else infer_state.b_seq_len
    )
    q_start_loc = infer_state.b_q_start_loc if infer_state.b_q_start_loc is not None else infer_state.b_seq_len

    _fused_compress_norm_rope_insert_kernel[(kv_score.shape[0],)](
        kv_score,
        kv_score.stride(0),
        kv_score.stride(1),
        state_buffer,
        infer_state.position_ids,
        token_to_batch_idx,
        infer_state.b_req_idx,
        infer_state.b_mtp_index,
        infer_state.b_seq_len,
        ready_cache_len,
        q_start_loc,
        checkpoint_buffer,
        requests.prefill_checkpoint_offsets,
        out_slots,
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
        STATE_TOKENS=state_tokens,
        CHECKPOINT_TOKENS=requests.cache_config.history_token_alignment,
        HOLD_REQ=requests.HOLD_REQUEST_ID,
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

    _save_partial_states_kernel[(kv_score.shape[0],)](
        kv_score,
        kv_score.stride(0),
        kv_score.stride(1),
        infer_state.position_ids,
        token_to_batch_idx,
        infer_state.b_req_idx,
        infer_state.b_seq_len,
        state_buffer,
        CANDIDATE_DIM=candidate_dim,
        STATE_DIM=state_dim,
        IS_PREFILL=infer_state.is_prefill,
        STATE_TOKENS=state_tokens,
        BLOCK=block_state,
        num_warps=4,
    )
    return out_buffer
