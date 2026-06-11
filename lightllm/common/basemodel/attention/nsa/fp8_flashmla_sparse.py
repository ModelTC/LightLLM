import dataclasses
import inspect
import torch
from typing import TYPE_CHECKING, Tuple

from ..base_att import AttControl, BaseAttBackend, BaseDecodeAttState, BasePrefillAttState
from lightllm.utils.dist_utils import get_current_device_id

if TYPE_CHECKING:
    from lightllm.common.basemodel.infer_struct import InferStateInfo


FLASHMLA_INDEX_ALIGN = 64
# this flash_mla extra-cache fork only instantiates h_q in {64, 128}; pad TP-split q heads up
# to the nearest supported count (zero heads are discarded from the output slice).
FLASHMLA_SUPPORTED_HEADS = (64, 128)


def _pad_q_heads(q_4d: torch.Tensor, attn_sink: torch.Tensor):
    h_q = q_4d.shape[2]
    if h_q in FLASHMLA_SUPPORTED_HEADS:
        return q_4d, attn_sink, h_q
    target = next((h for h in FLASHMLA_SUPPORTED_HEADS if h >= h_q), None)
    assert target is not None, f"num q heads {h_q} exceeds flash_mla support {FLASHMLA_SUPPORTED_HEADS}"
    q_pad = torch.nn.functional.pad(q_4d, (0, 0, 0, target - h_q))
    sink_pad = torch.nn.functional.pad(attn_sink, (0, target - h_q))
    return q_pad, sink_pad, h_q


class DeepseekV4MissingOperatorError(RuntimeError):
    pass


def _missing_attention_op(feature: str) -> None:
    raise DeepseekV4MissingOperatorError(
        f"DeepSeek-V4 {feature} has no production batch operator. The flashmla_kvcache path "
        f"(packed swa/c4/c128 pools + paged compressor + indexer top-k) is the supported route; "
        f"this legacy/non-flashmla entry point was never wired and is fenced on purpose."
    )


def _pad_last_dim(x: torch.Tensor, multiple: int = FLASHMLA_INDEX_ALIGN, value: int = -1) -> torch.Tensor:
    pad = (-x.shape[-1]) % multiple
    if pad == 0:
        return x.contiguous()
    out = torch.full((*x.shape[:-1], x.shape[-1] + pad), value, dtype=x.dtype, device=x.device)
    out[..., : x.shape[-1]] = x
    return out.contiguous()


def _view_dsv4_flashmla_cache(layer_buffer: torch.Tensor, page_size: int) -> torch.Tensor:
    from lightllm.common.kv_cache_mem_manager.deepseek4_mem_manager import DSV4_MLA_BYTES_PER_TOKEN

    usable = page_size * DSV4_MLA_BYTES_PER_TOKEN
    return layer_buffer[:, :usable].view(layer_buffer.shape[0], page_size, 1, DSV4_MLA_BYTES_PER_TOKEN)


def _load_flash_mla_with_extra():
    try:
        import flash_mla
    except Exception as exc:
        raise DeepseekV4MissingOperatorError(
            "DeepSeek-V4 packed FlashMLA requires the flash_mla package with compiled CUDA extension. "
            f"Import failed with: {type(exc).__name__}: {exc}"
        ) from exc

    fn = getattr(flash_mla, "flash_mla_with_kvcache", None)
    get_mla_metadata = getattr(flash_mla, "get_mla_metadata", None)
    missing_symbols = []
    if fn is None:
        missing_symbols.append("flash_mla_with_kvcache")
    if get_mla_metadata is None:
        missing_symbols.append("get_mla_metadata")
    if missing_symbols:
        raise DeepseekV4MissingOperatorError(
            "DeepSeek-V4 requires flash_mla.flash_mla_with_kvcache extra-cache wrapper. "
            f"Current module={getattr(flash_mla, '__file__', '<unknown>')} "
            f"is missing symbols {missing_symbols}."
        )

    sig = inspect.signature(fn)
    required = {
        "attn_sink",
        "extra_k_cache",
        "extra_indices_in_kvcache",
        "topk_length",
        "extra_topk_length",
    }
    missing = sorted(required.difference(sig.parameters))
    if missing:
        raise DeepseekV4MissingOperatorError(
            "DeepSeek-V4 requires flash_mla.flash_mla_with_kvcache with extra-cache arguments. "
            f"Current module={getattr(flash_mla, '__file__', '<unknown>')} is missing {missing}."
        )
    return flash_mla


def _build_dsv4_repeated_prefill_reqs(infer_state) -> torch.Tensor:
    return torch.repeat_interleave(infer_state.b_req_idx, infer_state.b_q_seq_len.long())


def _build_dsv4_prefill_positions(infer_state) -> torch.Tensor:
    total = infer_state.total_token_num - infer_state.prefix_total_token_num
    token_offsets = torch.arange(total, dtype=torch.int32, device=infer_state.b_q_seq_len.device)
    req_ids = torch.repeat_interleave(
        torch.arange(infer_state.batch_size, dtype=torch.long, device=infer_state.b_q_seq_len.device),
        infer_state.b_q_seq_len.long(),
    )
    local_offsets = token_offsets - infer_state.b_q_start_loc[req_ids]
    return infer_state.b_ready_cache_len[req_ids] + local_offsets


def _build_dsv4_swa_indices(
    req_manager,
    mem_manager,
    req_idx: torch.Tensor,
    positions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    window = int(mem_manager.sliding_window)
    offsets = positions[:, None] - torch.arange(window, dtype=positions.dtype, device=positions.device)[None, :]
    valid_pos = offsets >= 0
    safe_offsets = offsets.clamp_min(0).long()
    full_slots = req_manager.req_to_token_indexs[req_idx.long()[:, None], safe_offsets]
    swa_slots = mem_manager.full_to_swa_indexs[full_slots.long()].to(torch.int32)
    indices = torch.where(valid_pos, swa_slots, torch.full_like(swa_slots, -1))
    lengths = torch.clamp(positions + 1, min=1, max=window).to(torch.int32)
    return _pad_last_dim(indices.to(torch.int32)).unsqueeze(1), lengths.contiguous()


def _gather_dsv4_compress_slots(
    infer_state,
    mapping: torch.Tensor,
    req_idx: torch.Tensor,
    valid: torch.Tensor,
    offsets: torch.Tensor,
    ratio: int,
) -> torch.Tensor:
    """条目 g 的压缩槽 = full_to_c*[req_to_token[req, (g+1)*ratio-1]](组末 token 的 full 槽位)。
    无效条目(超出因果长度/HOLD 行)用位置 0 安全 gather 后由调用方按 valid 掩掉。"""
    end_pos = offsets[None, :] * ratio + (ratio - 1)
    safe_pos = torch.where(valid, end_pos, torch.zeros_like(end_pos))
    full_slots = infer_state.req_manager.req_to_token_indexs[req_idx.long()[:, None], safe_pos]
    return mapping[full_slots.long()].to(torch.int32)


def _build_dsv4_c128_indices(
    infer_state,
    req_idx: torch.Tensor,
    positions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    raw_lengths = (positions + 1) // 128
    lengths = torch.clamp(raw_lengths, min=1).to(torch.int32)
    max_len = max(1, int(infer_state.max_kv_seq_len) // 128)
    offsets = torch.arange(max_len, dtype=torch.long, device=positions.device)
    valid = offsets[None, :] < raw_lengths[:, None]
    slots = _gather_dsv4_compress_slots(
        infer_state, infer_state.mem_manager.full_to_c128_indexs, req_idx, valid, offsets, 128
    )
    indices = torch.where(valid, slots, torch.full_like(slots, -1))
    return _pad_last_dim(indices).unsqueeze(1), lengths.contiguous()


def _build_dsv4_c4_indices(
    infer_state,
    layer_index: int,
    req_idx: torch.Tensor,
    positions: torch.Tensor,
    nsa_dict: dict,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """c4(CSA) extra indices: causal all-entries when the entry space fits index_topk,
    otherwise Lightning-Indexer scored top-k. Pure tensor ops (decode runs inside cuda graphs)."""
    import torch.distributed as dist
    import torch.nn.functional as F
    from lightllm.distributed.communication_op import all_reduce

    mem_manager = infer_state.mem_manager
    raw_lengths = (positions + 1) // 4
    max_entries = max(1, int(infer_state.max_kv_seq_len) // 4)
    index_topk = int(nsa_dict["index_topk"])
    offsets = torch.arange(max_entries, dtype=torch.long, device=positions.device)
    valid = offsets[None, :] < raw_lengths[:, None]
    slots = _gather_dsv4_compress_slots(infer_state, mem_manager.full_to_c4_indexs, req_idx, valid, offsets, 4)

    if max_entries <= index_topk:
        lengths = torch.clamp(raw_lengths, min=1).to(torch.int32)
        indices = torch.where(valid, slots, torch.full_like(slots, -1))
        return _pad_last_dim(indices).unsqueeze(1), lengths.contiguous()

    idx_q = nsa_dict["idx_q"]  # [T, H, index_head_dim], rope applied
    idx_weight = nsa_dict["idx_weight"]  # [T, H] fp32, weight scale applied
    score_scale = float(nsa_dict["indexer_score_scale"])
    hold_slot = mem_manager.c4_indexer_pool.HOLD_TOKEN_MEMINDEX
    safe_slots = torch.where(valid, slots.long(), torch.full_like(slots.long(), hold_slot))
    k = mem_manager.gather_indexer_k(layer_index, safe_slots.reshape(-1)).view(positions.shape[0], max_entries, -1)

    num_tokens, num_heads = idx_q.shape[0], idx_q.shape[1]
    score_chunks = []
    chunk = max(1, min(num_tokens, (16 * 1024 * 1024) // max(1, num_heads * max_entries)))
    for start in range(0, num_tokens, chunk):
        end = min(num_tokens, start + chunk)
        scores = torch.einsum("thd,tnd->thn", idx_q[start:end].float(), k[start:end].float())
        scores = F.relu(scores) * score_scale
        score_chunks.append((scores * idx_weight[start:end].unsqueeze(-1)).sum(dim=1))
    index_scores = torch.cat(score_chunks, dim=0)
    if int(nsa_dict.get("tp_world_size", 1)) > 1:
        all_reduce(index_scores, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
    index_scores = index_scores.masked_fill(~valid, float("-inf"))
    top = index_scores.topk(index_topk, dim=-1).indices
    top_valid = torch.gather(valid, 1, top)
    top_slots = torch.gather(slots.long(), 1, top).to(torch.int32)
    indices = torch.where(top_valid, top_slots, torch.full_like(top_slots, -1))
    lengths = torch.clamp(torch.minimum(raw_lengths, torch.full_like(raw_lengths, index_topk)), min=1)
    return _pad_last_dim(indices).unsqueeze(1), lengths.to(torch.int32).contiguous()


def _build_dsv4_extra_metadata(
    infer_state,
    layer_index: int,
    compress_ratio: int,
    req_idx: torch.Tensor,
    positions: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lengths: torch.Tensor,
    nsa_dict: dict,
) -> "_Dsv4Metadata":
    from lightllm.common.kv_cache_mem_manager.deepseek4_mem_manager import DSV4_C128_PAGE_SIZE, DSV4_C4_PAGE_SIZE

    if compress_ratio == 0:
        return _Dsv4Metadata(swa_indices, swa_lengths)
    if compress_ratio == 4:
        extra_indices, extra_lengths = _build_dsv4_c4_indices(infer_state, layer_index, req_idx, positions, nsa_dict)
        extra_buffer = infer_state.mem_manager.get_compressed_kv_buffer(layer_index)
        extra_cache = _view_dsv4_flashmla_cache(extra_buffer, DSV4_C4_PAGE_SIZE)
        return _Dsv4Metadata(swa_indices, swa_lengths, extra_cache, extra_indices, extra_lengths)
    if compress_ratio == 128:
        extra_indices, extra_lengths = _build_dsv4_c128_indices(infer_state, req_idx, positions)
        extra_buffer = infer_state.mem_manager.get_compressed_kv_buffer(layer_index)
        extra_cache = _view_dsv4_flashmla_cache(extra_buffer, DSV4_C128_PAGE_SIZE)
        return _Dsv4Metadata(swa_indices, swa_lengths, extra_cache, extra_indices, extra_lengths)
    raise AssertionError(f"invalid DeepSeek-V4 compress ratio {compress_ratio}")


@dataclasses.dataclass
class _Dsv4Metadata:
    swa_indices: torch.Tensor
    swa_lengths: torch.Tensor
    extra_cache: torch.Tensor = None
    extra_indices: torch.Tensor = None
    extra_lengths: torch.Tensor = None


class NsaFlashMlaFp8SparseAttBackend(BaseAttBackend):
    def __init__(self, model):
        super().__init__(model=model)
        device = get_current_device_id()
        self.ragged_mem_buffers = [
            torch.empty(model.graph_max_batch_size * model.max_seq_length, dtype=torch.int32, device=device)
            for _ in range(2)
        ]
        self._flash_mla = None

    def create_att_prefill_state(self, infer_state: "InferStateInfo") -> "NsaFlashMlaFp8SparsePrefillAttState":
        return NsaFlashMlaFp8SparsePrefillAttState(backend=self, infer_state=infer_state)

    def create_att_decode_state(self, infer_state: "InferStateInfo") -> "NsaFlashMlaFp8SparseDecodeAttState":
        return NsaFlashMlaFp8SparseDecodeAttState(backend=self, infer_state=infer_state)

    def flash_mla(self):
        if self._flash_mla is None:
            self._flash_mla = _load_flash_mla_with_extra()
        return self._flash_mla


@dataclasses.dataclass
class NsaFlashMlaFp8SparsePrefillAttState(BasePrefillAttState):
    ks: torch.Tensor = None
    ke: torch.Tensor = None
    lengths: torch.Tensor = None
    ragged_mem_index: torch.Tensor = None

    def init_state(self):
        self.backend: NsaFlashMlaFp8SparseAttBackend = self.backend
        self.ragged_mem_index = torch.empty(
            self.infer_state.total_token_num,
            dtype=torch.int32,
            device=get_current_device_id(),
        )
        from lightllm.common.basemodel.triton_kernel.gen_nsa_ks_ke import gen_nsa_ks_ke

        self.ks, self.ke, self.lengths = gen_nsa_ks_ke(
            b_seq_len=self.infer_state.b_seq_len,
            b_q_seq_len=self.infer_state.b_q_seq_len,
            b_req_idx=self.infer_state.b_req_idx,
            req_to_token_index=self.infer_state.req_manager.req_to_token_indexs,
            q_token_num=self.infer_state.total_token_num - self.infer_state.prefix_total_token_num,
            ragged_mem_index=self.ragged_mem_index,
            hold_req_idx=self.infer_state.req_manager.HOLD_REQUEST_ID,
        )
        return

    def prefill_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        assert att_control.nsa_prefill, "nsa_prefill must be True for NSA prefill attention"
        assert att_control.nsa_prefill_dict is not None, "nsa_prefill_dict is required"
        if att_control.nsa_prefill_dict.get("flashmla_kvcache"):
            return self._flashmla_kvcache_prefill_att(
                q=q,
                packed_kv=k,
                nsa_dict=att_control.nsa_prefill_dict,
            )
        return self._nsa_prefill_att(q=q, packed_kv=k, att_control=att_control)

    def _nsa_prefill_att(
        self,
        q: torch.Tensor,
        packed_kv: torch.Tensor,
        att_control: AttControl,
    ) -> torch.Tensor:
        import flash_mla

        nsa_dict = att_control.nsa_prefill_dict
        topk_indices = nsa_dict["topk_indices"]
        softmax_scale = nsa_dict["softmax_scale"]
        kv_lora_rank = nsa_dict["kv_lora_rank"]
        topk_mem_indices = nsa_dict["topk_mem_indices"]
        prefill_cache_kv = nsa_dict["prefill_cache_kv"]
        attn_sink = nsa_dict.get("attn_sink")
        topk_length = nsa_dict.get("topk_length")

        if self.infer_state.prefix_total_token_num > 0:
            # 当前推理生成的token kv部分从 prefill_cache_kv 中获取，历史
            # 部分kv 从 packed_kv 中获取, 并进行反量化，这样可以避免 prefill_cache_kv
            # 部分的数据进行重复的反量化操作，提升整体的性能。
            kv, topk_indices = self.infer_state.mem_manager.get_prefill_kv_cache_and_remap_indices(
                packed_kv=packed_kv,
                topk_indices=topk_mem_indices,
                prefill_mem_index=self.infer_state.mem_index,
                prefill_cache_kv=prefill_cache_kv,
            )
        else:
            kv = prefill_cache_kv

        if topk_indices.ndim == 2:
            topk_indices = topk_indices.unsqueeze(1)

        mla_out, _, _ = flash_mla.flash_mla_sparse_fwd(
            q=q,
            kv=kv,
            indices=topk_indices,
            sm_scale=softmax_scale,
            d_v=kv_lora_rank,
            attn_sink=attn_sink,
            topk_length=topk_length,
        )
        return mla_out

    def _build_flashmla_kvcache_prefill_metadata(self, nsa_dict: dict) -> _Dsv4Metadata:
        infer_state = self.infer_state
        req_idx = _build_dsv4_repeated_prefill_reqs(infer_state)
        positions = _build_dsv4_prefill_positions(infer_state)
        swa_indices, swa_lengths = _build_dsv4_swa_indices(
            infer_state.req_manager,
            infer_state.mem_manager,
            req_idx,
            positions,
        )
        return _build_dsv4_extra_metadata(
            infer_state,
            nsa_dict["layer_index"],
            nsa_dict["compress_ratio"],
            req_idx,
            positions,
            swa_indices,
            swa_lengths,
            nsa_dict,
        )

    def _flashmla_kvcache_prefill_att(self, q: torch.Tensor, packed_kv: torch.Tensor, nsa_dict: dict) -> torch.Tensor:
        attn_sink = nsa_dict["attn_sink"].to(torch.float32).contiguous()
        metadata = self._build_flashmla_kvcache_prefill_metadata(nsa_dict)
        return self._flashmla_kvcache_att(q, packed_kv, metadata, attn_sink, nsa_dict)

    def _flashmla_kvcache_att(
        self,
        q: torch.Tensor,
        packed_kv: torch.Tensor,
        metadata: _Dsv4Metadata,
        attn_sink: torch.Tensor,
        nsa_dict: dict,
    ) -> torch.Tensor:
        flash_mla = self.backend.flash_mla()
        from lightllm.common.kv_cache_mem_manager.deepseek4_mem_manager import DSV4_SWA_PAGE_SIZE

        q_4d = q.unsqueeze(1).contiguous()
        q_4d, attn_sink, num_real_heads = _pad_q_heads(q_4d, attn_sink)
        k_cache = _view_dsv4_flashmla_cache(packed_kv, DSV4_SWA_PAGE_SIZE)
        sched_meta, _ = flash_mla.get_mla_metadata()
        out, _ = flash_mla.flash_mla_with_kvcache(
            q=q_4d,
            k_cache=k_cache,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=nsa_dict["head_dim_v"],
            tile_scheduler_metadata=sched_meta,
            num_splits=None,
            softmax_scale=nsa_dict["softmax_scale"],
            causal=False,
            is_fp8_kvcache=True,
            indices=metadata.swa_indices,
            attn_sink=attn_sink,
            topk_length=metadata.swa_lengths,
            extra_k_cache=metadata.extra_cache,
            extra_indices_in_kvcache=metadata.extra_indices,
            extra_topk_length=metadata.extra_lengths,
        )
        return out[:, 0, :num_real_heads].contiguous()


@dataclasses.dataclass
class NsaFlashMlaFp8SparseDecodeAttState(BaseDecodeAttState):
    ks: torch.Tensor = None
    ke: torch.Tensor = None
    lengths: torch.Tensor = None
    ragged_mem_index: torch.Tensor = None
    flashmla_sched_meta: object = None

    def init_state(self):
        self.backend: NsaFlashMlaFp8SparseAttBackend = self.backend
        model = self.backend.model
        use_cuda_graph = (
            self.infer_state.batch_size <= model.graph_max_batch_size
            and self.infer_state.max_kv_seq_len <= model.graph_max_len_in_batch
        )

        if use_cuda_graph:
            self.ragged_mem_index = self.backend.ragged_mem_buffers[self.infer_state.microbatch_index]
        else:
            self.ragged_mem_index = torch.empty(
                self.infer_state.total_token_num,
                dtype=torch.int32,
                device=get_current_device_id(),
            )

        from lightllm.common.basemodel.triton_kernel.gen_nsa_ks_ke import gen_nsa_ks_ke

        self.ks, self.ke, self.lengths = gen_nsa_ks_ke(
            b_seq_len=self.infer_state.b_seq_len,
            b_q_seq_len=self.infer_state.b_q_seq_len,
            b_req_idx=self.infer_state.b_req_idx,
            req_to_token_index=self.infer_state.req_manager.req_to_token_indexs,
            q_token_num=self.infer_state.b_seq_len.shape[0],
            ragged_mem_index=self.ragged_mem_index,
            hold_req_idx=self.infer_state.req_manager.HOLD_REQUEST_ID,
        )
        flash_mla = self.backend.flash_mla()
        # one sched_meta per layer type: the lazy config locks extra-cache geometry (page size,
        # presence) on first invocation, so swa-only/c4/c128 layers must not share one object.
        self.flashmla_sched_meta = {ratio: flash_mla.get_mla_metadata()[0] for ratio in (0, 4, 128)}
        return

    def reset_sched_meta_for_capture(self):
        # cuda-graph capture hook: the warmup pass already locked/stored sched meta on this
        # (shared) state object; reset so the capture pass re-plans INSIDE the graph and every
        # replay re-plans from the live tensors instead of binding warmup leftovers.
        flash_mla = self.backend.flash_mla()
        self.flashmla_sched_meta = {ratio: flash_mla.get_mla_metadata()[0] for ratio in (0, 4, 128)}
        return

    def decode_att(
        self,
        q: Tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        assert att_control.nsa_decode, "nsa_decode must be True for NSA decode attention"
        assert att_control.nsa_decode_dict is not None, "nsa_decode_dict is required"
        if att_control.nsa_decode_dict.get("flashmla_kvcache"):
            return self._flashmla_kvcache_decode_att(
                q=q,
                packed_kv=k,
                nsa_dict=att_control.nsa_decode_dict,
            )
        return self._nsa_decode_att(q=q, packed_kv=k, att_control=att_control)

    def _nsa_decode_att(
        self,
        q: Tuple[torch.Tensor, torch.Tensor],
        packed_kv: torch.Tensor,
        att_control: AttControl,
    ) -> torch.Tensor:
        import flash_mla

        nsa_dict = att_control.nsa_decode_dict
        topk_mem_indices = nsa_dict["topk_mem_indices"]
        softmax_scale = nsa_dict["softmax_scale"]
        kv_lora_rank = nsa_dict["kv_lora_rank"]
        attn_sink = nsa_dict.get("attn_sink")
        topk_length = nsa_dict.get("topk_length")
        extra_k_cache = nsa_dict.get("extra_k_cache")
        extra_indices = nsa_dict.get("extra_indices_in_kvcache")
        extra_topk_length = nsa_dict.get("extra_topk_length")

        if topk_mem_indices.ndim == 2:
            topk_mem_indices = topk_mem_indices.unsqueeze(1)
        assert topk_mem_indices.shape[1] == 1, "FlashMLA sparse decode path currently expects seq_len_q == 1"

        q_nope, q_rope = q
        q_all = torch.cat([q_nope, q_rope], dim=-1).unsqueeze(1).contiguous()
        kv = torch.as_strided(
            packed_kv,
            size=(packed_kv.shape[0], 1, 1, packed_kv.shape[-1]),
            stride=(packed_kv.stride(0), packed_kv.shape[-1], packed_kv.shape[-1], packed_kv.stride(-1)),
        )

        o_tensor, _ = flash_mla.flash_mla_with_kvcache(
            q=q_all,
            k_cache=kv,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=kv_lora_rank,
            tile_scheduler_metadata=self.flashmla_sched_meta[0],
            softmax_scale=softmax_scale,
            causal=False,
            is_fp8_kvcache=True,
            indices=topk_mem_indices,
            attn_sink=attn_sink,
            topk_length=topk_length,
            extra_k_cache=extra_k_cache,
            extra_indices_in_kvcache=extra_indices,
            extra_topk_length=extra_topk_length,
        )
        return o_tensor[:, 0, :, :]  # [b, 1, h, d] -> [b, h, d]

    def _build_flashmla_kvcache_decode_metadata(self, nsa_dict: dict) -> _Dsv4Metadata:
        infer_state = self.infer_state
        positions = infer_state.b_seq_len.to(torch.int32) - 1
        swa_indices, swa_lengths = _build_dsv4_swa_indices(
            infer_state.req_manager,
            infer_state.mem_manager,
            infer_state.b_req_idx,
            positions,
        )
        return _build_dsv4_extra_metadata(
            infer_state,
            nsa_dict["layer_index"],
            nsa_dict["compress_ratio"],
            infer_state.b_req_idx,
            positions,
            swa_indices,
            swa_lengths,
            nsa_dict,
        )

    def _flashmla_kvcache_decode_att(self, q: torch.Tensor, packed_kv: torch.Tensor, nsa_dict: dict) -> torch.Tensor:
        attn_sink = nsa_dict["attn_sink"].to(torch.float32).contiguous()
        metadata = self._build_flashmla_kvcache_decode_metadata(nsa_dict)
        return self._flashmla_kvcache_att(q, packed_kv, metadata, attn_sink, nsa_dict)

    def _flashmla_kvcache_att(
        self,
        q: torch.Tensor,
        packed_kv: torch.Tensor,
        metadata: _Dsv4Metadata,
        attn_sink: torch.Tensor,
        nsa_dict: dict,
    ) -> torch.Tensor:
        flash_mla = self.backend.flash_mla()
        from lightllm.common.kv_cache_mem_manager.deepseek4_mem_manager import DSV4_SWA_PAGE_SIZE

        q_4d = q.unsqueeze(1).contiguous()
        q_4d, attn_sink, num_real_heads = _pad_q_heads(q_4d, attn_sink)
        k_cache = _view_dsv4_flashmla_cache(packed_kv, DSV4_SWA_PAGE_SIZE)
        out, _ = flash_mla.flash_mla_with_kvcache(
            q=q_4d,
            k_cache=k_cache,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=nsa_dict["head_dim_v"],
            tile_scheduler_metadata=self.flashmla_sched_meta[nsa_dict["compress_ratio"]],
            num_splits=None,
            softmax_scale=nsa_dict["softmax_scale"],
            causal=False,
            is_fp8_kvcache=True,
            indices=metadata.swa_indices,
            attn_sink=attn_sink,
            topk_length=metadata.swa_lengths,
            extra_k_cache=metadata.extra_cache,
            extra_indices_in_kvcache=metadata.extra_indices,
            extra_topk_length=metadata.extra_lengths,
        )
        return out[:, 0, :num_real_heads].contiguous()
