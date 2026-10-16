"""PyTorch SDPA attention for Apple Metal using LightLLM's paged KV layout."""

import dataclasses

import torch
import torch.nn.functional as F
from .scatter import gather_kv

from lightllm.common.basemodel.attention.base_att import (
    AttControl,
    BaseAttBackend,
    BaseDecodeAttState,
    BasePrefillAttState,
)


class MetalAttBackend(BaseAttBackend):
    """Portable attention backend for ordinary dense KV caches on Metal."""

    def create_att_prefill_state(self, infer_state) -> "MetalPrefillAttState":
        return MetalPrefillAttState(backend=self, infer_state=infer_state)

    def create_att_decode_state(self, infer_state) -> "MetalDecodeAttState":
        return MetalDecodeAttState(backend=self, infer_state=infer_state)


def _supports_paged_attention(att_control):
    return not any(
        (
            att_control.use_alibi,
            att_control.use_sliding_window,
            att_control.use_att_sink,
            att_control.mla_prefill,
            att_control.mla_decode,
            att_control.nsa_prefill,
            att_control.nsa_decode,
            att_control.linear_att_prefill,
            att_control.linear_att_decode,
        )
    )


def _attend_decode_group(q, k, v, seq_len: int, att_control: AttControl):
    """Batched one-token attention for requests sharing a KV length."""
    return _attend_group(q.unsqueeze(1), k, v, query_start=seq_len - 1, att_control=att_control).squeeze(1)


def _attend_group(q, k, v, query_start: int, att_control: AttControl):
    """Batched attention for requests with matching query/key lengths and positions."""
    if (
        att_control.mla_prefill
        or att_control.mla_decode
        or att_control.nsa_prefill
        or att_control.nsa_decode
        or att_control.linear_att_prefill
        or att_control.linear_att_decode
    ):
        raise NotImplementedError("Metal dense attention cannot handle MLA, NSA, or linear-attention state.")
    if att_control.use_att_sink:
        raise NotImplementedError("Metal attention does not support attention sinks yet.")

    if not att_control.use_alibi and not att_control.use_sliding_window:
        from .prefill_attention import prefill

        result = prefill(q, k, v, query_start)
        if result is not None:
            return result

    output_dtype = q.dtype
    query = q.transpose(1, 2)
    key = k.transpose(1, 2)
    value = v.transpose(1, 2)
    if att_control.use_alibi:
        # Match LightLLM's FP32 accumulation for QK, positional bias, softmax,
        # and weighted V instead of rounding the mask to the model dtype.
        query = query.float()
        key = key.float()
        value = value.float()

    q_heads, kv_heads = query.shape[1], key.shape[1]
    if q_heads % kv_heads:
        raise ValueError(f"Metal GQA requires query heads ({q_heads}) to be divisible by KV heads ({kv_heads}).")
    if q_heads != kv_heads:
        groups = q_heads // kv_heads
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)

    query_len, key_len = query.shape[2], key.shape[2]
    if query_start == 0 and query_len == key_len and not att_control.use_alibi and not att_control.use_sliding_window:
        result = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        return result.transpose(1, 2).to(dtype=output_dtype)
    if query_len == 1 and not att_control.use_alibi and not att_control.use_sliding_window:
        result = F.scaled_dot_product_attention(query, key, value)
        return result.transpose(1, 2).to(dtype=output_dtype)

    query_positions = torch.arange(query_start, query_start + query_len, device=q.device)
    key_positions = torch.arange(key_len, device=q.device)
    allowed = key_positions.view(1, 1, 1, -1) <= query_positions.view(1, 1, -1, 1)
    if att_control.use_sliding_window:
        left, right = att_control.sliding_window
        if right != 0:
            raise NotImplementedError("Metal attention only supports left sliding windows with right=0.")
        if left >= 0:
            allowed = allowed & (key_positions.view(1, 1, 1, -1) >= query_positions.view(1, 1, -1, 1) - left)

    attn_mask = allowed
    if att_control.use_alibi:
        if att_control.tp_alibi is None:
            raise ValueError("ALiBi attention requires per-head slopes.")
        slopes = att_control.tp_alibi.to(device=q.device, dtype=torch.float32).reshape(1, q_heads, 1, 1)
        relative_positions = (key_positions.view(1, 1, 1, -1) - query_positions.view(1, 1, -1, 1)).to(torch.float32)
        bias = slopes * relative_positions
        attn_mask = torch.where(allowed, bias, torch.full_like(bias, float("-inf")))

    result = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
    return result.transpose(1, 2).to(dtype=output_dtype)


@dataclasses.dataclass
class MetalPrefillAttState(BasePrefillAttState):
    def init_state(self):
        infer_state = self.infer_state
        metadata = torch.stack(
            (
                infer_state.b_req_idx,
                infer_state.b_seq_len,
                infer_state.b_ready_cache_len,
                infer_state.b_q_seq_len,
                infer_state.b_q_start_loc,
            ),
            dim=1,
        )
        metadata = metadata.detach().to(device="cpu", dtype=torch.int64).tolist()
        request_to_token = infer_state.req_manager.req_to_token_indexs

        length_groups = {}
        for req_idx, seq_len, ready_len, query_len, query_start in metadata:
            if query_len > 0:
                length_groups.setdefault((seq_len, ready_len, query_len), []).append((req_idx, query_start))

        self._groups = []
        for (seq_len, ready_len, query_len), group in length_groups.items():
            device = request_to_token.device
            req_rows = torch.tensor([req for req, _ in group], dtype=torch.long, device=device)
            token_indexes = request_to_token[:, :seq_len].index_select(0, req_rows).to(dtype=torch.long)
            flat_token_indexes = token_indexes.reshape(-1)
            query_starts = torch.tensor([start for _, start in group], dtype=torch.long, device=device)
            query_offsets = torch.arange(query_len, dtype=torch.long, device=device)
            query_indexes = (query_starts[:, None] + query_offsets[None, :]).reshape(-1)
            lengths = torch.full((len(group),), seq_len, dtype=torch.int32, device=device) if query_len == 1 else None
            self._groups.append(
                (len(group), seq_len, ready_len, query_len, flat_token_indexes, query_indexes, req_rows, lengths)
            )

    def prefill_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        # Request metadata and token mappings are fixed for this forward and
        # shared by every attention layer. Read/build them once in init_state.
        if not hasattr(self, "_groups"):
            self.init_state()
        output = alloc_func(q.shape, dtype=q.dtype, device=q.device)
        for count, seq_len, ready_len, query_len, flat_token_indexes, query_indexes, req_rows, lengths in self._groups:
            query = q.index_select(0, query_indexes)
            # The shared mixed scheduler sends ordinary decode rows through
            # prefill. Consume their existing paged KV directly, just as decode
            # does; no cache copy or alternate request ownership is needed.
            if query_len == 1 and ready_len == seq_len - 1 and _supports_paged_attention(att_control):
                from .paged_attention import decode

                group_output = decode(
                    query,
                    k,
                    v,
                    self.infer_state.req_manager.req_to_token_indexs,
                    req_rows,
                    lengths,
                    seq_len,
                )
                if group_output is not None:
                    output.index_copy_(0, query_indexes, group_output)
                    continue
            gathered = gather_kv(k, v, flat_token_indexes)
            if gathered is None:
                gathered = k.index_select(0, flat_token_indexes), v.index_select(0, flat_token_indexes)
            key = gathered[0].view(count, seq_len, k.shape[1], k.shape[2])
            value = gathered[1].view(count, seq_len, v.shape[1], v.shape[2])
            query = query.view(count, query_len, q.shape[1], q.shape[2])
            group_output = _attend_group(query, key, value, ready_len, att_control)
            output.index_copy_(0, query_indexes, group_output.reshape(-1, q.shape[1], q.shape[2]))
        return output


@dataclasses.dataclass
class MetalDecodeAttState(BaseDecodeAttState):
    def init_state(self):
        # The native paged path consumes device metadata directly. Build the
        # grouped SDPA fallback lazily only if a layer actually needs it.
        self._groups = None

    def _init_groups(self):
        infer_state = self.infer_state
        metadata = torch.stack((infer_state.b_req_idx, infer_state.b_seq_len), dim=1)
        metadata = metadata.detach().to(device="cpu", dtype=torch.int64).tolist()
        request_to_token = infer_state.req_manager.req_to_token_indexs
        length_groups = {}
        for batch_idx, (req_idx, seq_len) in enumerate(metadata):
            length_groups.setdefault(seq_len, []).append((batch_idx, req_idx))
        self._groups = []
        for seq_len, group in length_groups.items():
            device = request_to_token.device
            batch_rows = torch.tensor([row for row, _ in group], dtype=torch.long, device=device)
            req_rows = torch.tensor([req for _, req in group], dtype=torch.long, device=device)
            token_indexes = request_to_token[:, :seq_len].index_select(0, req_rows).to(dtype=torch.long)
            self._groups.append((len(group), seq_len, batch_rows, token_indexes.reshape(-1)))

    def decode_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        infer_state = self.infer_state
        if q.shape[0] != infer_state.b_req_idx.shape[0]:
            raise NotImplementedError("Metal decode attention currently supports one query token per request.")

        if _supports_paged_attention(att_control):
            from .paged_attention import decode

            result = decode(
                q,
                k,
                v,
                infer_state.req_manager.req_to_token_indexs,
                infer_state.b_req_idx,
                infer_state.b_seq_len,
                getattr(infer_state, "max_kv_seq_len", None),
                alloc_func,
            )
            if result is not None:
                return result

        output = alloc_func(q.shape, dtype=q.dtype, device=q.device)
        if getattr(self, "_groups", None) is None:
            self._init_groups()
        for count, seq_len, batch_rows, flat_token_indexes in self._groups:
            gathered = gather_kv(k, v, flat_token_indexes)
            if gathered is None:
                gathered = k.index_select(0, flat_token_indexes), v.index_select(0, flat_token_indexes)
            key = gathered[0].view(count, seq_len, k.shape[1], k.shape[2])
            value = gathered[1].view(count, seq_len, v.shape[1], v.shape[2])
            group_output = _attend_decode_group(q.index_select(0, batch_rows), key, value, seq_len, att_control)
            output.index_copy_(0, batch_rows, group_output)
        return output


def get_prefill_att_backend_class(index=0, priority_list=None):
    return MetalAttBackend


def get_decode_att_backend_class(index=0, priority_list=None):
    return MetalAttBackend
