from copy import copy

import torch

from lightllm.models.deepseek_v4.infer_struct import DeepseekV4InferStateInfo

from .triton_kernel.attention import bound_swa_replay, build_decoder_tail_rows


class DeepseekV41InferStateInfo(DeepseekV4InferStateInfo):
    prefill_tail_token_num: int | None = None

    def init_some_extra_state(self, model):
        super().init_some_extra_state(model)
        self.global_kv_write_slots = self.mem_index
        if self.is_prefill and self.b_prefill_commit_start is not None:
            self.global_kv_write_slots = torch.empty_like(self.mem_index)
            bound_swa_replay(self, write_slots=self.global_kv_write_slots)
        self.v41_sparse_selection = None
        self.candidate_block_indices = None
        self.engram_hashes = model.engram_hasher(self) if self.is_prefill and model.engram_hasher is not None else None
        self.engram_stream = model.engram_stream
        self.engram_prefetch = None
        self.decoder_state = None
        self.decoder_token_indices = None

    def prefetch_engram(self, layer_weight, index):
        """Keep one local gather alive until its Engram layer consumes it."""
        stream = self.engram_stream
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            embedded = layer_weight.engram_embed_.gather_local(self.engram_hashes[:, index])
            ready = torch.cuda.Event()
            ready.record(stream)
        self.engram_prefetch = embedded, ready

    def init_decoder_state(self):
        """Keep the prompt's last window after writing decoder global KV.

        The original state still owns request advancement and prefix checkpoints.
        Decoder SWA reads only replayed rows, including after a prefix-cache hit.
        """
        window = self.req_manager.cache_config.sliding_window
        state = copy(self)
        if self.max_q_seq_len <= window:
            rows = slice(None)
        else:
            lengths = self.b_q_seq_len.clamp_max(window)
            state.b_q_seq_len = lengths
            state.b_ready_cache_len = self.b_seq_len - lengths
            state.b1_cu_q_seq_len = torch.cat((lengths.new_zeros(1), lengths.cumsum(0, dtype=lengths.dtype)))
            state.b_q_start_loc = state.b1_cu_q_seq_len[:-1]
            if self.b_seq_len.numel() == 1:
                rows = slice(-window, None)
            elif self.prefill_tail_token_num is not None:
                rows = build_decoder_tail_rows(
                    self.b1_cu_q_seq_len, state.b1_cu_q_seq_len, self.prefill_tail_token_num, window
                )
            else:
                batch = self._dsv4_token_to_batch_idx.long()
                rows = (self.position_ids >= state.b_ready_cache_len[batch]).nonzero(as_tuple=True)[0]
        # CED changes query rows while retaining this forward's SWA workspace.
        # FlashMLA schedulers cache the query row count and must be rebuilt.
        state.prefill_att_state = copy(self.prefill_att_state)
        state.prefill_att_state.init_state()
        state.indexer_metadata = None
        for name in (
            "input_ids",
            "mem_index",
            "position_ids",
            "position_cos_sliding",
            "position_sin_sliding",
            "position_cos_compress",
            "position_sin_compress",
            "dsv4_sparse_req_idx",
            "dsv4_swa_write_slots",
            "dsv4_swa_runtime_write_slots",
            "_dsv4_token_to_batch_idx",
        ):
            setattr(state, name, getattr(self, name)[rows])
        state.max_q_seq_len = min(self.max_q_seq_len, window)
        for name in ("dsv4_image_left", "dsv4_image_right"):
            value = getattr(self, name)
            if value is not None:
                setattr(state, name, value[rows])
        # Read encoder SWA slots directly when bounding the decoder rows.
        bound_swa_replay(state, rows=rows)
        state.v41_sparse_selection = {name: value[rows] for name, value in self.v41_sparse_selection.items()}
        # The first decoder indexer produces the candidates after this trim.
        if self.candidate_block_indices is not None:
            state.candidate_block_indices = self.candidate_block_indices[rows]
        if self.engram_hashes is not None:
            state.engram_hashes = self.engram_hashes[rows]
        self.decoder_state = state
        self.decoder_token_indices = rows
        return rows
