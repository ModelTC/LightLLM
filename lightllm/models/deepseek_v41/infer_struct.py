from copy import copy

import torch

from lightllm.models.deepseek_v4.infer_struct import DeepseekV4InferStateInfo


class DeepseekV41InferStateInfo(DeepseekV4InferStateInfo):
    def init_some_extra_state(self, model):
        super().init_some_extra_state(model)
        self.v41_sparse_selection = None
        self.candidate_block_mask = None
        self.engram_hashes = model.engram_hasher(self) if model.engram_hasher is not None else None
        self.decoder_state = None
        self.decoder_token_indices = None

    def init_decoder_state(self):
        """Keep each extend's last window after the final global-KV producer.

        The original state still owns request advancement and prefix checkpoints.
        Decoder SWA reads only replayed rows, including after a prefix-cache hit.
        """
        window = self.req_manager.cache_config.sliding_window
        lengths = self.b_q_seq_len.clamp_max(window)
        ready = self.b_seq_len - lengths
        batch = self._dsv4_token_to_batch_idx.long()
        rows = (self.position_ids >= ready[batch]).nonzero(as_tuple=True)[0]
        state = copy(self)
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
            "dsv4_swa_indices",
            "dsv4_swa_write_slots",
            "dsv4_swa_runtime_write_slots",
            "_dsv4_token_to_batch_idx",
        ):
            setattr(state, name, getattr(self, name)[rows])
        state.b_ready_cache_len = ready
        state.b_q_seq_len = lengths
        state.b1_cu_q_seq_len = torch.cat((lengths.new_zeros(1), lengths.cumsum(0, dtype=lengths.dtype)))
        state.b_q_start_loc = state.b1_cu_q_seq_len[:-1]
        state.max_q_seq_len = min(self.max_q_seq_len, window)
        state.dsv4_swa_lengths = state.position_ids - ready[batch[rows]] + 1
        state.dsv4_swa_indices.masked_fill_(
            torch.arange(window, device=rows.device)[None, :] >= state.dsv4_swa_lengths[:, None], -1
        )
        state.v41_sparse_selection = {name: value[rows] for name, value in self.v41_sparse_selection.items()}
        state.candidate_block_mask = self.candidate_block_mask[rows]
        if self.engram_hashes is not None:
            state.engram_hashes = self.engram_hashes[rows]
        self.decoder_state = state
        self.decoder_token_indices = rows
        return rows
