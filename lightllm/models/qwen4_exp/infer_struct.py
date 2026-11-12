import weakref

from lightllm.models.qwen3_5.infer_struct import Qwen35InferStateInfo
from .batch import PackedBatchMetadata
from .position import combine_mrope
from .state import QSABatch


class Qwen4ExpInferStateInfo(Qwen35InferStateInfo):
    def init_some_extra_state(self, model):
        self._qwen4_exp_model = weakref.ref(model)
        self.qwen4_exp_config = model.config
        self.qwen4_exp_fixed_decode = model.fixed_decode_config
        self.qwen4_exp_segment_capacity = model.max_req_num + 1 if self.is_prefill else None
        self.qwen4_exp_auto_commit = self.is_prefill or model.args.mtp_step == 0
        self.qwen4_exp_graph_preparation = not self.is_prefill and (
            self.is_cuda_graph_capture or self.is_cuda_graph_replay
        )
        if not self.qwen4_exp_graph_preparation:
            super().init_some_extra_state(model)
            self.prepare_qwen_state()

    def prepare_qwen_state(self):
        if self.qwen4_exp_graph_preparation:
            super().init_some_extra_state(self._qwen4_exp_model())
        config = self.qwen4_exp_config
        ratio = config["indexer_compress_ratio"]
        self.qwen4_exp_metadata = PackedBatchMetadata.prepare(
            self.b_req_idx,
            self.b_seq_len,
            self.b_q_seq_len,
            self.input_ids.numel(),
            self.req_manager.HOLD_REQUEST_ID,
            ratio,
            segment_capacity=self.qwen4_exp_segment_capacity,
            decode=not self.is_prefill,
        )
        cos, sin = combine_mrope(self.position_cos, self.position_sin, config["rope_scaling"]["mrope_section"])
        self.qwen4_exp_batch = QSABatch.from_metadata(
            self.qwen4_exp_metadata, self.req_manager.req_to_token_indexs, self.mem_index, cos, sin, ratio
        )
        from .attention_state import Qwen4ExpAttentionState
        from .lexical_state import Qwen4ExpLexicalState
        from .qsa_compressor import QSACompressorState

        states = self.att_state_manager.get_full_attention_states(self.att_state_role)
        bounded = self.att_state_manager.get_bounded_attention_states(self.att_state_role)
        compressor = next((state for state in bounded if isinstance(state, QSACompressorState)), None)
        self.qwen4_exp_qsa = {}
        for state in states:
            if isinstance(state, Qwen4ExpAttentionState):
                state.store_positions(self.qwen4_exp_batch.mem_indices, cos, sin)
                for group in state.config.logical_layer_groups:
                    for layer in group:
                        ring = compressor.get_layer_view(layer) if compressor is not None else None
                        self.qwen4_exp_qsa[layer] = state.qsa_view(layer, ring)
        self.qwen4_exp_lexical_state = next(
            (state for state in bounded if isinstance(state, Qwen4ExpLexicalState)),
            None,
        )
        self.qwen4_exp_ple = (
            {}
            if self.qwen4_exp_lexical_state is None
            else self.qwen4_exp_lexical_state.prepare_lexical(
                self.qwen4_exp_metadata, reset_empty_prefix=self.is_prefill
            )
        )

    def commit_lexical(self, accepted_lengths):
        self.qwen4_exp_lexical_state.commit_lexical(self.qwen4_exp_metadata, self.qwen4_exp_ple, accepted_lengths)

    def _copy_qwen_state(self, new_state):
        if self.qwen4_exp_auto_commit != new_state.qwen4_exp_auto_commit:
            raise ValueError("Qwen graph replay cannot change lexical commit mode")
        self.qwen4_exp_metadata.copy_from(new_state.qwen4_exp_metadata)
        old, new = self.qwen4_exp_batch, new_state.qwen4_exp_batch
        if old.max_new_blocks != new.max_new_blocks:
            raise ValueError("Qwen graph replay requires the same token capacity")
        for name in ("mem_indices", "cos", "sin"):
            getattr(old, name).copy_(getattr(new, name), non_blocking=True)
        for layer, batch in self.qwen4_exp_ple.items():
            source = new_state.qwen4_exp_ple[layer]
            batch.ngram_history.copy_(source.ngram_history, non_blocking=True)
            batch.conv_history.copy_(source.conv_history, non_blocking=True)

    def copy_for_cuda_graph(self, new_infer_state):
        super().copy_for_cuda_graph(new_infer_state)
        if self.qwen4_exp_graph_preparation:
            if not new_infer_state.qwen4_exp_graph_preparation:
                raise ValueError("Qwen graph replay cannot change state preparation mode")
        else:
            self._copy_qwen_state(new_infer_state)

    def copy_for_prefill_cuda_graph(self, new_infer_state):
        super().copy_for_prefill_cuda_graph(new_infer_state)
        self._copy_qwen_state(new_infer_state)
