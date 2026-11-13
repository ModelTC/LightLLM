"""Per-execution attention contexts and Graph replay preparation.

Request tensors remain owned by InferStateInfo. This runtime borrows them through
AttentionBatch, owns provider contexts, and borrows scratch from model execution slots.
"""

from dataclasses import replace

from lightllm.common.attention.api import AttentionPhase, ContextCapacity

from .attention_batch import AttentionBatchBuilder


class AttentionRuntime:
    def __init__(self):
        self.groups = ()
        self.contexts = {}
        self.batch = None

    def prepare(self, state, model):
        from lightllm.common.basemodel.attention import get_attention_workspace

        state.device_runtime = model.device_runtime
        if model._attention_storage_generation != state.att_state_manager.storage_generation:
            raise RuntimeError("attention storage changed; rebind layers and recapture CUDA graphs")
        phase = AttentionPhase.PREFILL if state.is_prefill else AttentionPhase.DECODE
        self.batch = AttentionBatchBuilder.from_infer_state(state)
        if state.is_cuda_graph_replay:
            return
        capacity = ContextCapacity(
            phase,
            state.batch_size,
            state.input_ids.shape[0],
            model.graph_max_len_in_batch if state.is_cuda_graph_capture else state.max_kv_seq_len,
            state.max_q_seq_len,
        )
        for group_id, group in enumerate(self.groups):
            workspace = get_attention_workspace(model, group_id, state.microbatch_index, state.input_ids.device)
            workspace.begin()
            backend = group.backend_for(phase)
            if state.is_cuda_graph_capture:
                workspace = workspace.fork()
            workspace.reserve(backend.resource_requirements(capacity, capture=state.is_cuda_graph_capture))
            context = backend.allocate_context(capacity, capture=state.is_cuda_graph_capture, workspace=workspace)
            backend.prepare(context, self.batch)
            self.contexts[group_id] = context

    def attention_context(self, binding):
        context = self.contexts[binding.group_id]
        if context.backend_identity is not binding.backend or context.capacity.phase is not binding.spec.phase:
            raise ValueError("layer binding does not belong to this attention execution")
        return context

    def update(self):
        for group_id, context in self.contexts.items():
            self.groups[group_id].backend_for(context.capacity.phase).update(context)

    def finish(self):
        for context in self.contexts.values():
            context.workspace.finish()

    def check_replay_source(self, source):
        if len(self.groups) != len(source.groups) or any(
            current is not incoming for current, incoming in zip(self.groups, source.groups)
        ):
            raise ValueError("captured attention belongs to a different model role or group")

    def prepare_replay(self, source):
        """Copy borrowed inputs in place and refresh provider plans outside the Graph."""
        self.check_replay_source(source)
        for context in self.contexts.values():
            context.capacity.check(source.batch)
        for context in self.contexts.values():
            context.workspace.begin()
        self.batch.copy_inputs(source.batch)
        batch = replace(
            self.batch,
            max_kv_seq_len=source.batch.max_kv_seq_len,
            max_q_seq_len=source.batch.max_q_seq_len,
            total_token_num=source.batch.total_token_num,
        )
        for group_id, context in self.contexts.items():
            self.groups[group_id].backend_for(context.capacity.phase).prepare(context, batch)
