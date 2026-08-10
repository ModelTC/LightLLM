from lightllm.models.llama.infer_struct import LlamaInferStateInfo


class Qwen3DFlashInferStateInfo(LlamaInferStateInfo):
    """DFlash attention metadata."""

    def init_some_extra_state(self, model):
        super().init_some_extra_state(model)
        if self.is_prefill:
            self.prefill_causal = False
        else:
            self.decode_causal = False
