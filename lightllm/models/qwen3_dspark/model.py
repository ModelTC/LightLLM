from lightllm.models.qwen3_dflash.model import Qwen3DFlashModel
from lightllm.models.qwen3_dspark.layer_infer.post_layer_infer import Qwen3DSparkPostLayerInfer
from lightllm.models.qwen3_dspark.model_output import DSparkModelOutput
from lightllm.models.qwen3_dspark.layer_weights.pre_and_post_layer_weight import Qwen3DSparkPreAndPostLayerWeight
from lightllm.utils.tensor_utils import tensor_to_no_ref_tensor


class DSparkModelOutputMixin:
    def _token_forward(self, infer_state):
        model_output = super()._token_forward(infer_state)
        confidence_logits = self.post_infer.pop_confidence_logits()
        draft_token_ids = self.post_infer.pop_draft_token_ids()
        if infer_state.is_cuda_graph:
            if confidence_logits is not None:
                confidence_logits = tensor_to_no_ref_tensor(confidence_logits)
            if draft_token_ids is not None:
                draft_token_ids = tensor_to_no_ref_tensor(draft_token_ids)

        return DSparkModelOutput(
            logits=model_output.logits,
            spec_hidden=model_output.spec_hidden,
            confidence_logits=confidence_logits,
            draft_token_ids=draft_token_ids,
        )

    def _create_unpad_decode_model_output(self, model_output: DSparkModelOutput, origin_batch_size: int):
        padded_batch_size = model_output.logits.shape[0]
        model_output = super()._create_unpad_decode_model_output(model_output, origin_batch_size)
        if padded_batch_size == origin_batch_size:
            return model_output

        if model_output.draft_token_ids is not None:
            model_output.draft_token_ids = model_output.draft_token_ids[:origin_batch_size]
        if model_output.confidence_logits is not None:
            confidence_rows = model_output.confidence_logits.shape[0]
            assert padded_batch_size % confidence_rows == 0
            rows_per_confidence = padded_batch_size // confidence_rows
            assert origin_batch_size % rows_per_confidence == 0
            model_output.confidence_logits = model_output.confidence_logits[: origin_batch_size // rows_per_confidence]
        return model_output


class Qwen3DSparkModel(DSparkModelOutputMixin, Qwen3DFlashModel):
    """Qwen3 DSpark draft model.

    DSpark keeps the DFlash block backbone.  Its extra Markov/confidence heads
    run in the post layer so the proposer can consume corrected logits through
    the same token-generation path as DFlash.
    """

    pre_and_post_weight_class = Qwen3DSparkPreAndPostLayerWeight
    post_layer_infer_class = Qwen3DSparkPostLayerInfer

    def _verify_params(self):
        super()._verify_params()
        assert self.config.get("enable_confidence_head", False), "DSpark requires enable_confidence_head=true"
