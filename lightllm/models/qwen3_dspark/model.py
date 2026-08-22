from types import SimpleNamespace

import torch

from lightllm.common.basemodel.batch_objs import ModelOutput
from lightllm.models.draft_registry import DraftModelRegistry
from lightllm.models.qwen3_dflash.model import Qwen3DFlashModel
from lightllm.models.qwen3_dspark.layer_infer.post_layer_infer import Qwen3DSparkPostLayerInfer
from lightllm.models.qwen3_dspark.layer_weights.pre_and_post_layer_weight import Qwen3DSparkPreAndPostLayerWeight


@DraftModelRegistry(model_type="qwen3", spec_modes="dspark")
class Qwen3DSparkModel(Qwen3DFlashModel):
    """Qwen3 DSpark draft model.

    DSpark keeps the DFlash block backbone.  Its extra Markov/confidence heads
    run in the post layer so the proposer can consume corrected logits through
    the same token-generation path as DFlash.
    """

    pre_and_post_weight_class = Qwen3DSparkPreAndPostLayerWeight
    post_layer_infer_class = Qwen3DSparkPostLayerInfer

    def _decode(self, model_input):
        if model_input.mtp_draft_input_hiddens is None:
            return super()._decode(model_input)

        assert model_input.mtp_draft_input_hiddens.shape[0] == model_input.batch_size

        # This path only projects target hidden and writes its RoPE'd KV.  It
        # deliberately avoids the full decode infer state and attention setup.
        position_ids = model_input.b_seq_len - 1
        infer_state = SimpleNamespace(
            mtp_draft_input_hiddens=model_input.mtp_draft_input_hiddens,
            position_cos=torch.index_select(self._cos_cached, 0, position_ids),
            position_sin=torch.index_select(self._sin_cached, 0, position_ids),
            mem_manager=self.mem_manager,
            mem_index=model_input.mem_indexes,
        )

        hidden = self.pre_infer.context_forward(None, infer_state, self.pre_post_weight)
        for layer, layer_weight in zip(self.layers_infer, self.trans_layers_weight):
            hidden = layer.context_forward(hidden, infer_state, layer_weight)

        return ModelOutput(logits=hidden.new_empty((model_input.batch_size, 1)))
