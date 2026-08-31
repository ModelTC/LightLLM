# SPDX-License-Identifier: Apache-2.0

from lightllm.models.deepseek_mtp.layer_infer.pre_layer_infer import (
    Deepseek3MTPPreLayerInfer,
)
from lightllm.models.glm5_next_mtp.triton_kernel.zero_position_embedding import (
    zero_position_embedding_,
)


class Glm5NextMTPPreLayerInfer(Deepseek3MTPPreLayerInfer):
    """GLM NextN input fusion with the trained position-zero convention."""

    def _mtp_context_forward(self, input_embdings, infer_state, layer_weight):
        # GLM's fused_eh_norm reference zeros the token embedding at absolute
        # position zero before applying enorm.  The first target hidden remains
        # intact; only the missing previous-token embedding is suppressed.
        zero_position_embedding_(input_embdings, infer_state.position_ids)
        return super()._mtp_context_forward(input_embdings, infer_state, layer_weight)
