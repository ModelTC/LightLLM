from lightllm.models.qwen3_5_dflash.model import Qwen3_5DFlashModel
from lightllm.models.qwen3_dspark.layer_infer.post_layer_infer import (
    Qwen3DSparkPostLayerInfer,
)
from lightllm.models.qwen3_dspark.layer_weights.pre_and_post_layer_weight import (
    Qwen3DSparkPreAndPostLayerWeight,
)


class Qwen3_5DSparkModel(Qwen3_5DFlashModel):
    """DSpark draft model paired with a Qwen3.5 hybrid-attention target.

    DeepSpec exports these checkpoints as ``Qwen3DSparkModel`` because the
    draft backbone itself is a stack of Qwen3 full-attention layers.  The
    target pairing still matters at serving time: Qwen3.5 uses draft-owned
    rotary caches and a target-owned compatible KV cache, while proposal logits
    use DSpark's Markov/confidence heads and the ordinary zero-based block
    layout.
    """

    pre_and_post_weight_class = Qwen3DSparkPreAndPostLayerWeight
    post_layer_infer_class = Qwen3DSparkPostLayerInfer
    share_target_embedding_and_lm_head = False
