from lightllm.models.qwen3_dflash.model import Qwen3DFlashModel
from lightllm.models.qwen3_dspark.layer_infer.post_layer_infer import Qwen3DSparkPostLayerInfer
from lightllm.models.qwen3_dspark.layer_weights.pre_and_post_layer_weight import Qwen3DSparkPreAndPostLayerWeight


class Qwen3DSparkModel(Qwen3DFlashModel):
    """Qwen3 DSpark draft model.

    DSpark keeps the DFlash block backbone.  Its extra Markov/confidence heads
    run in the post layer so the proposer can consume corrected logits through
    the same token-generation path as DFlash.
    """

    pre_and_post_weight_class = Qwen3DSparkPreAndPostLayerWeight
    post_layer_infer_class = Qwen3DSparkPostLayerInfer
