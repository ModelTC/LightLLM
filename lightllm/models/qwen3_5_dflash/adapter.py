from lightllm.models.qwen3_dflash.adapter import Qwen3DFlashModelAdapter
from lightllm.models.qwen3_5_dflash.layer_weights.pre_and_post_layer_weight import Qwen35DFlashPreAndPostLayerWeight


class Qwen35DFlashModelAdapter(Qwen3DFlashModelAdapter):
    pre_and_post_weight_class = Qwen35DFlashPreAndPostLayerWeight
    share_main_weights = True
