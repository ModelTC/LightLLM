import torch
import numpy as np
from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight

from lightllm.models.internlm2.layer_weights.pre_and_post_layer_weight import Internlm2PreAndPostLayerWeight
from lightllm.models.vit.model import VisionTransformer
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.common.image_cache_manager import image_cache_manager


# add key: language_model.xxx -> xxx
# only change keys at PreAndPostLayerWeight load, TransformLayerWeight is correct now
def rename_weight_keys(weights):
    prefix = "language_model."
    keys = list(weights.keys())
    for k in keys:
        if prefix in k:
            weights[k[len(prefix) :]] = weights[k]


class InternVLPreAndPostLayerWeight(LlamaPreAndPostLayerWeight):
    def __init__(self, data_type, network_config, mode):
        super().__init__(data_type, network_config, mode)
        # if we don't assign an extra process for visual model, we need initialize the image cache manager here
        if get_env_start_args().disable_extra_process_for_multimodal:
            kvargs = {
                "weight_dir": get_env_start_args().model_dir,
                "data_type": self.data_type_,
                "quant_type": get_env_start_args().vit_quant_type,
                "quant_cfg": get_env_start_args().vit_quant_cfg,
                "max_batch_size": get_env_start_args().visual_infer_batch_size,
            }
            self.visual_model = VisionTransformer(
                kvargs=kvargs,
            )
            image_cache_manager.set_max_size(get_env_start_args().cache_capacity * 2)
        return

    def load_hf_weights(self, weights):
        rename_weight_keys(weights)
        super().load_hf_weights(weights)


class InternVLPhi3PreAndPostLayerWeight(LlamaPreAndPostLayerWeight):
    def __init__(self, data_type, network_config, mode):
        super().__init__(data_type, network_config, mode)
        # if we don't assign an extra process for visual model, we need initialize the image cache manager here
        if get_env_start_args().disable_extra_process_for_multimodal:
            kvargs = {
                "weight_dir": get_env_start_args().model_dir,
                "data_type": self.data_type_,
                "quant_type": get_env_start_args().vit_quant_type,
                "quant_cfg": get_env_start_args().vit_quant_cfg,
                "max_batch_size": get_env_start_args().visual_infer_batch_size,
            }
            self.visual_model = VisionTransformer(
                kvargs=kvargs,
            )
            image_cache_manager.set_max_size(get_env_start_args().cache_capacity * 2)
        return

    def load_hf_weights(self, weights):
        rename_weight_keys(weights)
        super().load_hf_weights(weights)
        return


class InternVLInternlm2PreAndPostLayerWeight(Internlm2PreAndPostLayerWeight):
    def __init__(self, data_type, network_config, mode):
        super().__init__(data_type, network_config, mode)
        # if we don't assign an extra process for visual model, we need initialize the image cache manager here
        if get_env_start_args().disable_extra_process_for_multimodal:
            kvargs = {
                "weight_dir": get_env_start_args().model_dir,
                "data_type": self.data_type_,
                "quant_type": get_env_start_args().vit_quant_type,
                "quant_cfg": get_env_start_args().vit_quant_cfg,
                "max_batch_size": get_env_start_args().visual_infer_batch_size,
            }
            self.visual_model = VisionTransformer(
                kvargs=kvargs,
            )
            image_cache_manager.set_max_size(get_env_start_args().cache_capacity * 2)
        return

    def load_hf_weights(self, weights):
        rename_weight_keys(weights)
        super().load_hf_weights(weights)
        return


class InternVLLlamaPreAndPostLayerWeight(LlamaPreAndPostLayerWeight):
    def __init__(self, data_type, network_config, mode):
        super().__init__(data_type, network_config, mode)
        # if we don't assign an extra process for visual model, we need initialize the image cache manager here
        if get_env_start_args().disable_extra_process_for_multimodal:
            kvargs = {
                "weight_dir": get_env_start_args().model_dir,
                "data_type": self.data_type_,
                "quant_type": get_env_start_args().vit_quant_type,
                "quant_cfg": get_env_start_args().vit_quant_cfg,
                "max_batch_size": get_env_start_args().visual_infer_batch_size,
            }
            self.visual_model = VisionTransformer(
                kvargs=kvargs,
            )
            image_cache_manager.set_max_size(get_env_start_args().cache_capacity * 2)
        return

    def load_hf_weights(self, weights):
        rename_weight_keys(weights)
        super().load_hf_weights(weights)
        return
