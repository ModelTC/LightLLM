import torch
import numpy as np
from lightllm.utils.envs_utils import get_env_start_args
from transformers.configuration_utils import PretrainedConfig
from lightllm.models.qwen2.layer_weights.pre_and_post_layer_weight import Qwen2PreAndPostLayerWeight
from lightllm.models.qwen2_vl.qwen2_visual import Qwen2VLTransformer


def build_visual_model(args, data_type: torch.dtype):
    if args.disable_extra_process_for_multimodal:
        kvargs = {
            "weight_dir": args.model_dir,
            "data_type": args.data_type,
            "quant_type": args.vit_quant_type,
            "quant_cfg": args.vit_quant_cfg,
            "max_batch_size": args.visual_infer_batch_size,
        }
        model_cfg, _ = PretrainedConfig.get_config_dict(kvargs["weight_dir"])
        return Qwen2VLTransformer(kvargs=kvargs, **model_cfg["vision_config"]).eval().to(dtype=data_type)
    return None


class Qwen2VLPreAndPostLayerWeight(Qwen2PreAndPostLayerWeight):
    def __init__(self, data_type, network_config, mode):
        super().__init__(data_type, network_config, mode)
        self.visual_model = build_visual_model(get_env_start_args(), data_type)
        return
