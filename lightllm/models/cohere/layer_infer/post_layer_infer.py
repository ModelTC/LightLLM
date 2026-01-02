import torch
from lightllm.models.cohere.infer_struct import CohereInferStateInfo
from lightllm.models.cohere.layer_weights.pre_and_post_layer_weight import CoherePreAndPostLayerWeight
from lightllm.models.cohere.triton_kernels.layernorm import layernorm_forward
from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer
from lightllm.common.build_utils import repair_config


class CoherePostLayerInfer(LlamaPostLayerInfer):
    def __init__(self, network_config, mode):
        repair_config(config=network_config, same_names=["layer_norm_eps", "rms_norm_eps"])
        super().__init__(network_config, mode)
        self.eps_ = network_config["layer_norm_eps"]
        return

    def _norm(
        self, input: torch.Tensor, infer_state: CohereInferStateInfo, layer_weight: CoherePreAndPostLayerWeight
    ) -> torch.Tensor:
        return layernorm_forward(
            input.unsqueeze(1), layer_weight.final_norm_weight_.weight.unsqueeze(0), eps=self.eps_
        ).squeeze(1)
