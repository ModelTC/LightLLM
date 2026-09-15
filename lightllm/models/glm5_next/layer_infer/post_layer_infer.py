from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer
from lightllm.utils.envs_utils import get_env_start_args


class Glm5NextPostLayerInfer(LlamaPostLayerInfer):
    """Expose the normalized target/draft hidden expected by GLM NextN."""

    def __init__(self, network_config):
        super().__init__(network_config)
        self.mtp_enabled = get_env_start_args().mtp_mode is not None

    def token_forward(self, input_embdings, infer_state, layer_weight):
        if self.mtp_enabled:
            layer_weight.final_norm_weight_(input=input_embdings, eps=self.eps_, out=input_embdings)
        return super().token_forward(input_embdings, infer_state, layer_weight)

    def _norm(self, input, infer_state, layer_weight):
        if self.mtp_enabled:
            return input
        return super()._norm(input, infer_state, layer_weight)
