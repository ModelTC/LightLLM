from lightllm.models.qwen3_vl.layer_infer.pre_layer_infer import Qwen3VLMultimodalPreLayerInfer
from ..multimodal import canonical_image_ids


class Qwen4ExpPreLayerInfer(Qwen3VLMultimodalPreLayerInfer):
    def __init__(self, network_config):
        super().__init__(network_config)
        self.branches = network_config["hc_count"]
        self.image_token_id = network_config.get("image_token_id")

    def context_forward(self, input_ids, infer_state, layer_weight):
        if not any(params.get("images") or params.get("audios") for params in (infer_state.multimodal_params or [])):
            return self.token_forward(input_ids, infer_state, layer_weight)
        embeddings = super().context_forward(input_ids, infer_state, layer_weight)
        infer_state.qwen4_exp_lexical_input_ids = canonical_image_ids(
            input_ids, infer_state.img_start_token_ids, infer_state.img_token_lens, self.image_token_id
        )
        return embeddings.repeat(1, self.branches)

    def token_forward(self, input_ids, infer_state, layer_weight):
        if getattr(infer_state, "qwen4_exp_graph_preparation", False):
            infer_state.prepare_qwen_state()
        infer_state.qwen4_exp_lexical_input_ids = input_ids
        embeddings = super().token_forward(input_ids, infer_state, layer_weight)
        return embeddings.repeat(1, self.branches)
