import torch

from lightllm.models.llama.layer_infer.pre_layer_infer import LlamaPreLayerInfer
from ..layer_infer.transformer_layer_infer import Qwen4ExpTransformerLayerInfer
from ..multimodal import canonical_image_ids


class Qwen4ExpMTPPreLayerInfer(LlamaPreLayerInfer):
    def __init__(self, network_config):
        super().__init__(network_config)
        self.image_token_id = network_config.get("image_token_id")

    def context_forward(self, input_ids, infer_state, layer_weight):
        images = [image for params in (infer_state.multimodal_params or []) for image in params.get("images", [])]
        if images:
            starts = torch.tensor([image["token_id"] for image in images], dtype=torch.int64, device=input_ids.device)
            lengths = torch.tensor([image["token_num"] for image in images], dtype=torch.int64, device=input_ids.device)
            input_ids = canonical_image_ids(input_ids, starts, lengths, self.image_token_id)
        return self.token_forward(input_ids, infer_state, layer_weight)

    def token_forward(self, input_ids, infer_state, layer_weight):
        if getattr(infer_state, "qwen4_exp_graph_preparation", False):
            infer_state.prepare_qwen_state()
        embedding = super().token_forward(input_ids, infer_state, layer_weight)
        return layer_weight.fusion.component(embedding, infer_state.mtp_draft_input_hiddens)


class Qwen4ExpMTPTransformerLayerInfer(Qwen4ExpTransformerLayerInfer):
    def __init__(self, layer_num, network_config):
        config = dict(network_config, layer_types=None, full_attention_interval=1)
        super().__init__(layer_num, config)
