import torch
from lightllm.models.qwen_vl.layer_infer.pre_layer_infer import LlamaMultimodalPreLayerInfer


class Gemma3PreLayerInfer(LlamaMultimodalPreLayerInfer):
    def __init__(self, network_config):
        super().__init__(network_config)
        self.embed_scale = torch.tensor(network_config["hidden_size"] ** 0.5, dtype=torch.float32)
        self.boi_token_index: int = 255_999
        self.eoi_token_index: int = 256_000
        return

    def context_forward(self, input_ids, infer_state, layer_weight):
        device = layer_weight.wte_weight_.weight.device
        weight_mask = torch.zeros((len(input_ids)), dtype=torch.float32, device=device)

        # TODO
        scale = self.embed_scale
        for idx, input_id in enumerate(input_ids):
            if input_id == self.boi_token_index:
                weight_mask[idx] = scale
                scale = 1.0
            elif input_id == self.eoi_token_index:
                scale = self.embed_scale
                weight_mask[idx] = scale
            else:
                weight_mask[idx] = scale

        out = super().context_forward(input_ids, infer_state, layer_weight)
        input_dtype = out.dtype
        return (out.float() * weight_mask.unsqueeze(1).float()).to(input_dtype)

    def token_forward(self, input_ids, infer_state, layer_weight):
        input_embedding = super().token_forward(input_ids, infer_state, layer_weight)
        input_dtype = input_embedding.dtype
        return (input_embedding.float() * self.embed_scale.to(input_embedding.device).float()).to(input_dtype)
