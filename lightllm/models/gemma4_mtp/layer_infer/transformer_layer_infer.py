from lightllm.models.gemma4.layer_infer.transformer_layer_infer import Gemma4TransformerLayerInfer


class Gemma4MTPTransformerLayerInfer(Gemma4TransformerLayerInfer):
    """
    Gemma-4 assistant decoder block. Identical to the target's block except the
    attention is forced into the KV-shared (Q-only) path: K/V are read from the
    *target model's* committed cache, and nothing is computed or written here.

    `layer_num` is the assistant-local index (0..num_mtp_layers-1) - used for
    config lookups (layer_types / RoPE table / per-layer shapes).
    `kv_share_target_layer` is the *target model's* layer index whose KV cache
    this layer reads (the target's last non-KV-shared layer of the same attention
    type). The MTP network_config has num_kv_shared_layers forced to 0 so the
    parent __init__ leaves is_kv_shared_ False; it is forced True here.
    """

    def __init__(self, layer_num, network_config, kv_share_target_layer):
        super().__init__(layer_num, network_config)
        self.is_kv_shared_ = True
        self.kv_share_target_layer_ = kv_share_target_layer
