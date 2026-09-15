# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Adapted from OpenGVLab/InternVL3-8B@853e3a797a661694b1b8ece0cb72dc2b23e3dac9
# Source file: configuration_intern_vit.py
# Changes: omit example docs; config loading must not probe model/kernel dependencies.
from transformers import PretrainedConfig


class InternVisionConfig(PretrainedConfig):
    model_type = "intern_vit_6b"

    def __init__(
        self,
        num_channels=3,
        patch_size=14,
        image_size=224,
        qkv_bias=False,
        hidden_size=3200,
        num_attention_heads=25,
        intermediate_size=12800,
        qk_normalization=True,
        num_hidden_layers=48,
        use_flash_attn=True,
        hidden_act="gelu",
        norm_type="rms_norm",
        layer_norm_eps=1e-06,
        dropout=0.0,
        drop_path_rate=0.0,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.drop_path_rate = drop_path_rate
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.norm_type = norm_type
        self.qkv_bias = qkv_bias
        self.qk_normalization = qk_normalization
        self.use_flash_attn = use_flash_attn
