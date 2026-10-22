from lightllm.models.deepseek_v4.deepseek_v4_visual import DeepseekV4VisionModel


class DeepseekV41VisionModel(DeepseekV4VisionModel):
    def __init__(self, kvargs, **config):
        vision = config["vision_config"]
        super().__init__(
            kvargs,
            hidden_size=config["text_config"]["hidden_size"],
            vision_n_layers=vision["num_hidden_layers"],
            vision_dim=vision["hidden_size"],
            vision_n_heads=vision["num_attention_heads"],
            vision_inter_dim=vision["intermediate_size"],
            vision_patch_size=vision["patch_size"],
            vision_rope_theta=vision["rope_theta"],
            vision_downsample_ratio=vision["downsample_ratio"],
            vision_max_n_token=vision["max_image_tokens"],
            vision_min_pixels=vision["min_pixels"],
            vision_max_wh_ratio=vision.get("max_wh_ratio"),
            vision_row_major=True,
        )
