import torch

from lightllm.models.deepseek_v4.deepseek_v4_visual import DeepseekV4VisionModel, ViT, get_vision_cos_sin
from .image_processor import get_image_grid


class DeepseekV41ViT(ViT):
    def get_rope(self, n_h, n_w, device):
        return get_vision_cos_sin(n_h, n_w, self.rope_dim, self.rope_theta, device)


class DeepseekV41VisionModel(DeepseekV4VisionModel):
    vit_class = DeepseekV41ViT
    get_image_grid = staticmethod(get_image_grid)
    sentinel_names = ("image_start", "image_newline", "image_end")

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
        )

    @staticmethod
    def build_image_block(row_major_embeds, n_llm_h, n_llm_w, sentinel_table):
        start, newline, end = sentinel_table
        rows = row_major_embeds.view(n_llm_h, n_llm_w, -1)
        rows = torch.cat((rows, newline.expand(n_llm_h, 1, -1)), dim=1)
        return torch.cat((start[None], rows.flatten(0, 1), end[None]))
