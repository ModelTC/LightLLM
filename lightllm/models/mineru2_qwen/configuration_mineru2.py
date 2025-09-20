from transformers import Qwen2Config


class Mineru2QwenConfig(Qwen2Config):
    model_type = "mineru2_qwen"

    def __init__(
        self,
        image_token_index=151646,
        ignore_index=-100,
        mm_vision_tower="siglip-so400m-patch14-384",
        mm_hidden_size=1152,
        mm_patch_merge_type="spatial_unpad",
        image_aspect_ratio="anyres",
        image_grid_pinpoints=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_token_index = image_token_index
        self.ignore_index = ignore_index
        self.mm_vision_tower = mm_vision_tower
        self.mm_hidden_size = mm_hidden_size
        self.mm_patch_merge_type = mm_patch_merge_type
        self.image_aspect_ratio = image_aspect_ratio
        self.image_grid_pinpoints = (
            image_grid_pinpoints if image_grid_pinpoints is not None else [[384, 1152], [768, 768], [1152, 384]]
        )
