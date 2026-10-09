from lightllm.common.basemodel.multimodal_tokenizer import BaseMultiModalTokenizer
from lightllm.models.qwen2_vl.model import QWen2VLTokenizer
from .vision_process import Glm5NextImageProcessor


class Glm5NextTokenizer(QWen2VLTokenizer):
    def __init__(self, tokenizer, model_cfg, weight_dir):
        BaseMultiModalTokenizer.__init__(self, tokenizer)
        self.image_processor = Glm5NextImageProcessor.from_pretrained(weight_dir)
        self.image_start_id = model_cfg["image_start_token_id"]
        self.image_end_id = model_cfg["image_end_token_id"]
        self.image_token_id = model_cfg["image_token_id"]

    def get_image_token_length(self, img):
        height, width = self.image_processor.get_image_size(img.image_h, img.image_w)
        factor = self.image_processor.patch_size * self.image_processor.merge_size
        grid_h, grid_w = height // factor, width // factor
        img.grid_thwd = (1, grid_h, grid_w, 0)
        return grid_h * grid_w
