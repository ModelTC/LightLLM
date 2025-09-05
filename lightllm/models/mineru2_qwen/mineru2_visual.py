import re

from typing import List
from io import BytesIO
from PIL import Image

import torch
import torch.nn as nn
from transformers import (
    CLIPVisionModel,
    CLIPVisionConfig,
    SiglipVisionConfig,
    SiglipVisionModel,
)

from .configuration_mineru2 import Mineru2QwenConfig
from .image_processing_mineru2 import Mineru2ImageProcessor

from lightllm.server.multimodal_params import ImageItem
from lightllm.server.embed_cache.utils import read_shm, get_shm_name_data
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def build_vision_tower(config: Mineru2QwenConfig):
    vision_tower = getattr(config, "mm_vision_tower", getattr(config, "vision_tower", ""))
    model_path = getattr(config, "_name_or_path", "")

    if "clip" in vision_tower.lower():
        if model_path:
            vision_config = CLIPVisionConfig.from_pretrained(f"{model_path}/{vision_tower}")
            return CLIPVisionModel(vision_config)
        else:
            vision_config = CLIPVisionConfig.from_pretrained(vision_tower)
            return CLIPVisionModel(vision_config)
    elif "siglip" in vision_tower.lower():
        if model_path:
            vision_config = SiglipVisionConfig.from_pretrained(f"{model_path}/{vision_tower}")
            return SiglipVisionModel(vision_config)
        else:
            vision_config = SiglipVisionConfig.from_pretrained(vision_tower)
            return SiglipVisionModel(vision_config)
    else:
        raise ValueError(f"Unknown vision tower: {model_path}")


def build_vision_projector(config: Mineru2QwenConfig):
    projector_type = getattr(config, "mm_projector_type", "linear")

    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == "identity":
        return nn.Identity()

    raise ValueError(f"Unknown projector type: {projector_type}")


class Mineru2VisionModel:
    def __init__(self):
        pass

    def load_model(self, weight_dir):
        print(f"[debug] load_model: {weight_dir}")
        # config_file = os.path.join(weight_dir, "config.json")
        vision_config = Mineru2QwenConfig.from_pretrained(weight_dir)

        self.vision_tower = build_vision_tower(vision_config)
        self.projector = build_vision_projector(vision_config)
        self.image_processor = Mineru2ImageProcessor()

    def cuda(self):
        self.vision_tower = self.vision_tower.cuda()
        self.projector = self.projector.cuda()
        return self

    def forward(self, x):
        return self.projector(self.vision_tower(x))

    def encode(self, images: List[ImageItem]):
        img_tensors = []
        uuids = []
        valid_id = 0
        valid_ids = []

        for i, img in enumerate(images):
            if isinstance(img, ImageItem):
                uuids.append(img.uuid)
                print(f"[debug] read_shm: {get_shm_name_data(img.uuid)}")
                image_data = read_shm(get_shm_name_data(img.uuid))
                image_data = Image.open(BytesIO(image_data)).convert("RGB")
                t = self.image_processor.preprocess(image_data, return_tensors="pt")["pixel_values"]
                img_tensors.append(t)
            else:
                raise Exception("Unsupport input types: {} for {}".format(type(img), img))

            cur_num = img_tensors[-1].shape[0]
            valid_ids.append([valid_id, valid_id + cur_num])
            valid_id += cur_num

        if len(img_tensors) <= 0:
            return None

        img = torch.cat(img_tensors, dim=0)
        all_img_embeds = self.forward(img)

        return all_img_embeds, uuids, valid_ids
