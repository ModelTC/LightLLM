import re

from typing import List, Tuple
from io import BytesIO
from PIL import Image

import torch
import torch.nn as nn
import numpy as np
from transformers import (
    CLIPVisionModel,
    CLIPVisionConfig,
    SiglipVisionConfig,
    SiglipVisionModel,
)

from .configuration_mineru2 import Mineru2QwenConfig
from .image_processing_mineru2 import Mineru2ImageProcessor, expand2square, process_anyres_image

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
            print(f"[debug] load clip from {model_path}/{vision_tower}")
            return CLIPVisionModel(vision_config)
        else:
            vision_config = CLIPVisionConfig.from_pretrained(vision_tower)
            print(f"[debug] load clip from {vision_tower}")
            return CLIPVisionModel(vision_config)
    elif "siglip" in vision_tower.lower():
        if model_path:
            vision_config = SiglipVisionConfig.from_pretrained(f"{model_path}/{vision_tower}")
            print(f"[debug] load siglip from {model_path}/{vision_tower}")
        else:
            vision_config = SiglipVisionConfig.from_pretrained(vision_tower)
            print(f"[debug] load siglip from {vision_tower}")
        # 对齐ref：去掉最后一层并禁用head
        if hasattr(vision_config, "num_hidden_layers"):
            vision_config.num_hidden_layers = max(0, vision_config.num_hidden_layers - 1)
        if hasattr(vision_config, "vision_use_head"):
            vision_config.vision_use_head = False
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
        print(f"[debug] load vision model: {weight_dir}")
        vision_config = Mineru2QwenConfig.from_pretrained(weight_dir)

        self.vision_tower = build_vision_tower(vision_config)
        self.projector = build_vision_projector(vision_config)
        # 取配置参数传下去
        self.image_processor = Mineru2ImageProcessor(
            image_aspect_ratio=getattr(vision_config, "image_aspect_ratio", None),
            image_grid_pinpoints=getattr(vision_config, "image_grid_pinpoints", None),
        )

    def cuda(self):
        self.vision_tower = self.vision_tower.cuda()
        self.projector = self.projector.cuda()
        return self

    def forward(self, x) -> torch.Tensor:
        vision_out = self.vision_tower(x, output_hidden_states=True)
        last_hidden = vision_out.hidden_states[-1]
        # 对patch维度做平均池化，得到每视图一个向量
        pooled_per_view = last_hidden.mean(dim=1)
        return self.projector(pooled_per_view)

    def encode(self, images: List[ImageItem]) -> Tuple[torch.Tensor, List[str], List[List[int]]]:
        img_tensors: List[torch.Tensor] = []
        uuids: List[str] = []
        valid_id = 0
        valid_ids: List[List[int]] = []
        image_aspect_ratio = getattr(self.image_processor, "image_aspect_ratio", None)
        image_grid_pinpoints = getattr(self.image_processor, "image_grid_pinpoints", None)
        for i, img in enumerate(images):
            if isinstance(img, ImageItem):
                uuids.append(img.uuid)
                image_data = read_shm(get_shm_name_data(img.uuid))
                image_data = Image.open(BytesIO(image_data)).convert("RGB")
                if image_aspect_ratio == "pad":
                    image_proc = expand2square(image_data, tuple(int(x * 255) for x in self.image_processor.image_mean))
                    t = self.image_processor.preprocess(image_proc, return_tensors="pt")["pixel_values"]
                elif image_aspect_ratio and (image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio):
                    t = process_anyres_image(image_data, self.image_processor, image_grid_pinpoints)
                    if isinstance(t, np.ndarray):
                        t = torch.from_numpy(t)
                else:
                    t = self.image_processor.preprocess(image_data, return_tensors="pt")["pixel_values"]

                if t.ndim == 5:
                    print(f"[debug] mineru2_visual reshape t.ndim: {t.ndim}, t.shape: {t.shape}")
                    t = t.view(-1, t.shape[-3], t.shape[-2], t.shape[-1])
                elif t.ndim == 3:
                    print(f"[debug] mineru2_visual unsqueeze t.ndim: {t.ndim}, t.shape: {t.shape}")
                    t = t.unsqueeze(0)
                # 在修改前记录 manager 分配的 token_num
                try:
                    print(f"[debug] mineru2_visual manager_token_num_before={img.token_num} uuid={img.uuid}")
                except Exception:
                    pass
                # 对齐实际 K 与期望 token_num
                expected_k = img.token_num if getattr(img, "token_num", None) is not None else None
                actual_k = t.shape[0]
                if expected_k is None or expected_k <= 0:
                    expected_k = actual_k
                    print(f"[debug] mineru2_visual expected_k_from_actual uuid={img.uuid} expected_k={expected_k}")
                if actual_k != expected_k:
                    if actual_k % expected_k == 0:
                        factor = actual_k // expected_k
                        print(
                            f"[debug] mineru2_visual down_aggregate uuid={img.uuid}"
                            f" actual_k={actual_k} expected_k={expected_k} factor={factor}"
                        )
                        t = t.view(expected_k, factor, t.shape[1], t.shape[2], t.shape[3]).mean(dim=1)
                    elif expected_k % actual_k == 0:
                        factor = expected_k // actual_k
                        print(
                            f"[debug] mineru2_visual up_repeat uuid={img.uuid}"
                            f" actual_k={actual_k} expected_k={expected_k} factor={factor}"
                        )
                        t = t.repeat_interleave(repeats=factor, dim=0)
                    else:
                        k = min(actual_k, expected_k)
                        print(
                            f"[debug] mineru2_visual fallback_slice uuid={img.uuid}"
                            f" actual_k={actual_k} expected_k={expected_k} k={k}"
                        )
                        if actual_k >= expected_k:
                            t = t[:expected_k]
                        else:
                            # pad by repeating last
                            pad = t[-1:].repeat(expected_k - actual_k, 1, 1, 1)
                            t = torch.cat([t, pad], dim=0)
                img_tensors.append(t)
                # 最终 K
                final_k = t.shape[0]
                img.token_num = final_k
                print(
                    f"[debug] mineru2_visual actual_k={actual_k} "
                    f"expected_k={expected_k} final_k={final_k} uuid={img.uuid}"
                )
            else:
                raise Exception("Unsupport input types: {} for {}".format(type(img), img))

            cur_num = (
                img_tensors[-1].shape[0]
                if isinstance(img_tensors[-1], torch.Tensor) and img_tensors[-1].dim() == 4
                else 1
            )
            valid_ids.append([valid_id, valid_id + cur_num])
            print(
                f"[debug] mineru2_visual valid_ids_append uuid={img.uuid}"
                f" range=({valid_id},{valid_id + cur_num}) cur_num={cur_num}"
            )
            valid_id += cur_num

        if len(img_tensors) <= 0:
            return None, [], []
        # 保证全部为4维后拼接
        img = torch.cat(img_tensors, dim=0)
        img = img.cuda()
        all_img_embeds = self.forward(img)
        print(f"[debug] mineru2_visual all_img_embeds.shape={tuple(all_img_embeds.shape)} " f"total_K={img.shape[0]}")

        return all_img_embeds, uuids, valid_ids
