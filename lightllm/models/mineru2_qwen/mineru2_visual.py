import re

from typing import List, Tuple
from io import BytesIO
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import (
    CLIPVisionModel,
    SiglipVisionConfig,
    SiglipVisionModel,
)

from .configuration_mineru2 import Mineru2QwenConfig
from .image_processing_mineru2 import (
    Mineru2ImageProcessor,
    expand2square,
    process_anyres_image,
    get_anyres_image_grid_shape,
)

from lightllm.server.multimodal_params import ImageItem
from lightllm.server.embed_cache.utils import read_shm, get_shm_name_data
from lightllm.utils.log_utils import init_logger

import os
from safetensors import safe_open

logger = init_logger(__name__)


def build_vision_tower(weight_dir: str, config: Mineru2QwenConfig):
    vision_tower = getattr(config, "mm_vision_tower", getattr(config, "vision_tower", ""))
    model_path = os.path.join(weight_dir, vision_tower)
    if not os.path.exists(model_path):
        model_path = vision_tower

    if "clip" in vision_tower.lower():
        return CLIPVisionModel.from_pretrained(model_path)
    elif "siglip" in vision_tower.lower():
        cfg = SiglipVisionConfig.from_pretrained(model_path)
        cfg.num_hidden_layers = max(0, cfg.num_hidden_layers - 1)
        cfg.vision_use_head = False
        model = SiglipVisionModel.from_pretrained(model_path, config=cfg, ignore_mismatched_sizes=True)
        return model
    else:
        raise ValueError(f"Unknown vision tower: {vision_tower}")


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

    def _load_projector_weights(self, weight_dir: str):
        def assign_linear(linear: nn.Linear, w: torch.Tensor = None, b: torch.Tensor = None):
            if w is not None:
                linear.weight.data.copy_(w.to(dtype=linear.weight.dtype))
            if b is not None and linear.bias is not None:
                linear.bias.data.copy_(b.to(dtype=linear.bias.dtype))

        projector_weight_path = os.path.join(weight_dir, "model.safetensors")

        if isinstance(self.projector, nn.Linear):
            linear_modules = [self.projector]
        elif isinstance(self.projector, nn.Sequential):
            linear_modules = [m for m in self.projector if isinstance(m, nn.Linear)]
        else:
            raise RuntimeError(f"Unsupported projector type: {type(self.projector)}")

        try:
            with safe_open(projector_weight_path, framework="pt", device="cpu") as sf:
                sd = {k: sf.get_tensor(k) for k in sf.keys()}
        except Exception as e:
            raise RuntimeError(f"Failed to read projector weights: {projector_weight_path} due to {e}")

        # load projector weights
        layer_key_map = [
            (
                0,
                ("model.mm_projector.0.weight", "multi_modal_projector.linear_1.weight"),
                ("model.mm_projector.0.bias", "multi_modal_projector.linear_1.bias"),
            ),
            (
                1,
                ("model.mm_projector.2.weight", "multi_modal_projector.linear_2.weight"),
                ("model.mm_projector.2.bias", "multi_modal_projector.linear_2.bias"),
            ),
        ]
        for idx, w_keys, b_keys in layer_key_map:
            if idx >= len(linear_modules):
                continue
            w = next((sd[k] for k in w_keys if k in sd), None)
            b = next((sd[k] for k in b_keys if k in sd), None)
            if w is not None:
                assign_linear(linear_modules[idx], w, b)

        # load vision tower weights
        vt_prefix = "model.vision_tower.vision_tower."
        vt_sd = {k[len(vt_prefix) :]: v for k, v in sd.items() if k.startswith(vt_prefix)}
        if not vt_sd:
            logger.warning("vision_tower weights not found in checkpoint or skipped; keep pretrained weights")
            return

        try:
            self.vision_tower.load_state_dict(vt_sd, strict=False)
        except Exception as e:
            logger.warning(f"vision_tower load_state_dict failed (strict=False): {e}")

    def load_model(self, weight_dir):
        vision_config = Mineru2QwenConfig.from_pretrained(weight_dir)

        self.vision_tower = build_vision_tower(weight_dir, vision_config)
        self.vision_tower.eval()
        self.vision_tower.requires_grad_(False)
        self.projector = build_vision_projector(vision_config)
        self._load_projector_weights(weight_dir)
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
        hiddens = vision_out.hidden_states

        # 对齐ref的“减一层”语义：优先使用倒数第二层；若不可用则回退最后一层
        try:
            chosen_idx = -2 if isinstance(hiddens, (list, tuple)) and len(hiddens) >= 2 else -1
            feat = hiddens[chosen_idx]
        except Exception:
            feat = hiddens[-2] if isinstance(hiddens, (list, tuple)) and len(hiddens) >= 2 else hiddens[-1]
        # 切回 patch 序列特征：去除 CLS（若存在），按序列过 projector，再展平为 (views*patch, hidden)
        patch_side = self.vision_tower.config.image_size // self.vision_tower.config.patch_size
        patch_len = patch_side * patch_side
        if feat.shape[1] == patch_len + 1:
            feat = feat[:, 1:, :]
        proj_seq = self.projector(feat)

        proj = proj_seq.reshape(-1, proj_seq.shape[-1])
        return proj

    def encode(self, images: List[ImageItem]) -> Tuple[torch.Tensor, List[str], List[List[int]]]:
        img_tensors: List[torch.Tensor] = []
        uuids: List[str] = []
        valid_id = 0
        valid_ids: List[List[int]] = []
        image_aspect_ratio = getattr(self.image_processor, "image_aspect_ratio", None)
        image_grid_pinpoints = getattr(self.image_processor, "image_grid_pinpoints", None)
        # 每视图 patch_len
        patch_side = self.vision_tower.config.image_size // self.vision_tower.config.patch_size
        patch_len = patch_side * patch_side

        for i, img in enumerate(images):
            if isinstance(img, ImageItem):
                uuids.append(img.uuid)
                image_data = read_shm(get_shm_name_data(img.uuid))
                image_data = Image.open(BytesIO(image_data)).convert("RGB")
                # 多图/视频强制 pad，单图才允许 anyres
                force_pad = len(images) > 1
                if image_aspect_ratio == "pad" or force_pad:
                    image_proc = expand2square(image_data, tuple(int(x * 255) for x in self.image_processor.image_mean))
                    t = self.image_processor.preprocess(image_proc, return_tensors="pt")["pixel_values"]
                elif image_aspect_ratio and (image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio):
                    t = process_anyres_image(image_data, self.image_processor, image_grid_pinpoints)
                    if isinstance(t, np.ndarray):
                        t = torch.from_numpy(t)
                else:
                    t = self.image_processor.preprocess(image_data, return_tensors="pt")["pixel_values"]

                if t.ndim == 5:
                    t = t.view(-1, t.shape[-3], t.shape[-2], t.shape[-1])
                elif t.ndim == 3:
                    t = t.unsqueeze(0)

                # 对齐实际视图数 K 与期望视图数（anyres: Nx*Ny+1；否则：1）
                actual_k = t.shape[0]
                if (
                    image_aspect_ratio and (image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio)
                ) and not force_pad:
                    crop_size = self.image_processor.crop_size["height"]
                    grid_w, grid_h = get_anyres_image_grid_shape(
                        (img.image_w, img.image_h), image_grid_pinpoints, crop_size
                    )
                    expected_views = int(grid_w * grid_h + 1)
                else:
                    expected_views = 1
                if actual_k != expected_views:
                    if actual_k % expected_views == 0:
                        factor = actual_k // expected_views
                        t = t.view(expected_views, factor, t.shape[1], t.shape[2], t.shape[3]).mean(dim=1)
                    elif expected_views % actual_k == 0:
                        factor = expected_views // actual_k
                        t = t.repeat_interleave(repeats=factor, dim=0)
                    else:
                        if actual_k >= expected_views:
                            t = t[:expected_views]
                        else:
                            # pad by repeating last
                            pad = t[-1:].repeat(expected_views - actual_k, 1, 1, 1)
                            t = torch.cat([t, pad], dim=0)
                img_tensors.append(t)
            else:
                raise Exception("Unsupport input types: {} for {}".format(type(img), img))

        if len(img_tensors) <= 0:
            return None, [], []
        # 保证全部为4维后拼接
        img = torch.cat(img_tensors, dim=0)
        img = img.cuda()
        # 提取所有视图的 patch 序列嵌入（views * patch_len, hidden）
        all_img_embeds = self.forward(img)

        # 将每张图的视图嵌入进行 spatial+unpad(+anyres_max) 重组，并追加换行列
        new_embeds: List[torch.Tensor] = []
        cur = 0
        for i, img in enumerate(images):
            # 计算本图视图数
            t = img_tensors[i]
            K = t.shape[0]
            # 取出本图的所有 view 的 patch 序列嵌入
            tokens_len = K * patch_len
            cur_views_embeds = all_img_embeds[cur : cur + tokens_len]
            cur += tokens_len

            # 非 anyres 或多图/视频强制 pad：直接使用展平序列（K 通常为 1）
            force_pad = len(images) > 1
            aspect = getattr(self.image_processor, "image_aspect_ratio", None)
            if not aspect or ("anyres" not in str(aspect)) or force_pad or K <= 1:
                seq = cur_views_embeds
                new_embeds.append(seq)
                # 记录区间
                valid_ids.append([valid_id, valid_id + seq.shape[0]])
                valid_id += seq.shape[0]
                continue

            # anyres 单图路径：
            # 切分 base 视图与其余视图
            base_feature = cur_views_embeds[:patch_len]
            rest = cur_views_embeds[patch_len:]
            # (K-1, patch_len, hidden)
            hidden = rest.shape[-1]
            rest = rest.view(K - 1, patch_len, hidden)

            # 计算 Nx, Ny
            crop_size = self.image_processor.crop_size["height"]
            grid_w, grid_h = get_anyres_image_grid_shape((img.image_w, img.image_h), image_grid_pinpoints, crop_size)
            # (Ny, Nx, patch_side, patch_side, hidden)
            rest = rest.view(grid_w * grid_h, patch_side, patch_side, hidden)
            rest = rest.view(grid_h, grid_w, patch_side, patch_side, hidden)
            # (hidden, Ny, patch_side, Nx, patch_side) -> (hidden, H, W)
            rest = rest.permute(4, 0, 2, 1, 3).contiguous()
            H = grid_h * patch_side
            W = grid_w * patch_side
            rest = rest.view(hidden, H, W)

            # anyres_max 下采样
            m = re.search(r"anyres_max_(\d+)", str(aspect))
            if m is not None:
                max_num_patches = int(m.group(1))
                times = (H * W) / (max_num_patches * patch_len)
                if times > 1.1:
                    scale = (int(H // (times ** 0.5)), int(W // (times ** 0.5)))
                    rest = F.interpolate(rest.unsqueeze(0), size=scale, mode="bilinear", align_corners=False)[0]
                    H, W = rest.shape[1], rest.shape[2]

            # 追加换行列（列数+1），换行列取 0 向量占位
            newline_col = torch.zeros((hidden, H, 1), device=rest.device, dtype=rest.dtype)
            rest = torch.cat([rest, newline_col], dim=2)  # (hidden, H, W+1)
            # 展平成 (H*(W+1), hidden)
            rest = rest.flatten(1, 2).transpose(0, 1).contiguous()

            # 拼接 base + 其余
            seq = torch.cat([base_feature, rest], dim=0)
            new_embeds.append(seq)

            # 记录区间
            valid_ids.append([valid_id, valid_id + seq.shape[0]])
            valid_id += seq.shape[0]

        # 拼接所有图的重组后嵌入
        all_new = torch.cat(new_embeds, dim=0)
        return all_new, uuids, valid_ids
