import os
import math
import json
from io import BytesIO
from typing import List, Optional

from torch import nn
import torch.utils.checkpoint

from PIL import Image

from transformers.activations import ACT2FN
from transformers import AutoModel
from safetensors import safe_open

from lightllm.models.qwen2_vl.qwen2_visual import Qwen2VisionTransformerPretrainedModel
from lightllm.server.embed_cache.utils import read_shm, get_shm_name_data
from lightllm.server.multimodal_params import ImageItem
from lightllm.models.qwen2_vl.vision_process import Qwen2VLImageProcessor, get_image


def add_split_tokens(image_features, image_newline_embed, image_new_embed):
    num_images, num_image_patches, embed_dim = image_features.shape
    num_height_patches, num_width_patches = int(math.sqrt(num_image_patches)), int(math.sqrt(num_image_patches))

    # add image_newline
    image_features = image_features.view(num_images, num_height_patches, num_width_patches, embed_dim)
    image_features = torch.cat(
        [image_features, image_newline_embed.expand((num_images, num_height_patches, 1, embed_dim))], dim=2
    )
    num_image_patches += num_height_patches
    image_features = image_features.view(num_images, num_image_patches, embed_dim)

    # add image_new
    image_features = torch.cat([image_features, image_new_embed.expand((num_images, 1, embed_dim))], dim=1)

    return image_features


class PixelShuffleMultiModalProjector(nn.Module):
    def __init__(
        self,
        image_newline_idx,
        image_new_idx,
        vit_hidden_size,
        llm_hidden_size,
        vision_feature_select_strategy,
        vision_feature_layer,
    ):
        super().__init__()
        self.downsample_ratio = 0.5

        self.mlp = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

        image_newline_idx = torch.tensor([image_newline_idx], dtype=torch.long)
        image_new_idx = torch.tensor([image_new_idx], dtype=torch.long)
        self.register_buffer("image_newline_idx", image_newline_idx, persistent=False)
        self.register_buffer("image_new_idx", image_new_idx, persistent=False)

        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer

    def forward(self, image_features, input_embeddings):
        selected_image_feature = image_features[self.vision_feature_layer]

        if self.vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif self.vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        else:
            raise ValueError(f"Unexpected select feature strategy: {self.vision_feature_select_strategy}")

        image_features = self.pixel_shuffle(selected_image_feature)
        hidden_states = self.mlp(image_features)

        image_newline_embed = input_embeddings(self.image_newline_idx).squeeze()
        image_new_embed = input_embeddings(self.image_new_idx).squeeze()
        hidden_states = add_split_tokens(hidden_states, image_newline_embed, image_new_embed)

        return hidden_states

    def pixel_shuffle(self, x, scale_factor=0.5):
        if scale_factor == 1:
            return x
        n, wh, c = x.shape
        h, w = int(math.sqrt(wh)), int(math.sqrt(wh))
        x = x.view(n, h, w, c)

        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor), int(c / (scale_factor * scale_factor)))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.shape[0], -1, x.shape[-1])
        return x


class LlavaMultiModalProjector(nn.Module):
    def __init__(
        self,
        image_newline_idx,
        image_new_idx,
        vit_hidden_size,
        llm_hidden_size,
        vision_feature_select_strategy,
        vision_feature_layer,
        projector_hidden_act,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(vit_hidden_size, llm_hidden_size, bias=True)
        self.act = ACT2FN[projector_hidden_act]
        self.linear_2 = nn.Linear(llm_hidden_size, llm_hidden_size, bias=True)

        image_newline_idx = torch.tensor([image_newline_idx], dtype=torch.long)
        image_new_idx = torch.tensor([image_new_idx], dtype=torch.long)
        self.register_buffer("image_newline_idx", image_newline_idx, persistent=False)
        self.register_buffer("image_new_idx", image_new_idx, persistent=False)

        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer

    def forward(self, image_features, input_embeddings):

        selected_image_feature = image_features[self.vision_feature_layer]

        if self.vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif self.vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        else:
            raise ValueError(f"Unexpected select feature strategy: {self.vision_feature_select_strategy}")

        hidden_states = self.linear_1(selected_image_feature)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)

        image_newline_embed = input_embeddings(self.image_newline_idx).squeeze()
        image_new_embed = input_embeddings(self.image_new_idx).squeeze()
        hidden_states = add_split_tokens(hidden_states, image_newline_embed, image_new_embed)
        return hidden_states


class TarsierVisionTransformerPretrainedModel(nn.Module):
    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=32000,
        projector_hidden_act="gelu",
        vision_feature_select_strategy="default",
        vision_feature_layer=-2,
        image_newline_idx=32002,
        image_new_idx=32003,
        projection_head="MLP",
        **kwargs,
    ):
        super().__init__()
        self.vision_tower = Qwen2VisionTransformerPretrainedModel(**vision_config)

        if projection_head == "Pixel_Shuffle":
            self.multi_modal_projector = PixelShuffleMultiModalProjector(
                image_newline_idx,
                image_new_idx,
                vision_config.hidden_size,
                text_config.hidden_size,
                vision_feature_select_strategy,
                vision_feature_layer,
            )
        elif projection_head == "MLP":
            self.multi_modal_projector = LlavaMultiModalProjector(
                image_newline_idx,
                image_new_idx,
                vision_config.hidden_size,
                text_config.hidden_size,
                vision_feature_select_strategy,
                vision_feature_layer,
                projector_hidden_act,
            )
        elif projection_head == "auto_map":
            raise Exception("Unsupport projection_head auto_map")
        elif projection_head is None:
            self.multi_modal_projector = lambda x, *args, **kwargs: x
        self.llm_model_type = text_config["model_type"]

        self.image_token_index = image_token_index
        self.merge_size = 1

    def forward(
        self,
        pixel_values: torch.Tensor = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:

        image_features = None
        if pixel_values is not None:  # training / first step in generation
            if self.llm_model_type == "qwen2_vl":
                pixel_values = pixel_values.type(self.vision_tower.get_dtype())
                image_features = self.vision_tower(pixel_values, image_grid_thw)
            else:
                image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
                image_features = self.multi_modal_projector(
                    image_outputs.hidden_states,
                    self.get_input_embeddings(),
                )

        return image_features

    def load_model(self, weight_dir):
        processor_config_path = os.path.join(weight_dir, "preprocessor_config.json")
        with open(processor_config_path, "r") as f:
            processor_config_dict = json.load(f)
        self.processor = Qwen2VLImageProcessor(**processor_config_dict)

        bin_weight_files = [file_ for file_ in os.listdir(weight_dir) if file_.endswith(".bin")]
        if bin_weight_files:
            weight_dict = {}
            for file_ in bin_weight_files:
                f = torch.load(os.path.join(weight_dir, file_), "cpu")
                for k, v in f.items():
                    if "vision_tower" in k:
                        weight_dict[k] = v
        else:
            hf_weight_files = [file_ for file_ in os.listdir(weight_dir) if file_.endswith(".safetensors")]
            weight_dict = {}
            for file_ in hf_weight_files:
                f = safe_open(os.path.join(weight_dir, file_), "pt", "cpu")
                for k in f.keys():
                    if "vision_tower" in k:
                        weight_dict[k] = f.get_tensor(k)

        self.load_state_dict(weight_dict)

    def encode(self, images: List[ImageItem]):
        img_tensors = []
        valid_ids = []
        valid_id = 0
        img_grids = []
        uuids = []

        for i, img in enumerate(images):
            if isinstance(img, ImageItem):
                uuids.append(img.uuid)
                image_data = read_shm(get_shm_name_data(img.uuid))
                image_data = Image.open(BytesIO(image_data))
                image_data = get_image(image_data)
                image_inputs = self.processor.preprocess(images=image_data, return_tensors="pt")
                pixel_values = image_inputs["pixel_values"].to(dtype=torch.bfloat16)
                image_grid_thw = image_inputs["image_grid_thw"]
                img_tensors.append(pixel_values)
                img_grids.append(image_grid_thw)
            else:
                raise Exception("Unsupport input types: {} for {}".format(type(img), img))

            # must devide merge_length
            cur_num = img_tensors[-1].shape[0] // (self.merge_size ** 2)

            valid_ids.append([valid_id, valid_id + cur_num])
            valid_id += cur_num

        if len(img_tensors) <= 0:
            return None

        imgs = torch.cat(img_tensors, dim=0)
        grid_thw = torch.cat(img_grids, dim=0)

        pixel_values = imgs.cuda()
        image_grid_thw = grid_thw.cuda()

        all_img_embeds = self.forward(pixel_values=pixel_values, image_grid_thw=image_grid_thw)

        return all_img_embeds, uuids, valid_ids
