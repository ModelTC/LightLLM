import torch
import torch.nn.functional as F
import json
import os
from PIL import Image
from typing import List, Union
from safetensors import safe_open
from io import BytesIO
from lightllm.server.multimodal_params import MultimodalParams, ImageItem
from lightllm.server.embed_cache.utils import read_shm, get_shm_name_data
from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)


class LlavaVisionModel:
    def __init__(self):
        pass

    def load_model(self, weight_dir):
        config_file = os.path.join(weight_dir, "config.json")
        config = json.load(open(config_file))

        # for llava-v1.5-7b-hf model, should load config from transformers
        if "text_config" in config:
            self.load_hf_model(config, weight_dir)
        else:
            self.load_bin_model(config, weight_dir)

        self.vision_tower.requires_grad_(False)
        self.device = torch.device("cpu")

        assert "model.mm_projector.0.weight" in self.projector_weights
        assert "model.mm_projector.0.bias" in self.projector_weights
        assert "model.mm_projector.2.weight" in self.projector_weights
        assert "model.mm_projector.2.bias" in self.projector_weights

    def load_hf_model(self, config, weight_dir):
        from transformers import AutoConfig, AutoProcessor, LlavaForConditionalGeneration

        config = AutoConfig.from_pretrained(weight_dir, trust_remote_code=True)
        self.select_layer = config.vision_feature_layer
        self.select_feature = config.vision_feature_select_strategy
        processor = AutoProcessor.from_pretrained(weight_dir)
        self.image_processor = processor.image_processor
        model = LlavaForConditionalGeneration.from_pretrained(
            weight_dir,
            torch_dtype=torch.float16,
        )
        self.vision_tower = model.vision_tower
        model.multi_modal_projector = None
        model.language_model = None

        # load projector weights
        self.projector_weights = {}
        for f in os.listdir(weight_dir):
            if f.endswith(".safetensors"):
                d = safe_open(os.path.join(weight_dir, f), "pt", "cpu")
                for k in d.keys():
                    if "multi_modal_projector.linear_1" in k:
                        self.projector_weights[
                            k.replace("multi_modal_projector.linear_1", "model.mm_projector.0")
                        ] = d.get_tensor(k).half()
                    if "multi_modal_projector.linear_2" in k:
                        self.projector_weights[
                            k.replace("multi_modal_projector.linear_2", "model.mm_projector.2")
                        ] = d.get_tensor(k).half()

    def load_bin_model(self, config, weight_dir):
        self.select_layer = config.get("mm_vision_select_layer", -2)
        self.select_feature = config.get("mm_vision_select_feature", "patch")

        # load clip vision model by cfg['mm_vision_tower']:
        #   huggingface_name or path_of_clip_relative_to_llava_model_dir
        vision_path = config.get("mm_vision_tower", "openai/clip-vit-large-patch14-336")
        if isinstance(vision_path, list):
            vision_path = vision_path[0]
        if vision_path.startswith("./"):
            vision_path = os.path.join(weight_dir, vision_path)

        from transformers import CLIPVisionModel, CLIPImageProcessor

        self.image_processor = CLIPImageProcessor.from_pretrained(vision_path)
        self.vision_tower = CLIPVisionModel.from_pretrained(vision_path).half()

        # load projector weights
        self.projector_weights = {}
        for f in os.listdir(weight_dir):
            if f.endswith(".bin"):
                d = torch.load(os.path.join(weight_dir, f), "cpu")
                for k, v in d.items():
                    if "model.mm_projector" in k:
                        self.projector_weights[k] = v.half()

    def cuda(self):
        self.vision_tower = self.vision_tower.cuda()
        for k, v in self.projector_weights.items():
            self.projector_weights[k] = v.cuda()
        return self

    # batch images infer
    def forward(self, x):
        x = x.half().cuda()
        x = self.vision_tower(x, output_hidden_states=True)
        x = x.hidden_states[self.select_layer]
        if self.select_feature == "patch" or self.select_feature == "default":
            x = x[:, 1:].contiguous()
        B, L, N = x.shape
        x = x.view(-1, N).half()

        # mm_project
        x = F.linear(
            x,
            weight=self.projector_weights["model.mm_projector.0.weight"],
            bias=self.projector_weights["model.mm_projector.0.bias"],
        )
        x = F.gelu(x)
        x = F.linear(
            x,
            weight=self.projector_weights["model.mm_projector.2.weight"],
            bias=self.projector_weights["model.mm_projector.2.bias"],
        )
        x = x.view(B, L, -1)
        return x

    def encode(self, images: List[ImageItem]):
        img_tensors = []
        uuids = []
        valid_id = 0
        valid_ids = []

        for i, img in enumerate(images):
            if isinstance(img, ImageItem):
                uuids.append(img.uuid)
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
