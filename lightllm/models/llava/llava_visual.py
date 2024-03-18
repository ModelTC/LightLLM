import torch
import torch.nn.functional as F
import json
import os
from PIL import Image
from typing import List, Union


class LlavaVisionModel:

    def __init__(self):
        pass

    def load_model(self, weight_dir):
        config_file = os.path.join(weight_dir, "config.json")
        config = json.load(open(config_file))
        self.select_layer = config.get('mm_vision_select_layer', -2)
        self.select_feature = config.get('mm_vision_select_feature', 'patch')

        # load clip vision model by cfg['mm_vision_tower']:
        #   huggingface_name or path_of_clip_relative_to_llava_model_dir
        vision_path = config.get('mm_vision_tower', 'openai/clip-vit-large-patch14-336')
        if isinstance(vision_path, list):
            vision_path = vision_path[0]
        if vision_path.startswith("./"):
            vision_path = os.path.join(weight_dir, vision_path)

        from transformers import CLIPVisionModel, CLIPImageProcessor
        self.image_processor = CLIPImageProcessor.from_pretrained(vision_path)
        self.vision_tower = CLIPVisionModel.from_pretrained(vision_path).half()
        self.vision_tower.requires_grad_(False)
        self.device = torch.device('cpu')

        # load projector weights
        self.projector_weights = {}
        for f in os.listdir(weight_dir):
            if f.endswith(".bin"):
                d = torch.load(os.path.join(weight_dir, f), "cpu")
                for k, v in d.items():
                    if 'model.mm_projector' in k:
                        self.projector_weights[k] = v.half()

        assert 'model.mm_projector.0.weight' in self.projector_weights
        assert 'model.mm_projector.0.bias' in self.projector_weights
        assert 'model.mm_projector.2.weight' in self.projector_weights
        assert 'model.mm_projector.2.bias' in self.projector_weights

    def cuda(self):
        self.vision_tower = self.vision_tower.cuda()
        for k, v in self.projector_weights.items():
            self.projector_weights[k] = v.cuda()
        self.device = torch.device('cuda')
        return self

    # batch images infer
    def forward(self, x):
        x = x.half().to(device=self.device)

        x = self.vision_tower(x, output_hidden_states=True)
        x = x.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            x = x[:, 1:].contiguous()
        B, L, N = x.shape
        x = x.view(-1, N)

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

    def encode(self, image_items: List[Union[str, Image.Image]]):
        images = []
        for item in image_items:
            if isinstance(item, Image.Image):
                image = item
            elif item.startswith("http://") or item.startswith("https://"):
                image = Image.open(requests.get(item, stream=True).raw)
            else:
                image = Image.open(item)
            images.append(image.convert("RGB"))

        images = self.image_processor.preprocess(images, return_tensors='pt')['pixel_values']
        return self.forward(images)
