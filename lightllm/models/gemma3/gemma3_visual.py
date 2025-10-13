import torch
import torch.nn.functional as F
import torch.nn as nn
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


class Gemma3VisionModel:
    def __init__(self):
        pass

    def load_model(self, weight_dir):
        config_file = os.path.join(weight_dir, "config.json")
        config = json.load(open(config_file))

        # for llava-v1.5-7b-hf model, should load config from transformers
        if "text_config" in config:
            self.load_hf_model(config, weight_dir)
        else:
            assert False, "only hf format model is supported for Gemma3"

        self.patches_per_image = int(config["vision_config"]["image_size"] // config["vision_config"]["patch_size"])
        self.tokens_per_side = int(config["mm_tokens_per_image"] ** 0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side
        self.avg_pool = nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.kernel_size)

        self.vision_tower.requires_grad_(False)
        self.device = torch.device("cpu")

        assert "model.mm_projector.linear" in self.projector_weights
        assert "model.mm_projector.norm" in self.projector_weights

    def load_hf_model(self, config, weight_dir):
        from transformers import AutoConfig, AutoProcessor, Gemma3ForConditionalGeneration

        # config = AutoConfig.from_pretrained(weight_dir, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(weight_dir)
        self.image_processor = processor.image_processor

        model = Gemma3ForConditionalGeneration.from_pretrained(
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
                    if "multi_modal_projector.mm_input_projection_weight" in k:
                        self.projector_weights[
                            k.replace("multi_modal_projector.mm_input_projection_weight", "model.mm_projector.linear")
                        ] = d.get_tensor(k).to(torch.bfloat16)
                    if "multi_modal_projector.mm_soft_emb_norm.weight" in k:
                        self.projector_weights[
                            k.replace("multi_modal_projector.mm_soft_emb_norm.weight", "model.mm_projector.norm")
                        ] = d.get_tensor(k).to(torch.bfloat16)

    def cuda(self):
        self.vision_tower = self.vision_tower.cuda()
        for k, v in self.projector_weights.items():
            self.projector_weights[k] = v.cuda()
        return self

    def gemma3_rms_norm(self, input, weight, eps: float = 1e-6):
        def _norm(x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

        output = _norm(input.float())
        # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + weight.float())
        return output.type_as(input)

    # batch images infer
    def forward(self, x):
        x = x.to(torch.bfloat16).cuda()
        x = self.vision_tower(x, output_hidden_states=True).last_hidden_state

        batch_size, _, seq_length = x.shape

        reshaped_vision_outputs = x.transpose(1, 2)
        reshaped_vision_outputs = reshaped_vision_outputs.reshape(
            batch_size, seq_length, self.patches_per_image, self.patches_per_image
        )
        reshaped_vision_outputs = reshaped_vision_outputs.contiguous()

        pooled_vision_outputs = self.avg_pool(reshaped_vision_outputs)
        pooled_vision_outputs = pooled_vision_outputs.flatten(2)
        pooled_vision_outputs = pooled_vision_outputs.transpose(1, 2)

        normed_vision_outputs = self.gemma3_rms_norm(
            pooled_vision_outputs.float(), self.projector_weights["model.mm_projector.norm"]
        ).to(torch.bfloat16)

        projected_vision_outputs = torch.matmul(
            normed_vision_outputs, self.projector_weights["model.mm_projector.linear"]
        )

        return projected_vision_outputs.type_as(x)

    def encode(self, images: List[ImageItem]):
        img_tensors = []
        uuids = []
        valid_id = 0
        valid_ids = []

        for i, img in enumerate(images):
            if isinstance(img, ImageItem):
                uuids.append(img.uuid)
                image_data = read_shm(get_shm_name_data(img.uuid))
                image_data = Image.open(BytesIO(image_data))
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
