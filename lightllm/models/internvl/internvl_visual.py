import os
import re
import json
import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Union
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from lightllm.server.multimodal_params import MultimodalParams, ImageItem
from lightllm.server.embed_cache.utils import read_shm, get_shm_name_data
from io import BytesIO
from lightllm.models.internvl.img_process import load_image
from lightllm.models.vit import get_load_image_func
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class InternVLVisionModel:
    def __init__(self):
        pass

    def load_model(self, weight_dir):
        assert torch.cuda.is_available()
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        self.config = json.load(open(os.path.join(weight_dir, "config.json")))
        # self.model = AutoModel.from_pretrained(
        #     weight_dir,
        #     torch_dtype=self.dtype,
        #     trust_remote_code=True,
        #     language_model="fake_language_model",
        # )
        from internvl_chat import InternVLChatModel, InternVLChatConfig

        cfg = InternVLChatConfig.from_pretrained(weight_dir)
        self.model = InternVLChatModel.from_pretrained(
            weight_dir, config=cfg, torch_dtype=self.dtype, language_model="fake_language_model"
        )
        self.model.eval().cuda()
        self.load_image_func = get_load_image_func(weight_dir)

    def cuda(self):
        return self

    def encode(self, images: List[ImageItem]):
        img_tensors = []
        valid_ids = []
        valid_id = 0
        uuids = []

        for i, img in enumerate(images):
            if isinstance(img, ImageItem):
                uuids.append(img.uuid)
                image_data = read_shm(get_shm_name_data(img.uuid))
                image_data = Image.open(BytesIO(image_data))
                t = self.load_image_func(image_data, max_num=img.extra_params["image_patch_max_num"])
                img_tensors.append(t)
            else:
                raise Exception("Unsupport input types: {} for {}".format(type(img), img))

            cur_num = img_tensors[-1].shape[0]
            valid_ids.append([valid_id, valid_id + cur_num])
            valid_id += cur_num

        if len(img_tensors) <= 0:
            return None

        imgs = torch.cat(img_tensors, dim=0)
        pixel_values = imgs.cuda().to(dtype=self.dtype)
        all_img_embeds = self.model.extract_feature(pixel_values)

        return all_img_embeds, uuids, valid_ids
