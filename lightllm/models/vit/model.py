import os
import json
import torch
from lightllm.models.vit.layer_infer.pre_layer_infer import ViTPreLayerInfer
from lightllm.models.vit.layer_infer.post_layer_infer import ViTPostLayerInfer
from lightllm.models.vit.layer_infer.transformer_layer_infer import ViTTransformerLayerInfer
from lightllm.models.vit.layer_weights.pre_and_post_layer_weight import ViTPreAndPostLayerWeight
from lightllm.models.vit.layer_weights.transformer_layer_weight import ViTTransformerLayerWeight
from lightllm.models.vit.layer_weights.hf_load_utils import load_hf_weights
from lightllm.server.multimodal_params import MultimodalParams, ImageItem
from lightllm.common.build_utils import repair_config
from lightllm.utils.log_utils import init_logger
from lightllm.models.vit import get_load_image_func
import torchvision.transforms as T
from lightllm.server.embed_cache.utils import read_shm, get_shm_name_data
from PIL import Image
from typing import List, Union, final
from io import BytesIO
from rpyc.utils.classic import obtain
from lightllm.common.quantization import Quantcfg
from lightllm.utils.dist_utils import get_dp_world_size
from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager


logger = init_logger(__name__)


class VisionTransformer:

    # weight class
    pre_and_post_weight_class = ViTPreAndPostLayerWeight
    transformer_weight_class = ViTTransformerLayerWeight

    # infer class
    pre_layer_infer_class = ViTPreLayerInfer
    transformer_layer_infer_class = ViTTransformerLayerInfer
    post_layer_infer_class = ViTPostLayerInfer

    def __init__(self, kvargs):
        self.tp_world_size_ = get_dp_world_size()
        self.weight_dir_ = kvargs["weight_dir"]
        self.load_way = kvargs.get("load_way", "HF")
        self.weight_dict = kvargs.get("weight_dict", None)
        self.data_type = kvargs.get("data_type", "float16")
        self.quant_type = kvargs.get("quant_type", None)
        self.quant_cfg_path = kvargs.get("quant_cfg", None)
        self.load_image_func = get_load_image_func(self.weight_dir_)
        self.max_batch_size = kvargs.get("max_batch_size", 1)

        self._init_datatype()
        self._init_config()
        self._padding_hidden_size()
        self._init_quant()
        self._init_weights()
        self._init_infer_layer()
        self._check_max_len_infer()
        return

    @final
    @torch.no_grad()
    def _check_max_len_infer(self):
        disable_check_max_len_infer = os.getenv("DISABLE_CHECK_MAX_LEN_INFER", None) is not None
        if disable_check_max_len_infer:
            return

        try:
            dummy_images = torch.randn(
                (self.MAX_PATH_NUM * self.max_batch_size, 3, self.IMAGE_H, self.IMAGE_W), dtype=self.data_type
            ).cuda()
            all_img_embeds = self.forward(dummy_images)
            del all_img_embeds
            logger.info(f"vit check max_len {self.max_batch_size} infer ok")
        except (RuntimeError, torch.OutOfMemoryError) as e:
            logger.exception(str(e))
            exception_str = (
                "Vit check max len infer fail, you can try:" "1.Set the --visual_infer_batch_size to a smaller value."
            )
            logger.error(exception_str)
            raise Exception(exception_str)
        return

    def _init_config(self):
        with open(os.path.join(self.weight_dir_, "config.json"), "r") as json_file:
            self.config = json.load(json_file)
            self.select_layer = self.config["select_layer"]
            self.config["vision_config"]["llm_hidden_size"] = self.config["llm_config"]["hidden_size"]
            self.config["vision_config"]["downsample_ratio"] = self.config["downsample_ratio"]
            self.config = self.config["vision_config"]
        repair_config(self.config, same_names=["num_attention_heads", "n_head"])
        repair_config(self.config, same_names=["hidden_size", "n_embd", "n_embed"])
        repair_config(self.config, same_names=["num_hidden_layers", "n_layer"])
        self.layers_num = self.config["num_hidden_layers"]

        # infer info
        self.IMAGE_H = int(os.getenv("IMAGE_H", 448))
        self.IMAGE_W = int(os.getenv("IMAGE_W", 448))
        self.MAX_PATH_NUM = os.getenv("MAX_PATH_NUM", 13)
        return

    def _padding_hidden_size(self):
        self.config["padding_hidden_size"] = 0
        self.config["padding_head_num"] = self.config["num_attention_heads"]

        head_dim = self.config["hidden_size"] // self.config["num_attention_heads"]
        if self.config["num_attention_heads"] % self.tp_world_size_ != 0:
            padding_head_num = (
                self.config["num_attention_heads"] + self.tp_world_size_ - 1
            ) // self.tp_world_size_ * self.tp_world_size_ - self.config["num_attention_heads"]
            self.config["padding_hidden_size"] = padding_head_num * head_dim
            self.config["padding_head_num"] += padding_head_num
        return

    def _init_weights(self):
        self.pre_post_weight = self.pre_and_post_weight_class(self.data_type, network_config=self.config)
        self.trans_layers_weight = [
            self.transformer_weight_class(
                i,
                self.data_type,
                network_config=self.config,
                quant_cfg=self.quant_cfg,
            )
            for i in range(self.config["num_hidden_layers"])
        ]
        load_hf_weights(
            self.data_type,
            weight_dir=self.weight_dir_,
            pre_post_layer=self.pre_post_weight,
            transformer_layer_list=self.trans_layers_weight,
            weight_dict=self.weight_dict,
        )
        self.pre_post_weight.verify_load()
        [weight.verify_load() for weight in self.trans_layers_weight]
        return

    def _init_quant(self):
        self.quant_cfg = Quantcfg(self.config, self.quant_type, self.quant_cfg_path)
        logger.info(f"Initial quantization. " f"The default quantization method is {self.quant_cfg.quant_type}")

    def _init_infer_layer(self):
        self.pre_infer = self.pre_layer_infer_class(network_config=self.config)
        self.post_infer = self.post_layer_infer_class(network_config=self.config)
        self.layers_infer = [
            self.transformer_layer_infer_class(i, network_config=self.config)
            for i in range(self.config["num_hidden_layers"])
        ]
        return

    def _init_datatype(self):
        if self.data_type in ["fp16", "float16"]:
            self.data_type = torch.float16
        elif self.data_type in ["bf16", "bfloat16"]:
            self.data_type = torch.bfloat16
        elif self.data_type in ["fp32", "float32"]:
            self.data_type = torch.float32
        else:
            raise ValueError(f"Unsupport datatype {self.data_type}!")

    @torch.no_grad()
    def forward(self, pixel_values):
        g_cache_manager.cache_env_in()
        input_embs = self.pre_infer.forward(pixel_values, self.pre_post_weight)
        for i in range(self.layers_num + self.select_layer + 1):
            input_embs = self.layers_infer[i].forward(input_embs, self.trans_layers_weight[i])
        input_embs = self.post_infer.forward(input_embs[:, 1:, :], self.pre_post_weight)
        g_cache_manager.cache_env_out()
        return input_embs

    @torch.no_grad()
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

            cur_num = img.token_num
            valid_ids.append([valid_id, valid_id + cur_num])
            valid_id += cur_num

        if len(img_tensors) <= 0:
            return None

        imgs = torch.cat(img_tensors, dim=0)
        pixel_values = imgs.cuda().to(dtype=self.data_type)
        all_img_embeds = self.forward(pixel_values)
        return all_img_embeds.view(-1, all_img_embeds.shape[-1]), uuids, valid_ids

    def cuda(self):
        return self

    def load_model(self, weight_dir):
        pass
