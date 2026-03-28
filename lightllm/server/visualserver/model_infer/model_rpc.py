import asyncio
import numpy as np
import rpyc
import torch
import socket
import inspect
import uuid
import os
import torch.multiprocessing as mp
import collections
from datetime import timedelta
from typing import Dict, List, Tuple, Deque, Optional
from transformers.configuration_utils import PretrainedConfig
from lightllm.utils.retry_utils import retry
from rpyc.utils.classic import obtain, unix_connect
from rpyc.utils.server import ThreadedServer
from lightllm.models.qwen_vl.qwen_visual import QWenVisionTransformer
from lightllm.models.llava.llava_visual import LlavaVisionModel
from lightllm.models.internvl.internvl_visual import InternVLVisionModel
from lightllm.models.gemma3.gemma3_visual import Gemma3VisionModel
from lightllm.models.vit.model import VisionTransformer
from lightllm.server.multimodal_params import MultimodalParams, ImageItem
from lightllm.models.qwen2_vl.qwen2_visual import Qwen2VisionTransformerPretrainedModel
from lightllm.models.qwen2_5_vl.qwen2_5_visual import Qwen2_5_VisionTransformerPretrainedModel
from lightllm.models.qwen3_vl.qwen3_visual import Qwen3VisionTransformerPretrainedModel
from lightllm.models.tarsier2.tarsier2_visual import TarsierVisionTransformerPretrainedModel
from lightllm.models.qwen3_omni_moe_thinker.qwen3_omni_visual import Qwen3OmniMoeVisionTransformerPretrainedModel
from lightllm.utils.infer_utils import set_random_seed
from lightllm.utils.dist_utils import init_vision_distributed_env
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.server.embed_cache.embed_cache_client import CpuEmbedCacheClient
from lightllm.server.visualserver import set_vit_att_backend
from lightllm.server.embed_cache.afs_utils import SepEmbedHandler


class VisualModelRpcServer(rpyc.Service):
    def exposed_init_model(self, kvargs):
        kvargs = obtain(kvargs)

        # kvargs = {
        #     "weight_dir": self.model_weightdir,
        #     "device_id": device_id,
        #     "vit_tp": self.vit_tp,
        #     "cache_port": self.args.cache_port,
        #     "tp_rank_id": tp_rank_id,
        #     "dp_rank_id": dp_rank_id,
        #     "data_type": self.args.data_type,
        #     "visual_nccl_port": self.args.visual_nccl_ports[dp_rank_id],
        #     "quant_type": self.args.vit_quant_type,
        #     "quant_cfg": self.args.vit_quant_cfg,
        #     "max_batch_size": min(self.infer_batch_size // self.vit_dp, 1),
        #     "vit_attn_backend": self.vit_attn_backend,
        # }

        weight_dir = kvargs["weight_dir"]
        self.vit_tp = kvargs["vit_tp"]
        self.dp_rank_id = kvargs["dp_rank_id"]
        self.tp_rank_id = kvargs["tp_rank_id"]
        self.cache_port = kvargs["cache_port"]
        self.vit_rank_id = kvargs["vit_rank_id"]
        self.is_visual_only_mode = get_env_start_args().run_mode == "visual_only"
        self.data_type = kvargs["data_type"]
        self.vit_attn_backend = kvargs["vit_attn_backend"]
        set_vit_att_backend(self.vit_attn_backend)
        init_vision_distributed_env(kvargs)
        model_cfg, _ = PretrainedConfig.get_config_dict(weight_dir)

        try:
            kvargs = {
                "weight_dir": weight_dir,
                "data_type": self.data_type,
                "quant_type": kvargs["quant_type"],
                "quant_cfg": kvargs["quant_cfg"],
                "max_batch_size": kvargs["max_batch_size"],
            }
            self.model_type = model_cfg["model_type"]
            if self.model_type == "qwen":
                self.model = QWenVisionTransformer(**model_cfg["visual"]).eval().bfloat16()
            elif self.model_type == "qwen2_vl":
                self.model = (
                    Qwen2VisionTransformerPretrainedModel(kvargs, **model_cfg["vision_config"]).eval().bfloat16()
                )
            elif self.model_type == "qwen2_5_vl":
                self.model = (
                    Qwen2_5_VisionTransformerPretrainedModel(kvargs, **model_cfg["vision_config"]).eval().bfloat16()
                )
            elif self.model_type in ["qwen3_vl", "qwen3_vl_moe"]:
                self.model = (
                    Qwen3VisionTransformerPretrainedModel(kvargs, **model_cfg["vision_config"]).eval().bfloat16()
                )
            elif model_cfg["architectures"][0] == "TarsierForConditionalGeneration":
                self.model = TarsierVisionTransformerPretrainedModel(**model_cfg).eval().bfloat16()
            elif self.model_type == "llava":
                self.model = LlavaVisionModel()
            elif self.model_type == "internvl_chat":
                self.model = VisionTransformer(kvargs)
                # self.model = InternVLVisionModel()
            elif self.model_type == "gemma3":
                self.model = Gemma3VisionModel()
            elif (
                model_cfg.get("thinker_config", {}).get("vision_config", {}).get("model_type")
                == "qwen3_omni_moe_vision_encoder"
            ):
                self.model = (
                    Qwen3OmniMoeVisionTransformerPretrainedModel(kvargs, **model_cfg["thinker_config"]["vision_config"])
                    .eval()
                    .bfloat16()
                )
            else:
                raise Exception(f"can not support {self.model_type} now")

            self.model.load_model(weight_dir)
            self.model = self.model.cuda()
            if not self.is_visual_only_mode:
                # 独立部署vit模式下，不需要连接 cache_client
                self.cache_client = rpyc.connect("localhost", self.cache_port, config={"allow_pickle": True})
                self.cache_client._channel.stream.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.cpu_embed_cache_client = CpuEmbedCacheClient(create_meta_data=False, init_shm_data=False)
            else:
                args = get_env_start_args()
                assert args.visual_dp == 1
                self.redis_afs_client = SepEmbedHandler(
                    afs_embed_dir=args.afs_embed_dir,
                    redis_host=args.config_server_host,
                    redis_port=args.config_server_vit_redis_port,
                    capacity=args.afs_embed_capacity,
                )
        except Exception as e:
            print("#" * 16)
            print("load model error:", str(e), e, type(e))
            import traceback

            traceback.print_exc()
            raise e

        set_random_seed(2147483647)
        return

    # @calculate_time(show=True, min_cost_ms=150)
    @torch.no_grad()
    def forward(self, images: List[ImageItem]):
        return self.model.encode(images)
