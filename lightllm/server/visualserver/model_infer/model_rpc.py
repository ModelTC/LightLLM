import asyncio
import numpy as np
import rpyc
import torch
import socket
import inspect
from datetime import timedelta
from typing import Dict, List, Tuple
from transformers.configuration_utils import PretrainedConfig
from rpyc.utils.classic import obtain
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
from lightllm.utils.infer_utils import set_random_seed
from lightllm.utils.dist_utils import init_vision_distributed_env
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.server.embed_cache.embed_cache_client import CpuEmbedCacheClient


class VisualModelRpcServer(rpyc.Service):
    def exposed_init_model(self, kvargs):
        kvargs = obtain(kvargs)
        import torch
        import torch.distributed as dist

        self.vit_dp = kvargs["vit_dp"]
        self.vit_tp = kvargs["vit_tp"]
        self.dp_rank_id = kvargs["dp_rank_id"]
        self.tp_rank_id = kvargs["tp_rank_id"]
        self.cache_port = kvargs["cache_port"]
        weight_dir = kvargs["weight_dir"]
        self.vit_rank_id = kvargs["vit_rank_id"]
        self.cache_client = rpyc.connect("localhost", self.cache_port, config={"allow_pickle": True})
        self.cache_client._channel.stream.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.data_type = kvargs["data_type"]

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
            else:
                raise Exception(f"can not support {self.model_type} now")

            self.model.load_model(weight_dir)
            self.model = self.model.cuda()
            self.cpu_embed_cache_client = CpuEmbedCacheClient(create_meta_data=False, init_shm_data=True)
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

    # @calculate_time(show=False, min_cost_ms=300)
    def exposed_encode(self, images: List[ImageItem]):
        images = obtain(images)
        all_img_embeds, uuids, valid_ids = self.forward(images)
        all_img_embeds = all_img_embeds.to(torch.device("cuda"))

        if self.tp_rank_id == 0:
            ready_flags = obtain(self.cache_client.root.get_items_embed(uuids))
            ids_to_set = []
            for i, ready in enumerate(ready_flags):
                if ready:
                    continue
                uid = uuids[i]
                start, end = valid_ids[i]
                image = images[i]
                self.cpu_embed_cache_client.copy_to_cache(
                    embed_tensor=all_img_embeds[start:end], start_index_in_cache=image.start_index_in_embed_cache
                )
                ids_to_set.append(uid)
            if ids_to_set:
                self.cache_client.root.set_items_embed(ids_to_set)
                torch.cuda.current_stream().synchronize()
        return


class VisualModelRpcClient:
    def __init__(self, model_rpc, vit_tp, rpc_server_process=None):
        self.model: VisualModelRpcServer = model_rpc
        self.vit_tp = vit_tp
        self.rpc_server_process = rpc_server_process
        self.use_rpc = True
        if self.use_rpc:

            def async_wrap(f):
                f = rpyc.async_(f)

                async def _func(*args, **kwargs):
                    ans = f(*args, **kwargs)
                    await asyncio.to_thread(ans.wait)
                    # raise if exception
                    return ans.value

                return _func

            self._init_model = async_wrap(self.model.init_model)
            self._encode = async_wrap(self.model.encode)
        else:
            self._init_model = self.model.exposed_init_model
            self._encode = self.model.exposed_encode
        return

    async def init_model(self, kvargs):
        ans: rpyc.AsyncResult = self._init_model(kvargs)
        if self.use_rpc:
            await ans
            return
        else:
            return

    async def encode(self, images: List[ImageItem]):
        ans = self._encode(images)
        if self.use_rpc:
            return await ans
        else:
            return ans


def _init_env(port, device_id):
    # 注册graceful 退出的处理
    graceful_registry(inspect.currentframe().f_code.co_name)

    import lightllm.utils.rpyc_fix_utils as _

    t = ThreadedServer(VisualModelRpcServer(), port=port, protocol_config={"allow_pickle": True})
    t.start()
    return


async def start_model_process(port, vit_tp, device_id):
    import multiprocessing

    proc = multiprocessing.Process(
        target=_init_env,
        args=(
            port,
            device_id,
        ),
    )
    proc.start()
    await asyncio.sleep(2)
    repeat_count = 0
    while repeat_count < 20:
        try:
            con = rpyc.connect("localhost", port, config={"allow_pickle": True})
            con._channel.stream.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            break
        except BaseException:
            await asyncio.sleep(1)
        repeat_count += 1
    if repeat_count == 20:
        raise Exception("init rpc env error!")

    assert proc.is_alive()
    return VisualModelRpcClient(con.root, vit_tp, rpc_server_process=proc)
