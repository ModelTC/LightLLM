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
                self.async_ret_handle_dict: Dict[str, tuple] = {}
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
    def exposed_encode(self, images: List[ImageItem], infer_uid: Optional[str] = None):
        images = obtain(images)
        all_img_embeds, uuids, valid_ids = self.forward(images)
        all_img_embeds = all_img_embeds.to(torch.device("cuda"))

        if not self.is_visual_only_mode:
            assert infer_uid is None
            self._not_visual_only_mode_handle(
                all_img_embeds=all_img_embeds, uuids=uuids, valid_ids=valid_ids, images=images
            )
        else:
            self._visual_only_mode_handle(
                all_img_embeds=all_img_embeds, uuids=uuids, valid_ids=valid_ids, images=images, infer_uid=infer_uid
            )
        return

    def _not_visual_only_mode_handle(self, all_img_embeds, uuids, valid_ids, images):
        if self.tp_rank_id == 0:
            ready_flags = obtain(self.cache_client.root.get_items_embed(uuids))
            ids_to_set = []
            for i, ready in enumerate(ready_flags):
                if ready:
                    continue
                uid = uuids[i]
                start, end = valid_ids[i]
                image = images[i]
                self.cpu_embed_cache_client.copy_vision_to_cache(
                    embed_tensor=all_img_embeds[start:end], start_index_in_cache=image.start_index_in_embed_cache
                )
                ids_to_set.append(uid)
            if ids_to_set:
                self.cache_client.root.set_items_embed(ids_to_set)
                torch.cuda.current_stream().synchronize()

    def _visual_only_mode_handle(self, all_img_embeds, uuids, valid_ids, images, infer_uid):
        if self.tp_rank_id == 0:
            all_img_embeds = all_img_embeds.detach().cpu()
            self.async_ret_handle_dict[infer_uid] = (all_img_embeds, valid_ids, images)

    def exposed_put_to_afs(self, infer_uid: str):
        assert self.tp_rank_id == 0
        ret = self.async_ret_handle_dict.pop(infer_uid, None)
        if ret is not None:
            all_img_embeds, valid_ids, images = ret
            for i in enumerate(len(images)):
                start, end = valid_ids[i]
                image = images[i]
                self.redis_afs_client.insert(image.md5, all_img_embeds[start:end])


class VisualModelRpcClient:
    def __init__(self, rpc_conn):
        self.rpc_conn: VisualModelRpcServer = rpc_conn

        def async_wrap(f):
            f = rpyc.async_(f)

            async def _func(*args, **kwargs):
                ans = f(*args, **kwargs)
                await asyncio.to_thread(ans.wait)
                # raise if exception
                return ans.value

            return _func

        self._init_model = async_wrap(self.rpc_conn.init_model)
        self._encode = async_wrap(self.rpc_conn.encode)
        self._put_to_afs = async_wrap(self.rpc_conn.put_to_afs)

        return

    async def init_model(self, kvargs):
        ans: rpyc.AsyncResult = self._init_model(kvargs)
        await ans
        return

    async def encode(self, images: List[ImageItem]):
        ans = self._encode(images)
        return await ans

    async def put_to_afs(self):
        ans = self._put_to_afs()
        return await ans


def _init_env(scoket_path: str, success_event: "mp.Event"):
    # 注册graceful 退出的处理
    graceful_registry(inspect.currentframe().f_code.co_name)

    import lightllm.utils.rpyc_fix_utils as _

    t = ThreadedServer(VisualModelRpcServer(), socket_path=scoket_path, protocol_config={"allow_pickle": True})
    success_event.set()
    t.start()
    return


async def start_model_process():
    socket_path = _generate_unix_socket_path()
    if os.path.exists(socket_path):
        os.remove(socket_path)

    success_event = mp.Event()
    proc = mp.Process(
        target=_init_env,
        args=(
            socket_path,
            success_event,
        ),
    )
    proc.start()
    await asyncio.to_thread(success_event.wait, timeout=40)

    if get_env_start_args().run_mode != "visual_only":
        conn = retry(max_attempts=20, wait_time=2)(unix_connect)(socket_path, config={"allow_pickle": True})

        assert proc.is_alive()
        return VisualModelRpcClient(conn.root)
    else:
        conn = retry(max_attempts=20, wait_time=2)(unix_connect)(socket_path, config={"allow_pickle": True})
        conn1 = retry(max_attempts=20, wait_time=2)(unix_connect)(socket_path, config={"allow_pickle": True})

        assert proc.is_alive()
        return VisualModelRpcClient(conn.root), VisualModelRpcClient(conn1.root)


def _generate_unix_socket_path() -> str:
    """Generate a random Unix socket path"""
    unique_id = uuid.uuid4().hex[:8]
    return f"/tmp/lightllm_model_infer_{unique_id}.sock"
