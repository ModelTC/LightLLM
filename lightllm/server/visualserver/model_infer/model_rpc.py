import rpyc
import torch
import socket
import torch.multiprocessing as mp
import queue
import threading
import time
import torch.distributed as dist
from typing import Dict, List, Tuple, Deque, Optional
from transformers.configuration_utils import PretrainedConfig
from rpyc.utils.classic import obtain
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
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.server.embed_cache.embed_cache_client import CpuEmbedCacheClient
from lightllm.server.visualserver import set_vit_att_backend
from lightllm.server.embed_cache.afs_utils import SepEmbedHandler
from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)


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
        #     "max_batch_size": max(self.infer_batch_size // self.vit_dp, 1),
        #     "vit_attn_backend": self.vit_attn_backend,
        # }

        weight_dir = kvargs["weight_dir"]
        self.infer_max_batch_size = kvargs["max_batch_size"]
        self.device_id = kvargs["device_id"]
        self.vit_tp = kvargs["vit_tp"]
        self.dp_rank_id = kvargs["dp_rank_id"]
        self.tp_rank_id = kvargs["tp_rank_id"]
        self.cache_port = kvargs["cache_port"]
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
            elif self.model_type in ["qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe"]:
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
                self.cache_client = rpyc.connect("localhost", self.cache_port, config={"allow_pickle": True})
                self.cache_client._channel.stream.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.cpu_embed_cache_client = CpuEmbedCacheClient(create_meta_data=False, init_shm_data=False)
            else:
                # 独立部署vit模式下，不需要连接 cache_client, 结果是写入 afs
                args = get_env_start_args()
                self.args = args
                assert args.visual_dp == 1
                if self.tp_rank_id == 0:
                    self.afs_handler = SepEmbedHandler(
                        afs_embed_dir=self.args.afs_image_embed_dir,
                        redis_host=self.args.config_server_host,
                        redis_port=self.args.config_server_visual_redis_port,
                        capacity=self.args.afs_embed_capacity,
                    )

            self._init_taskes()
        except Exception as e:
            print("#" * 16)
            print("load model error:", str(e), e, type(e))
            import traceback

            traceback.print_exc()
            raise e

        set_random_seed(2147483647)
        return

    def exposed_run_task(self, images: List["ImageItem"], ref_event_list: List[threading.Event]):
        try:
            images = obtain(images)
            for i in range(len(images)):
                images[i].event = ref_event_list[i]
                images[i].start_time = time.time()
                self.infer_queue.put(images[i])

        except BaseException as e:
            logger.exception(str(e))
            raise e
        return

    def _log_latency(self, image: ImageItem, stage: str):
        latency = time.time() - image.start_time
        if latency > 0.02:
            logger.info(f"{stage} latency {latency:.4f} seconds for image with md5 {image.md5}")
        image.start_time = time.time()

    def _init_taskes(self):
        self.args = get_env_start_args()

        # 异步队列, 用于接受任务
        self.infer_queue = queue.Queue()
        # 将计算得到的结果放入 afs 或者 embed cache 的 queue
        self.store_queue = queue.Queue()

        # 限制并发, 主要是为了控制内存用量，防止过多造成内存OOM
        self.sempare = threading.Semaphore(self.infer_max_batch_size * 8)

        # 用于同步各个推理tp每次拿到一样的image数量建立的gloo通信组
        self.gloo_group = dist.new_group(ranks=list(range(self.vit_tp)), backend="gloo")

        # 启动任务处理线程
        self._infer_thread = threading.Thread(target=self._infer_worker, daemon=True)
        self._infer_thread.start()

        self._store_thread = threading.Thread(target=self._store_worker, daemon=True)
        self._store_thread.start()
        return

    # @calculate_time(show=True, min_cost_ms=150)
    @torch.no_grad()
    def _forward(self, images: List[ImageItem]):
        return self.model.encode(images)

    def _get_image_items_from_infer_queue(self, max_num: int, force_same: bool = False) -> List[ImageItem]:
        """
        从队列中批量获取任务，直到达到 max_num 或队列为空。
        """
        tasks = []
        # 至少获取一个任务，阻塞
        self.sempare.acquire()
        task = self.infer_queue.get(block=True)
        tasks.append(task)

        if not force_same:
            # 尝试继续获取更多任务，直到达到 max_num
            while len(tasks) < max_num:
                try:
                    self.sempare.acquire()
                    task = self.infer_queue.get(block=False)
                    tasks.append(task)
                except queue.Empty:
                    self.sempare.release()
                    break
        else:
            while len(tasks) < max_num:
                self.sempare.acquire()
                task = self.infer_queue.get(block=True)
                tasks.append(task)

        return tasks

    def _get_image_items_from_store_queue(self, max_num: int) -> List[ImageItem]:
        """
        从队列中批量获取任务，直到达到 max_num 或队列为空。
        """
        tasks = []
        # 至少获取一个任务，阻塞
        task = self.store_queue.get(block=True)
        tasks.append(task)

        while len(tasks) < max_num:
            try:
                task = self.store_queue.get(block=False)
                tasks.append(task)
            except queue.Empty:
                break

        return tasks

    def _infer_worker(self):
        """
        任务处理循环: 从队列中取出任务, 执行完成后通知调用者
        """
        torch.cuda.set_device(self.device_id)
        while True:
            try:
                # 从队列获取任务, 阻塞等待
                if self.tp_rank_id == 0:
                    images = self._get_image_items_from_infer_queue(max_num=self.infer_max_batch_size)
                    dist.broadcast_object_list([len(images)], src=0, group=self.gloo_group)
                else:
                    ans = [None]
                    dist.broadcast_object_list(ans, src=0, group=self.gloo_group)
                    images = self._get_image_items_from_infer_queue(max_num=ans[0], force_same=True)

                for image in images:
                    self._log_latency(image, stage="queue_cost_time")

                # 执行任务: 调用父类的forward方法处理图像
                all_img_embeds, uuids, valid_ids = self._forward(images)
                all_img_embeds = all_img_embeds.to(torch.device("cuda"))

                if self.is_visual_only_mode:
                    self._store_to_afs(all_img_embeds, valid_ids, images)
                else:
                    self._store_to_cpu_cache(all_img_embeds, valid_ids, images)

            except Exception as e:
                logger.exception(str(e))
                raise e

    def _store_to_cpu_cache(self, all_img_embeds, valid_ids, images):
        for i in range(len(images)):
            start, end = valid_ids[i]
            image = images[i]
            if self.tp_rank_id == 0:
                self.cpu_embed_cache_client.copy_vision_to_cache(
                    embed_tensor=all_img_embeds[start:end], start_index_in_cache=image.start_index_in_embed_cache
                )
            cuda_event = torch.cuda.Event()
            cuda_event.record()
            image.cuda_event = cuda_event
            self.store_queue.put(image)

    def _store_to_afs(self, all_img_embeds, valid_ids, images):
        all_img_embeds = all_img_embeds.detach().cpu()
        for image, valid_id in zip(images, valid_ids):
            self._log_latency(image, stage="inference")
            start, end = valid_id
            gen_embed = all_img_embeds[start:end]
            image.gen_embed = gen_embed
            self.store_queue.put(image)

    def _store_worker(self):
        """
        任务处理循环: 从队列中取出ImageItem和embed 放入 afs中, 执行完成后通知调用者
        """
        while True:
            try:
                # 从队列获取任务, 阻塞等待
                images: List[ImageItem] = self._get_image_items_from_store_queue(max_num=self.infer_max_batch_size)

                if self.is_visual_only_mode:
                    self._commit_to_afs(images=images)
                else:
                    self._commit_to_cpu_cache(images=images)

                for _ in images:
                    self.sempare.release()

            except Exception as e:
                logger.exception(str(e))
                raise e

    def _commit_to_afs(self, images):
        if self.tp_rank_id == 0:
            for image in images:
                self.afs_handler.insert(image.md5, image.gen_embed)
                self._log_latency(image, stage="store_to_afs")
                image.event.set()
                self._log_latency(image, stage="set_event")

    def _commit_to_cpu_cache(self, images):
        if self.tp_rank_id == 0:
            for image in images:
                # 等待拷贝到cpu cache 完成。
                image.cuda_event.synchronize()
                self._log_latency(image, stage="inference")

            uuids = [image.uuid for image in images]
            self.cache_client.root.set_items_embed(uuids)

            for image in images:
                self._log_latency(image, stage="set_items_embed")

            for image in images:
                image.event.set()
                self._log_latency(image, stage="set_event")
