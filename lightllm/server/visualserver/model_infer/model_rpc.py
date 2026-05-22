import os
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
from lightllm.server.visualserver.model_infer.batching import pull_batch_with_budget
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
        self.visual_batch_max_tokens = kvargs.get("visual_batch_max_tokens", None)
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
                # sync_request_timeout 让阻塞的 RPyC 调用从 socket 层真正抛 TimeoutError,
                # 避免我们手写的 _call_with_timeout 留下"永远 hang 在 sock.recv 上"的孤儿线程
                # (2026-05-09 incident, AC#8)。
                set_items_embed_timeout = float(
                    getattr(get_env_start_args(), "visual_set_items_embed_timeout", 0) or 0
                ) or 30.0
                self.cache_client = rpyc.connect(
                    "localhost",
                    self.cache_port,
                    config={"allow_pickle": True, "sync_request_timeout": set_items_embed_timeout},
                )
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

    def exposed_run_task(self, images: List["ImageItem"], ref_result_list):
        try:
            # 长驻 worker 已挂掉时直接抛错, 让 manager 端走 abort 路径,
            # 避免请求被 enqueue 后永远卡在 result.wait() 上 (2026-05-09 incident)。
            self._assert_workers_alive()
            images = obtain(images)
            for i in range(len(images)):
                # ref_result_list[i] 是 manager 端 VisualInferResult 的 RPyC netref;
                # 调用其 mark_success / mark_failure 会回到 manager 进程修改状态并触发 event。
                images[i].result = ref_result_list[i]
                images[i].start_time = time.time()
                self.infer_queue.put(images[i])

        except BaseException as e:
            logger.exception(str(e))
            raise e
        return

    def _assert_workers_alive(self):
        infer_alive = getattr(self, "_infer_thread", None) is not None and self._infer_thread.is_alive()
        store_alive = getattr(self, "_store_thread", None) is not None and self._store_thread.is_alive()
        if not (infer_alive and store_alive):
            msg = (
                f"visual worker dead: infer_alive={infer_alive} store_alive={store_alive} "
                f"dp={getattr(self, 'dp_rank_id', '?')} tp={getattr(self, 'tp_rank_id', '?')}"
            )
            logger.error(msg)
            raise RuntimeError(msg)

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

        # 串行化经过 self.cache_client 的同步 RPyC 调用 (set_items_embed 等)。
        # 单线程持有的锁, 防止超时后台 thread 与新一轮调用共用同一条 RPyC 连接,
        # 同时把"卡死的 cache 调用"维持在最多 1 个 leaked thread 内 (而非每次 timeout 都新增一个)。
        self._cache_call_lock = threading.Lock()

        # 用于同步各个推理tp每次拿到一样的image数量建立的gloo通信组
        self.gloo_group = dist.new_group(ranks=list(range(self.vit_tp)), backend="gloo")

        # 启动任务处理线程
        self._infer_thread = threading.Thread(target=self._infer_worker, daemon=True)
        self._infer_thread.start()

        self._store_thread = threading.Thread(target=self._store_worker, daemon=True)
        self._store_thread.start()

        # 周期性 watchdog: 不依赖新 run_task 触发, 即使一段时间没流量也能在第一时间发现
        # worker 意外死亡并打错误日志, 满足 issue.md 验收 9 (watchdog / health signal)。
        self._workers_dead_reported = False
        # Stale sentinel cleanup: glob ALL visual-unhealthy sentinels on the host
        # (not just the current unique_name) and remove any whose recorded PID is
        # no longer alive. Handles both:
        #   * crash + same-name restart (old code already covered this)
        #   * crash + different-unique-name restart (e.g. port change, run mode change)
        #     where the old sentinel would otherwise linger forever and falsely trip
        #     external glob-based probes.
        self._sweep_stale_sentinels()
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True, name="visual_watchdog")
        self._watchdog_thread.start()
        return

    @staticmethod
    def _parse_env_bool(name: str) -> bool:
        """Truthy env-var parser that treats "0"/"false"/"no"/"off" as false.

        ``bool(os.getenv(name))`` is wrong because the empty string is the only
        falsy str — "0", "false", "False" would all evaluate true.
        """
        raw = os.getenv(name)
        if raw is None:
            return False
        return raw.strip().lower() not in ("", "0", "false", "no", "off")

    _SENTINEL_PREFIX = "/tmp/lightllm_visual_unhealthy_"

    def _watchdog_unhealthy_path(self) -> str:
        """Sentinel file path consumed by /health and external probes.

        Includes the unique server name so multiple LightLLM instances on the same
        host don't collide on /tmp paths. Format must stay in sync with
        ``lightllm.utils.health_check.scan_visual_unhealthy_sentinels``.
        """
        from lightllm.utils.envs_utils import get_unique_server_name

        return f"{self._SENTINEL_PREFIX}{get_unique_server_name()}_dp{self.dp_rank_id}_tp{self.tp_rank_id}"

    def _sweep_stale_sentinels(self):
        """Remove any visual-unhealthy sentinels whose writer PID is dead.

        Glob-based external probes need this — without it, a process crash leaves
        the file behind forever and probes mark a freshly-healthy service unhealthy.
        We identify staleness by checking the embedded PID with ``os.kill(pid, 0)``.
        """
        import glob

        for path in glob.glob(self._SENTINEL_PREFIX + "*"):
            try:
                with open(path) as fh:
                    content = fh.read()
                pid = None
                for tok in content.split():
                    if tok.startswith("pid="):
                        try:
                            pid = int(tok.split("=", 1)[1])
                        except ValueError:
                            pid = None
                        break
                stale = False
                if pid is None:
                    # Pre-PID file format from an older crash — treat as stale.
                    stale = True
                else:
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        stale = True
                    except PermissionError:
                        # PID belongs to another user — assume still alive, don't remove.
                        stale = False
                if stale:
                    os.remove(path)
                    logger.info(f"[watchdog] removed stale sentinel from dead PID: {path}")
            except FileNotFoundError:
                pass
            except Exception:
                logger.exception(f"[watchdog] sweep failed for {path}; continuing")

    def _watchdog_loop(self):
        """Periodic ViT-worker liveness check.

        On detected death we:
          1. Log ERROR once (with queue sizes for diagnostics).
          2. Write a sentinel file at ``_watchdog_unhealthy_path()`` — operators /
             external probes can `test -e` this to flag the visualserver unhealthy
             without needing to talk to the RPC server.
          3. Optionally ``os._exit(1)`` if ``LIGHTLLM_VISUAL_EXIT_ON_WORKER_DEATH`` is
             set, so a process supervisor (systemd / k8s / parent_check) can perform
             controlled restart. Off by default — letting the process linger preserves
             debuggability; opt-in enables auto-recovery.

        Without traffic the previous `_assert_workers_alive()` gate would never fire,
        so the watchdog guarantees an active signal regardless of request rate.
        """
        exit_on_death = self._parse_env_bool("LIGHTLLM_VISUAL_EXIT_ON_WORKER_DEATH")
        while True:
            try:
                time.sleep(30)
                infer_alive = self._infer_thread.is_alive()
                store_alive = self._store_thread.is_alive()
                queue_size = self.infer_queue.qsize()
                store_size = self.store_queue.qsize()
                if not (infer_alive and store_alive):
                    if not self._workers_dead_reported:
                        logger.error(
                            f"[watchdog] visual worker dead: "
                            f"infer_alive={infer_alive} store_alive={store_alive} "
                            f"dp={self.dp_rank_id} tp={self.tp_rank_id} "
                            f"infer_queue={queue_size} store_queue={store_size}"
                        )
                        # Sentinel file for external probes / k8s readiness checks.
                        # Embed the writer PID so a future restart can sweep stale files
                        # whose PID is no longer alive (see _sweep_stale_sentinels).
                        try:
                            with open(self._watchdog_unhealthy_path(), "w") as fh:
                                fh.write(
                                    f"pid={os.getpid()} "
                                    f"infer_alive={infer_alive} store_alive={store_alive} "
                                    f"dp={self.dp_rank_id} tp={self.tp_rank_id} "
                                    f"at={time.time()}\n"
                                )
                        except Exception:
                            logger.exception("[watchdog] failed to write sentinel file; continuing")
                        self._workers_dead_reported = True
                        if exit_on_death:
                            logger.error(
                                "[watchdog] LIGHTLLM_VISUAL_EXIT_ON_WORKER_DEATH set; "
                                "exiting visual rpc process so supervisor can restart"
                            )
                            os._exit(1)
                else:
                    # Sentinel cleanup on recovery (defensive — worker threads do not actually revive).
                    if self._workers_dead_reported:
                        logger.warning("[watchdog] workers recovered from previous dead state")
                        try:
                            path = self._watchdog_unhealthy_path()
                            if os.path.exists(path):
                                os.remove(path)
                        except Exception:
                            logger.exception("[watchdog] sentinel cleanup failed; continuing")
                        self._workers_dead_reported = False
                    logger.debug(
                        f"[watchdog] alive infer={infer_alive} store={store_alive} "
                        f"infer_queue={queue_size} store_queue={store_size}"
                    )
            except Exception:
                # watchdog 本身不能死掉
                logger.exception("[watchdog] loop iteration raised; continuing")

    # @calculate_time(show=True, min_cost_ms=150)
    @torch.no_grad()
    def _forward(self, images: List[ImageItem]):
        return self.model.encode(images)

    def _get_image_items_from_infer_queue(
        self, max_num: int, force_same: bool = False, timeout: float = None
    ) -> List[ImageItem]:
        """
        从队列中批量获取任务，直到达到 max_num 或队列为空。

        timeout 仅对首个任务的阻塞等待生效；超时返回空列表。rank 0 使用
        timeout 作为心跳，避免其他 rank 在 gloo broadcast 上长时间无响应
        而触发 30 分钟超时崩溃。

        On rank 0 the cumulative ``img.token_num`` is additionally capped by
        ``visual_batch_max_tokens`` so a dynamic-resolution image (or batch of
        them) cannot blow the ViT's memory budget. The non-rank-0 ``force_same``
        path follows rank 0's already-decided count via the gloo broadcast.
        """
        if force_same:
            tasks = []
            self.sempare.acquire()
            tasks.append(self.infer_queue.get(block=True))
            while len(tasks) < max_num:
                self.sempare.acquire()
                tasks.append(self.infer_queue.get(block=True))
            return tasks

        return pull_batch_with_budget(
            infer_queue=self.infer_queue,
            semaphore=self.sempare,
            max_num=max_num,
            max_tokens=self.visual_batch_max_tokens,
            timeout=timeout,
        )

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

        一个 batch 出错 (例如截断图片导致 PIL 解码失败) 不能让长驻线程退出,
        否则后续所有图片请求都会卡死在 ``image.result.event.wait()`` 上。
        失败要通过 ``image.result.mark_failure`` 通知 manager, 让其走 abort 路径,
        否则下游 router 会拿到没有 embedding 的请求 (2026-05-09 incident)。
        """
        torch.cuda.set_device(self.device_id)
        while True:
            images: List[ImageItem] = []
            handed_off = False
            try:
                # 从队列获取任务, 阻塞等待
                # rank 0 用带超时的 get, 空闲时也会广播 [0] 当作心跳,
                # 避免其他 rank 在 gloo broadcast 上长时间无响应而触发 30 分钟超时崩溃。
                if self.tp_rank_id == 0:
                    images = self._get_image_items_from_infer_queue(max_num=self.infer_max_batch_size, timeout=60.0)
                    dist.broadcast_object_list([len(images)], src=0, group=self.gloo_group)
                    if len(images) == 0:
                        continue
                else:
                    ans = [None]
                    dist.broadcast_object_list(ans, src=0, group=self.gloo_group)
                    if ans[0] == 0:
                        continue
                    images = self._get_image_items_from_infer_queue(max_num=ans[0], force_same=True)

                for image in images:
                    self._log_latency(image, stage="queue_cost_time")

                # 执行任务: 调用父类的forward方法处理图像
                # 部分 visual model (qwen3_vl) 会把单张坏图标记为 image.preprocess_failed,
                # 这里从 batch 中拆出失败 image 单独标记 mark_failure, 保证 batch 中其他正常图片继续。
                all_img_embeds, uuids, valid_ids = self._forward(images)

                successful_images = [img for img in images if not getattr(img, "preprocess_failed", False)]
                failed_images = [img for img in images if getattr(img, "preprocess_failed", False)]
                # 先把 failed_images 从 local images 里移走再处理, 避免后面 _store 失败时
                # _fail_batch 重复释放它们的 semaphore / 重复 mark_failure。
                images = list(successful_images)

                for image in failed_images:
                    if self.tp_rank_id == 0:
                        self._mark_failure(image, "preprocess_failed")
                    try:
                        self.sempare.release()
                    except Exception:
                        logger.exception("semaphore release for preprocess-failed image raised; continuing")

                if not successful_images:
                    continue

                all_img_embeds = all_img_embeds.to(torch.device("cuda"))

                # _store_to_* 内部全量准备好状态再统一入队, 入队完成后才把 ownership 移交给 store_worker。
                # 在入队前抛异常时, images 仍然有效, 外层 except 会通过 _fail_batch 通知所有图片失败。
                if self.is_visual_only_mode:
                    self._store_to_afs(all_img_embeds, valid_ids, successful_images)
                else:
                    self._store_to_cpu_cache(all_img_embeds, valid_ids, successful_images)

                # 入队成功后, ownership 已经移交给 store_worker。同 batch 中的 failed_images
                # 已经在前面处理完, 此处只需要让 finally 不要再处理已经入队的 successful_images。
                handed_off = True
                images = []

            except Exception as e:
                # handed_off=True 时不会到这里 (上面已经将 images 清空), 但即使到了也不会双重处理。
                if not handed_off:
                    self._fail_batch(images, stage="_infer_worker", exc=e)
                else:
                    logger.exception(f"_infer_worker post-handoff exception (ignored): {e}")

    def _store_to_cpu_cache(self, all_img_embeds, valid_ids, images):
        """全量准备好 cuda event 后再统一入队, 避免中途失败留下半批已入队的 image。"""
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
        for image in images:
            self.store_queue.put(image)

    def _store_to_afs(self, all_img_embeds, valid_ids, images):
        all_img_embeds = all_img_embeds.detach().cpu()
        for image, valid_id in zip(images, valid_ids):
            self._log_latency(image, stage="inference")
            start, end = valid_id
            gen_embed = all_img_embeds[start:end]
            image.gen_embed = gen_embed
        for image in images:
            self.store_queue.put(image)

    def _store_worker(self):
        """
        任务处理循环: 从队列中取出ImageItem和embed 放入 afs中, 执行完成后通知调用者

        与 ``_infer_worker`` 同样需要在异常时不退出, 保证长驻线程持续消费 store_queue。
        """
        while True:
            images: List[ImageItem] = []
            try:
                # 从队列获取任务, 阻塞等待
                images = self._get_image_items_from_store_queue(max_num=self.infer_max_batch_size)

                if self.is_visual_only_mode:
                    self._commit_to_afs(images=images)
                else:
                    self._commit_to_cpu_cache(images=images)

                for _ in images:
                    self.sempare.release()

                images = []

            except Exception as e:
                self._fail_batch(images, stage="_store_worker", exc=e)

    def _commit_to_afs(self, images):
        if self.tp_rank_id == 0:
            for image in images:
                try:
                    self.afs_handler.insert(image.md5, image.gen_embed)
                    self._log_latency(image, stage="store_to_afs")
                    self._mark_success(image)
                except Exception as e:
                    logger.exception(f"afs insert failed for md5={image.md5}: {e}")
                    self._mark_failure(image, f"afs_insert_failed: {e}")
                self._log_latency(image, stage="set_event")

    def _commit_to_cpu_cache(self, images):
        if self.tp_rank_id == 0:
            for image in images:
                # 等待拷贝到cpu cache 完成。
                image.cuda_event.synchronize()
                self._log_latency(image, stage="inference")

            uuids = [image.uuid for image in images]
            timeout = float(getattr(self.args, "visual_set_items_embed_timeout", 0) or 0) or 30.0
            logger.info(f"set_items_embed START dp={self.dp_rank_id} batch={len(uuids)} timeout={timeout}s")
            cache_ok = False
            cache_err: str = ""
            try:
                # set_items_embed 是同步 RPyC 调用, embed cache 卡死时它本身不会返回。
                # 用线程 + join(timeout) 包一层, 同时通过 _cache_call_lock 串行化所有 cache 调用,
                # 卡住时让本批次直接以 cache_failed 告知 manager, 并防止 daemon 线程累积。
                self._call_cache_with_timeout(
                    self.cache_client.root.set_items_embed,
                    args=(uuids,),
                    timeout=timeout,
                    desc="set_items_embed",
                )
                cache_ok = True
                logger.info(f"set_items_embed DONE dp={self.dp_rank_id} batch={len(uuids)}")
            except Exception as e:
                cache_err = str(e)
                logger.exception(
                    f"set_items_embed failed dp={self.dp_rank_id} batch={len(uuids)} "
                    f"uuids={uuids}: {e}"
                )

            for image in images:
                self._log_latency(image, stage="set_items_embed")

            for image in images:
                if cache_ok:
                    self._mark_success(image)
                else:
                    self._mark_failure(image, f"set_items_embed_failed: {cache_err}")
                self._log_latency(image, stage="set_event")

    def _call_cache_with_timeout(self, fn, args, timeout, desc: str):
        """Serialize cache_client RPC calls + bound wall-clock budget.

        A persistent embed-cache stall used to leak unbounded daemon threads on a
        shared ``self.cache_client`` connection. The lock guarantees only one
        thread is actually issuing an RPyC call at a time. New callers either
        succeed within ``timeout`` or time out cleanly via ``_call_with_timeout``;
        they do not pile up against the connection.
        """
        # Test affordance: inject a stall to verify the timeout path. Set
        # LIGHTLLM_VISUAL_INJECT_CACHE_STALL=<seconds> to make every cache RPC
        # sleep that many seconds instead of running. Used to exercise issue.md
        # acceptance criterion 8 end-to-end without standing up a broken cache server.
        stall_s = os.getenv("LIGHTLLM_VISUAL_INJECT_CACHE_STALL", "")
        if stall_s:
            try:
                stall = float(stall_s)
            except ValueError:
                stall = 0.0
            if stall > 0:
                fn = lambda *a, **k: time.sleep(stall) or None  # noqa: E731

        def _locked_call(*locked_args):
            # Sub-timeout acquire so a stuck predecessor doesn't park us indefinitely;
            # the outer _call_with_timeout still bounds the whole call.
            if not self._cache_call_lock.acquire(timeout=timeout):
                raise TimeoutError(
                    f"{desc} could not acquire _cache_call_lock within {timeout}s (cache appears stalled)"
                )
            try:
                return fn(*locked_args)
            finally:
                self._cache_call_lock.release()

        return self._call_with_timeout(_locked_call, args=args, timeout=timeout, desc=desc)

    @staticmethod
    def _call_with_timeout(fn, args, timeout, desc: str):
        """同步调用 ``fn(*args)``, 超过 ``timeout`` 秒抛 TimeoutError。

        线程化等待避免 RPyC 同步调用永远阻塞调用方。注意: 后台线程在超时后仍会
        持续运行, 因为我们无法安全地取消一个 C-extension 内的远端调用, 但它不会
        阻塞 worker 推进 (worker 已经走 mark_failure 通知 manager)。
        """
        result: List = [None]
        exc: List[Optional[BaseException]] = [None]
        done = threading.Event()

        def runner():
            try:
                result[0] = fn(*args)
            except BaseException as e:
                exc[0] = e
            finally:
                done.set()

        t = threading.Thread(target=runner, daemon=True, name=f"call_with_timeout_{desc}")
        t.start()
        if not done.wait(timeout):
            raise TimeoutError(f"{desc} timed out after {timeout}s")
        if exc[0] is not None:
            raise exc[0]
        return result[0]

    def _mark_success(self, image: ImageItem):
        """成功完成单张图片: 通过 RPyC netref 通知 manager 端的 VisualInferResult。"""
        result_ref = getattr(image, "result", None)
        if result_ref is None:
            return
        try:
            result_ref.mark_success()
        except Exception:
            logger.exception(
                f"mark_success failed uuid={getattr(image, 'uuid', None)} md5={getattr(image, 'md5', None)}"
            )

    def _mark_failure(self, image: ImageItem, reason: str):
        """单张图片失败: 通过 RPyC netref 让 manager 走 abort 路径。"""
        result_ref = getattr(image, "result", None)
        if result_ref is None:
            return
        try:
            result_ref.mark_failure(reason)
        except Exception:
            logger.exception(
                f"mark_failure failed uuid={getattr(image, 'uuid', None)} "
                f"md5={getattr(image, 'md5', None)} reason={reason}"
            )

    def _fail_batch(self, images: List[ImageItem], stage: str, exc: Exception):
        """
        Per-batch 失败兜底: 记录上下文, 释放信号量, 把所有 image 标记为失败, 然后继续 worker 循环。

        - 仅 rank 0 持有外部 result 引用, 因此只在 rank 0 调 mark_failure。
        - 信号量在所有 rank 上各自维护, 因此所有 rank 都需要释放。
        - 任何一步出错都被吞掉, 因为 worker 线程不能因为兜底逻辑再次退出。
        - 失败状态会让 manager.handle_images 的 wait 后检查抛 RuntimeError, 触发 abort 路径。
        """
        uuids_log = [getattr(img, "uuid", None) for img in images]
        md5s_log = [getattr(img, "md5", None) for img in images]
        logger.exception(
            f"{stage} batch failed, recovering and continuing: "
            f"batch_size={len(images)} dp={self.dp_rank_id} tp={self.tp_rank_id} "
            f"uuids={uuids_log} md5s={md5s_log}: {exc}"
        )
        reason = f"{stage}: {exc}"
        for image in images:
            if self.tp_rank_id == 0:
                self._mark_failure(image, reason)
            try:
                self.sempare.release()
            except Exception:
                logger.exception("semaphore release during fail_batch raised; continuing")
