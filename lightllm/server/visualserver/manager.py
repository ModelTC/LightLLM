import zmq
import zmq.asyncio
import asyncio
import concurrent.futures
import uvloop
import rpyc
import socket
import pickle
import inspect
import setproctitle
import threading
import collections
from typing import List
from lightllm.server.core.objs.io_objs.group_req import GroupReqIndexes
from lightllm.server.core.objs import ShmReqManager, StartArgs

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from lightllm.server.multimodal_params import MultimodalParams, ImageItem
from .model_infer import start_model_process, VisualModelRpcClient
from lightllm.common.basemodel.attention_vit.create_utils import init_vit_att_backend
from lightllm.utils.log_utils import init_logger
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.utils.process_check import start_parent_check_thread
from lightllm.utils.envs_utils import get_unique_server_name
from rpyc.utils.classic import obtain


logger = init_logger(__name__)


class VisualInferResult:
    """Per-image visual inference outcome.

    Combines completion signaling with success/failure status so the manager can
    distinguish "embed is in cache, OK to forward" from "ViT failed, abort the
    request". Workers call ``mark_success`` / ``mark_failure`` (the manager
    receives the call via the RPyC netref) and the manager waits + inspects
    ``success`` after ``event.wait()`` returns.

    Before this object was introduced, workers only signaled a bare
    ``threading.Event`` on both success and failure, so the manager forwarded
    failed requests to the router with missing embeddings (2026-05-09 incident).
    """

    __slots__ = ("event", "success", "error")

    def __init__(self):
        self.event = threading.Event()
        self.success = False
        self.error: str = ""

    def mark_success(self):
        self.success = True
        self.event.set()

    def mark_failure(self, error: str = ""):
        # success stays False; record the reason so manager logs are useful.
        self.error = error or self.error or "unknown"
        self.event.set()

    def wait(self, timeout):
        return self.event.wait(timeout)


class VisualManager:
    def __init__(
        self,
        args: StartArgs,
    ):
        self.args = args
        context = zmq.Context(2)
        enable_audio = not args.disable_audio
        if enable_audio:
            self.send_to_next_module = context.socket(zmq.PUSH)
            self.send_to_next_module.connect(f"{args.zmq_mode}127.0.0.1:{args.audio_port}")
        else:
            if args.enable_cpu_cache:
                self.send_to_next_module = context.socket(zmq.PUSH)
                self.send_to_next_module.connect(f"{args.zmq_mode}127.0.0.1:{args.multi_level_kv_cache_port}")
            else:
                self.send_to_next_module = context.socket(zmq.PUSH)
                self.send_to_next_module.connect(f"{args.zmq_mode}127.0.0.1:{args.router_port}")

        self.zmq_recv_socket = context.socket(zmq.PULL)
        self.zmq_recv_socket.bind(f"{args.zmq_mode}127.0.0.1:{args.visual_port}")
        # sync_request_timeout 让阻塞的 RPyC 调用 (get_items_embed/set_items_embed) 从 socket
        # 层真正抛 TimeoutError, 避免泄漏 default executor 线程导致 net-io 饿死。
        cache_rpyc_timeout = max(
            int(getattr(args, "visual_get_items_embed_timeout", 0) or 0),
            int(getattr(args, "visual_set_items_embed_timeout", 0) or 0),
            10,
        )
        self.cache_client = rpyc.connect(
            "localhost",
            args.cache_port,
            config={"allow_pickle": True, "sync_request_timeout": cache_rpyc_timeout},
        )
        self.cache_client._channel.stream.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # 专用 executor: 同步 cache RPC 都走这个池, 与 asyncio default executor 隔离,
        # 即便某次 cache 调用 hang (sync_request_timeout 兜底前) 也不会饿死 zmq recv_pyobj。
        self._cache_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="visual_cache_rpc"
        )
        self.model_weightdir = args.model_dir
        self.vit_dp = args.visual_dp
        self.vit_tp = args.visual_tp
        # image 最大推理 batch size
        self.infer_batch_size = args.visual_infer_batch_size
        self.send_batch_size = args.visual_send_batch_size
        self.shm_req_manager = ShmReqManager()
        self.lock = asyncio.Lock()

    async def wait_to_model_ready(self):

        self.model_rpcs: List[List[VisualModelRpcClient]] = [[] for _ in range(self.vit_dp)]
        self.vit_attn_backend = init_vit_att_backend(index=0)
        for dp_rank_id in range(self.vit_dp):
            for tp_rank_id in range(self.vit_tp):

                rpc_model = await start_model_process()
                self.model_rpcs[dp_rank_id].append(rpc_model)

        init_model_ret = []
        for dp_rank_id in range(self.vit_dp):  # async init model process
            for tp_rank_id in range(self.vit_tp):
                device_id = self.args.visual_gpu_ids[dp_rank_id * self.vit_tp + tp_rank_id]
                kvargs = {
                    "weight_dir": self.model_weightdir,
                    "device_id": device_id,
                    "vit_tp": self.vit_tp,
                    "cache_port": self.args.cache_port,
                    "tp_rank_id": tp_rank_id,
                    "dp_rank_id": dp_rank_id,
                    "data_type": self.args.data_type,
                    "visual_nccl_port": self.args.visual_nccl_ports[dp_rank_id],
                    "quant_type": self.args.vit_quant_type,
                    "quant_cfg": self.args.vit_quant_cfg,
                    "max_batch_size": max(self.infer_batch_size // self.vit_dp, 1),
                    "visual_batch_max_tokens": self.args.visual_batch_max_tokens,
                    "vit_attn_backend": self.vit_attn_backend,
                }
                init_model_ret.append(self.model_rpcs[dp_rank_id][tp_rank_id].init_model(kvargs))
        await asyncio.gather(*init_model_ret)
        return

    def get_need_infer_images(self, group_req_indexes: GroupReqIndexes) -> List[ImageItem]:
        """同步路径: 检查 shm req 是否 aborted, 必要时调用 embed cache 询问哪些 image 已就绪。

        ``cache_client.root.get_items_embed`` 是同步 RPyC 调用; 调用方需要包一层
        ``asyncio.to_thread`` 并加超时, 避免 embed cache 卡死时阻塞 asyncio 事件循环
        (2026-05-09 incident)。
        """
        shm_req = self.shm_req_manager.get_req_obj_by_index(group_req_indexes.shm_req_indexes[0])
        is_aborted = shm_req.is_aborted
        disable_prompt_cache = shm_req.sample_params.disable_prompt_cache
        self.shm_req_manager.put_back_req_obj(shm_req)
        # case 0
        if is_aborted:
            # 因为连接断开 aborted 掉的请求也需要传输到后续的模块进行处理
            # 因为采用 shm 来映射所有的 req 对象以后，引用管理情况复杂了
            # 需要一些一致的流程来保证不出现异步问题。
            return []

        multimodal_params = group_req_indexes.multimodal_params
        img_uuids = [img.uuid for img in multimodal_params.images]
        # disable prompt cache通常用来测试，需要也去掉image cache的影响
        if disable_prompt_cache:
            ready_image = [False] * len(img_uuids)
        else:
            if len(img_uuids) > 0:
                ready_image = obtain(self.cache_client.root.get_items_embed(img_uuids))
            else:
                ready_image = []

        images_need_infer = []
        for img, ready in zip(multimodal_params.images, ready_image):
            if not ready:
                images_need_infer.append(img)

        return images_need_infer

    async def handle_group_indexes(self, group_req_indexes: GroupReqIndexes):
        # 把 get_need_infer_images 也放在 try 内, 并通过 dedicated executor + wait_for 加超时,
        # 因为内部包含同步 RPyC 调用 (cache_client.root.get_items_embed); embed cache 卡死时
        # 不应阻塞 asyncio 事件循环或绕过 abort 路径 (2026-05-09 incident)。
        # 关键: 用 _cache_executor 而不是 default executor, 否则卡死的 cache 调用会占满
        # default executor 进而饿死 loop_for_netio_req 里的 zmq_recv_socket.recv_pyobj。
        cache_timeout = max(int(getattr(self.args, "visual_get_items_embed_timeout", 0) or 0), 0) or 10
        loop = asyncio.get_running_loop()
        try:
            images_need_infer = await asyncio.wait_for(
                loop.run_in_executor(self._cache_executor, self.get_need_infer_images, group_req_indexes),
                timeout=cache_timeout,
            )

            if len(images_need_infer) == 0:
                self.send_to_next_module.send_pyobj(group_req_indexes, protocol=pickle.HIGHEST_PROTOCOL)
                return

            await self.handle_images(images_need_infer)
        except Exception as e:
            # visual 推理失败 (例如 worker 异常 / event 等待超时 / get_items_embed 卡住),
            # 把请求标记为 aborted 再转发, 由下游 router 正常走 abort 释放路径。
            logger.exception(
                f"handle_group_indexes failed, marking group_req_id={group_req_indexes.group_req_id} aborted: {e}"
            )
            for shm_req_index in group_req_indexes.shm_req_indexes:
                shm_req = self.shm_req_manager.get_req_obj_by_index(shm_req_index)
                shm_req.is_aborted = True
                self.shm_req_manager.put_back_req_obj(shm_req)

        self.send_to_next_module.send_pyobj(group_req_indexes, protocol=pickle.HIGHEST_PROTOCOL)
        return

    async def handle_images(self, images_need_infer: List[ImageItem]):
        if not hasattr(self, "cur_dp_index"):
            self.cur_dp_index = 0

        dp_to_handle_images = collections.defaultdict(list)
        for image in images_need_infer:
            self.cur_dp_index += 1
            select_dp = self.cur_dp_index % self.vit_dp
            dp_to_handle_images[select_dp].append((image, VisualInferResult()))

        taskes = []
        for dp_index in range(self.vit_dp):
            _images = dp_to_handle_images[dp_index]
            if _images:
                taskes.append(
                    self.infer_images(dp_index, images=[e[0] for e in _images], results=[e[1] for e in _images])
                )

        async with self.lock:
            try:
                await asyncio.gather(*taskes)
            except BaseException as e:
                logger.exception(str(e))
                raise e

        # 等待每张图片各自的完成事件 + 检查成功状态。
        # event.wait() 必须有 timeout, 否则 ViT worker 异常退出 / cache 卡死时, 这里
        # 永远不会返回, 同时被 asyncio.to_thread 占用的 default executor 线程也会被耗尽
        # (2026-05-09 incident)。
        #
        # 关键: 即使发现有失败 image, 也要先把同 batch 中其他 image 都等到 (success 或 timeout),
        # 再统一抛出。否则当一张 preprocess_failed 的图片先触发 mark_failure 时, 我们立刻
        # raise → 上层走 abort → 下游释放 multimodal cache id, 但 store_worker 此刻还在
        # 给同 batch 中成功的 image 写 embedding, 造成写到已释放槽位的竞态。
        wait_timeout = max(int(getattr(self.args, "visual_infer_timeout", 0) or 0), 0) or 120
        errors: List[str] = []
        for dp_index in range(self.vit_dp):
            _images = dp_to_handle_images[dp_index]
            for img, result in _images:
                ok = await asyncio.to_thread(result.wait, wait_timeout)
                if not ok:
                    errors.append(
                        f"timeout dp={dp_index} uuid={img.uuid} "
                        f"md5={getattr(img, 'md5', None)} timeout={wait_timeout}s"
                    )
                    continue
                if not result.success:
                    errors.append(
                        f"failed dp={dp_index} uuid={img.uuid} "
                        f"md5={getattr(img, 'md5', None)} error={result.error}"
                    )
        if errors:
            raise RuntimeError("visual infer batch had failures: " + "; ".join(errors))
        return

    async def infer_images(self, dp_index: int, images, results):
        taskes = []
        for vit_tp_rank in range(self.vit_tp):
            task = self.model_rpcs[dp_index][vit_tp_rank].run_task(images, results)
            taskes.append(task)
        await asyncio.gather(*taskes)

    async def loop_for_netio_req(self):
        try:
            while True:
                recv_req: GroupReqIndexes = await asyncio.to_thread(self.zmq_recv_socket.recv_pyobj)
                if isinstance(recv_req, GroupReqIndexes):
                    logger.debug(
                        f"visual recv req id {recv_req.group_req_id} "
                        f"img count {len(recv_req.multimodal_params.images)}"
                    )
                    asyncio.create_task(self.handle_group_indexes(group_req_indexes=recv_req))
                else:
                    assert False, f"Error Req Inf {recv_req}"
        except Exception as e:
            logger.exception(str(e))

    def clean_up(self):
        return


def start_visual_process(args, pipe_writer):
    import lightllm.utils.rpyc_fix_utils as _

    # 注册graceful 退出的处理
    graceful_registry(inspect.currentframe().f_code.co_name)
    setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::visual_server")
    start_parent_check_thread()
    try:
        visualserver = VisualManager(args=args)
        asyncio.run(visualserver.wait_to_model_ready())
    except Exception as e:
        logger.exception(str(e))
        visualserver.clean_up()
        raise e

    pipe_writer.send("init ok")

    def handle_exception(loop, context):
        logger.exception(f"VisualServer Caught exception: {str(context)}")

    loop = asyncio.new_event_loop()
    loop.set_exception_handler(handle_exception)
    asyncio.set_event_loop(loop)
    loop.run_until_complete(visualserver.loop_for_netio_req())
    return
