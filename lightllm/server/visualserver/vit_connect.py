import asyncio
import zmq
import zmq.asyncio
import time
import pickle
from typing import Dict, List, Optional, Any
from lightllm.utils.log_utils import init_logger
import httpx
import base64
from dataclasses import dataclass

logger = init_logger(__name__)


@dataclass
class VIT_Obj:
    node_id: int
    host_ip_port: str

    def to_log_str(self):
        return f"VIT host_ip_port: {self.host_ip_port} node_id: {self.node_id}"


class VITConnectionManager:
    """VIT连接管理器"""

    def __init__(self, args, context, local_visual_port: int):
        self.args = args
        self.context = context
        self.local_visual_port = local_visual_port

        self.send_to_visual = None
        self.remote_vit_instances = {}
        self.current_vit_index = 0
        self.remote_vit = args.enable_remote_vit
        self.remote_vit_port = args.remote_vit_port

        self._setup_vit_connections()

    def _setup_vit_connections(self):
        """
        设置VIT连接，支持本地和远程VIT实例
        支持多种连接模式：
        1. 本地VIT实例 (默认)
        2. 远程多个VIT实例 (负载均衡)
        """
        if self.remote_vit:
            # 远程VIT实例模式
            self._setup_remote_vit_connections()
        else:
            self._setup_local_vit_connection()

    def _setup_local_vit_connection(self):
        self.send_to_visual = self.context.socket(zmq.PUSH)
        self.send_to_visual.connect(f"{self.args.zmq_mode}127.0.0.1:{self.local_visual_port}")
        logger.info(f"Connected to local VIT instance at {self.args.zmq_mode}127.0.0.1:{self.local_visual_port}")

    def _setup_remote_vit_connections(self):
        """
        初始化远程VIT连接，同步获取初始实例
        """
        logger.info("Setting up remote VIT connections...")

        self._sync_init_vit_instances()

        retry_count = 0
        max_retries = 30  # 最多等待30秒
        while len(self.remote_vit_instances) == 0 and retry_count < max_retries:
            logger.info(f"Waiting for VIT instances... (attempt {retry_count + 1}/{max_retries})")
            time.sleep(1)
            retry_count += 1
            self._sync_init_vit_instances()

        if len(self.remote_vit_instances) == 0:
            logger.warning("No VIT instances available after initialization")
        else:
            logger.info(f"Successfully connected to {len(self.remote_vit_instances)} VIT instances")

    def _sync_init_vit_instances(self):
        """
        同步初始化VIT实例连接
        """
        try:
            # 使用同步方式获取VIT实例
            vit_objs = self._sync_get_vit_objs()
            if vit_objs:
                self._update_vit_connections(vit_objs)
        except Exception as e:
            logger.error(f"Failed to initialize VIT instances: {e}")

    def _sync_get_vit_objs(self) -> Optional[Dict[int, VIT_Obj]]:
        """
        同步获取VIT实例信息
        """
        import requests

        uri = f"http://{self.args.config_server_host}:{self.args.config_server_port}/registered_visual_objects"
        try:
            response = requests.get(uri, timeout=10)
            if response.status_code == 200:
                base64data = response.json()["data"]
                id_to_vit_obj = pickle.loads(base64.b64decode(base64data))
                return id_to_vit_obj
            else:
                logger.error(f"Failed to get VIT instances: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error getting VIT instances: {e}")
            return None

    def _update_vit_connections(self, id_to_vit_obj: Dict[int, VIT_Obj]):
        """
        更新VIT连接，添加新的连接，关闭失效的连接
        """
        # 关闭不再存在的连接
        closed_ids = []
        for id, remote_instance in self.remote_vit_instances.items():
            if id not in id_to_vit_obj:
                try:
                    remote_instance.close()
                except:
                    pass
                closed_ids.append(id)
                logger.info(f"Closed VIT connection {id}")

        for id in closed_ids:
            self.remote_vit_instances.pop(id)

        # 建立新的连接
        for id, vit_obj in id_to_vit_obj.items():
            if id not in self.remote_vit_instances:
                try:
                    socket = self.context.socket(zmq.PUSH)
                    print(vit_obj.host_ip_port, self.args.remote_vit_port, flush=True)
                    ip, port = vit_obj.host_ip_port.split(":")
                    socket.connect(f"tcp://{ip}:{port}")
                    self.remote_vit_instances[id] = socket
                    logger.info(f"Connected to VIT instance {id} at {vit_obj.host_ip_port}")
                except Exception as e:
                    logger.error(f"Failed to connect to VIT instance {id}: {e}")

    def _get_vit_instance(self):
        """
        获取下一个可用的VIT实例 (轮询负载均衡)
        """
        if not self.remote_vit:
            return self.send_to_visual

        if len(self.remote_vit_instances) == 0:
            raise Exception("No available VIT instances")

        # 简单的轮询负载均衡
        index = (self.current_vit_index + 1) % len(self.remote_vit_instances)
        self.current_vit_index = index
        return list(self.remote_vit_instances.values())[index]

    async def send_to_vit(self, data, protocol=pickle.HIGHEST_PROTOCOL):
        """
        发送数据到VIT实例，支持本地和远程模式
        """
        instance = self._get_vit_instance()
        try:
            print(instance, flush=True)
            instance.send_pyobj(data, protocol=protocol)
        except Exception as e:
            logger.error(f"Failed to send to VIT instance: {e}")
            raise Exception(f"Failed to send to VIT instance: {e}")
        finally:
            # 释放图片资源
            data.multimodal_params.free()
        await self._wait_visual_embed_ready()

    async def vit_handle_loop(self):
        """
        异步VIT连接管理循环，由外部启动
        """
        logger.info("Starting VIT connection management loop")
        while True:
            try:
                id_to_vit_obj = await self._async_get_vit_objs()
                if id_to_vit_obj:
                    logger.debug(f"Retrieved {len(id_to_vit_obj)} VIT instances")
                    self._update_vit_connections(id_to_vit_obj)
                await asyncio.sleep(30)
            except Exception as e:
                logger.exception(f"Error in VIT handle loop: {e}")
                await asyncio.sleep(10)

    async def _async_get_vit_objs(self) -> Optional[Dict[int, VIT_Obj]]:
        """
        异步获取VIT实例信息
        """
        uri = f"ws://{self.args.config_server_host}:{self.args.config_server_port}/registered_visual_objects"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(uri)
                if response.status_code == 200:
                    base64data = response.json()["data"]
                    id_to_vit_obj = pickle.loads(base64.b64decode(base64data))
                    return id_to_vit_obj
                else:
                    logger.error(f"Failed to get VIT instances: {response.status_code}")
                    return None
        except Exception as e:
            logger.exception(f"Error getting VIT instances: {e}")
            return None

    async def _wait_visual_embed_ready(self):
        """
        等待VIT实例的embed准备好
        """
        await asyncio.sleep(10)
