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
        self.remote_vit_instances = []
        self.current_vit_index = 0
        self.remote_vit = args.enable_remote_vit
        self.remote_vit_port = args.remote_vit_port

        self._setup_vit_connections()

    def _setup_vit_connections(self):
        """
        设置VIT连接，支持本地和远程VIT实例
        支持多种连接模式：
        1. 本地VIT实例 (默认)
        2. 远程单个VIT实例
        3. 远程多个VIT实例 (负载均衡)
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
        print("_setup_remote_vit_connections", "fdakpgdakgjadpgkjadk")
        asyncio.create_task(self.vit_handle_loop())

        # wait for remote vit instances
        while True:
            if len(self.remote_vit_instances) > 0:
                break
            time.sleep(1)

    def _get_vit_instance(self):
        """
        获取下一个可用的VIT实例 (轮询负载均衡)
        """
        if not self.remote_vit:
            return self.send_to_visual

        # 简单的轮询负载均衡
        index = (self.current_vit_index + 1) % len(self.remote_vit_instances)
        self.current_vit_index = index
        return self.remote_vit_instances[index]

    async def send_to_vit(self, data, protocol=pickle.HIGHEST_PROTOCOL):
        """
        发送数据到VIT实例，支持本地和远程模式
        """
        instance = self._get_vit_instance()
        try:
            instance.send_pyobj(data, protocol=protocol)
        except Exception as e:
            logger.error(f"Failed to send to VIT instance {instance.host_ip_port}: {e}")
            raise Exception(f"Failed to send to VIT instance {instance.host_ip_port}: {e}")

    async def vit_handle_loop(self):
        print("vit_handle_loop", "fdakpgdakgjadpgkjadk")
        while True:
            try:
                id_to_vit_obj = await self._get_vit_objs()
                logger.info(f"get vit_objs {id_to_vit_obj}")
                for id, remote_instance in self.remote_vit_instances.items():
                    if id not in id_to_vit_obj:
                        try:
                            remote_instance[id].close()
                        except:
                            pass
                        self.remote_vit_instances.pop(id)
                        logger.info(f"remote vit {id} closed")

                for id, vit_obj in id_to_vit_obj.items():
                    if id not in self.remote_vit_instances:
                        self.remote_vit_instances[id] = self.context.socket(zmq.PUSH)
                        self.remote_vit_instances[id].connect(
                            f"tcp://{vit_obj.host_ip_port}:{self.args.remote_vit_port}"
                        )
                await asyncio.sleep(30)
            except Exception as e:
                logger.exception(str(e))
                await asyncio.sleep(10)

    async def _get_vit_objs(self) -> Optional[Dict[int, VIT_Obj]]:
        """
        get_vit_objs 主要负责从 config_server 获取所有的vit远程服务。
        """
        # 使用 config_server 服务来发现所有的 pd_master 节点。
        uri = f"ws://{self.args.config_server_host}:{self.args.config_server_port}/registered_visual_objects"
        print("uri", uri)
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(uri)
                if response.status_code == 200:
                    base64data = response.json()["data"]
                    id_to_vit_obj = pickle.loads(base64.b64decode(base64data))
                    return id_to_vit_obj
                else:
                    logger.error(f"get pd_master_objs error {response.status_code}")
                    return None
        except Exception as e:
            logger.exception(str(e))
            await asyncio.sleep(10)
            return None
