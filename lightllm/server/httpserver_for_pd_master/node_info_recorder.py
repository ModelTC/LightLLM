import copy

from ..pd_io_struct import PD_Client_Obj
from lightllm.server.httpserver.manager import SamplingParams, MultimodalParams
from typing import Union, List, Dict
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class NodeInfoRecorder:
    def __init__(self):
        self.prefill_node_info: dict = {}
        self.decode_node_info: dict = {}

    def register_node(self, pd_client: PD_Client_Obj):
        node_info = {
            "node_id": pd_client.node_id,
            "client_ip_port": pd_client.client_ip_port,
            "mode": pd_client.mode,
            "node": pd_client,
            "mem_len": 0,
            # "batch_size": 0,
        }
        if pd_client.mode == "prefill":
            self.prefill_node_info[pd_client.client_ip_port] = node_info
        elif pd_client.mode == "decode":
            self.decode_node_info[pd_client.client_ip_port] = node_info
        else:
            assert False, f"mode must in ['prefill', 'decode'], but get {pd_client.mode}"

    def remove_node(self, pd_client: PD_Client_Obj):
        if pd_client.mode == "prefill":
            del self.prefill_node_info[pd_client.client_ip_port]
        elif pd_client.mode == "decode":
            del self.decode_node_info[pd_client.client_ip_port]
        else:
            assert False, f"mode must in ['prefill', 'decode'], but get {pd_client.mode}"

    def update_node_load_info(self, load_info: dict):
        if "client_ip_port" in load_info:
            ip_port = load_info["client_ip_port"]
            if ip_port in self.prefill_node_info:
                self.prefill_node_info[ip_port]["mem_len"] = load_info["mem_len"]
            elif ip_port in self.decode_node_info:
                self.decode_node_info[ip_port]["mem_len"] = load_info["mem_len"]
            else:
                logger.warning(f"Received load info for unknown node: {ip_port}")
        else:
            logger.warning("Received load info without client_ip_port")


class PredictNodeInfoRecorder(NodeInfoRecorder):
    def __init__(self):
        super().__init__()
        self.prefill_predict_node_info: dict = {}
        self.decode_predict_node_info: dict = {}

    def register_node(self, pd_client: PD_Client_Obj):
        super().register_node(pd_client)
        if pd_client.mode == "prefill":
            self.prefill_predict_node_info[pd_client.client_ip_port] = copy.copy(self.prefill_node_info[pd_client.client_ip_port])
        elif pd_client.mode == "decode":
            self.decode_predict_node_info[pd_client.client_ip_port] = copy.copy(self.decode_node_info[pd_client.client_ip_port])

    def remove_node(self, pd_client: PD_Client_Obj):
        super().remove_node(pd_client)
        if pd_client.mode == "prefill":
            del self.prefill_predict_node_info[pd_client.client_ip_port]
        elif pd_client.mode == "decode":
            del self.decode_predict_node_info[pd_client.client_ip_port]

    def update_node_load_info(self, load_info: dict):
        super().update_node_load_info(load_info)
        ip_port = load_info["client_ip_port"]
        if ip_port in self.prefill_node_info:
            self.prefill_predict_node_info[ip_port] = copy.copy(self.prefill_node_info[ip_port])
        elif ip_port in self.decode_node_info:
            self.decode_predict_node_info[ip_port] = copy.copy(self.decode_node_info[ip_port])
        else:
            logger.warning(f"Received load info for unknown node: {ip_port}")

    def update_predict_node_info(self, p_node: PD_Client_Obj, d_node: PD_Client_Obj, prompt: Union[str, List[int]], sampling_params: SamplingParams, multimodal_params: MultimodalParams):
        self.prefill_predict_node_info[p_node.client_ip_port]["mem_len"] += len(prompt)
        self.decode_predict_node_info[d_node.client_ip_port]["mem_len"] += sampling_params.max_new_tokens

    def get_predict_node_infos(self) -> Dict[str, dict]:
        return {
            "prefill": self.prefill_predict_node_info,
            "decode": self.decode_predict_node_info,
        }
