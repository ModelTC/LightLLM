import random
from typing import Union, List, Tuple, Dict
from lightllm.server.pd_io_struct import PD_Client_Obj
from lightllm.server.core.objs import SamplingParams
from lightllm.server.multimodal_params import MultimodalParams


class PDSelector:
    def __init__(self, pd_manager):
        self.prefill_nodes: List[PD_Client_Obj] = []
        self.decode_nodes: List[PD_Client_Obj] = []
        self.pd_manager = pd_manager

    def update_nodes(self, prefill_nodes, decode_nodes):
        self.prefill_nodes = prefill_nodes
        self.decode_nodes = decode_nodes

    def select_p_d_node(
        self, prompt: Union[str, List[int]], sampling_params: SamplingParams, multimodal_params: MultimodalParams
    ) -> Tuple[PD_Client_Obj, PD_Client_Obj]:
        raise NotImplementedError("Subclass must implement this method")


class RandomSelector(PDSelector):
    """随机选择器"""

    def select_p_d_node(
        self, prompt: Union[str, List[int]], sampling_params: SamplingParams, multimodal_params: MultimodalParams
    ) -> Tuple[PD_Client_Obj, PD_Client_Obj]:
        p_node = random.choice(self.prefill_nodes)
        d_node = random.choice(self.decode_nodes)
        return p_node, d_node


class RoundRobinSelector(PDSelector):
    """轮询选择器"""

    def __init__(self, pd_manager):
        super().__init__(pd_manager)
        self.prefill_node_index: int = 0
        self.decode_node_index: int = 0

    def select_p_d_node(
        self, prompt: Union[str, List[int]], sampling_params: SamplingParams, multimodal_params: MultimodalParams
    ) -> Tuple[PD_Client_Obj, PD_Client_Obj]:
        self.prefill_node_index = self.prefill_node_index % len(self.prefill_nodes)
        self.decode_node_index = self.decode_node_index % len(self.decode_nodes)
        p_node = self.prefill_nodes[self.prefill_node_index]
        d_node = self.decode_nodes[self.decode_node_index]
        self.prefill_node_index += 1
        self.decode_node_index += 1
        return p_node, d_node


class AdaptiveLoadSelector(PDSelector):
    """基于负载使用情况的选择器"""

    def select_p_d_node(
        self, prompt: Union[str, List[int]], sampling_params: SamplingParams, multimodal_params: MultimodalParams
    ) -> Tuple[PD_Client_Obj, PD_Client_Obj]:
        p_node = self._importance_sampling(self.prefill_nodes)
        d_node = self._importance_sampling(self.decode_nodes)

        return p_node, d_node

    def _importance_sampling(self, nodes: List[PD_Client_Obj]):
        return random.choices(nodes, weights=[max(1.0 - e.run_status.total_token_usage_rate, 0.02) for e in nodes])
