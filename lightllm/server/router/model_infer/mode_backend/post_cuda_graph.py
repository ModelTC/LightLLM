import os
import torch
import copy
import bisect
from typing import Optional
from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.distributed import dist_group_manager, lightllm_capture_graph, CustomProcessGroup


logger = init_logger(__name__)


class PostCudaGraph:
    # CudaGraph forward pass for the decoding stage.

    def __init__(self):
        self.args = get_env_start_args()
        self.graph = {}
        self.mempool = torch.cuda.graph_pool_handle() if torch.cuda.is_available() else None
        self.mtp_step = self.args.mtp_step
        # self.max_batch_size = self.args.graph_max_batch_size
        self.max_batch_size = 8

        batch_sizes = [i * (self.mtp_step + 1) for i in range(1, self.max_batch_size + 1)]

        self.cuda_graph_batch_sizes = batch_sizes
        assert batch_sizes[-1] == self.max_batch_size
        logger.info(f"post cuda graph batch_sizes: {self.cuda_graph_batch_sizes}")

    def need_capture(self, batch_size):
        return batch_size not in self.graph

    def capture_post_process(
        self, post_process_func, logits: torch.Tensor, b_req_idx: torch.Tensor, b_mtp_index: torch.Tensor
    ):
        graph_obj = torch.cuda.CUDAGraph()
        batch_size = logits.shape[0]
        logger.info(f"I'm capturing post process for batch size: {batch_size}")

        with torch.cuda.graph(graph_obj, pool=self.mempool):
            model_output = post_process_func(logits, b_req_idx, b_mtp_index)
        self.graph[batch_size] = (graph_obj, logits, b_req_idx, b_mtp_index, model_output)
        graph_obj.replay()
        return model_output

    def replay(self, logits: torch.Tensor, b_req_idx: torch.Tensor, b_mtp_index: torch.Tensor):
        batch_size = logits.shape[0]
        graph_obj, graph_logits, graph_b_req_idx, graph_b_mtp_index, graph_output = self.graph[batch_size]
        graph_logits.copy_(logits, non_blocking=True)
        graph_b_req_idx.copy_(b_req_idx, non_blocking=True)
        graph_b_mtp_index.copy_(b_mtp_index, non_blocking=True)
        graph_obj.replay()
        return graph_output
