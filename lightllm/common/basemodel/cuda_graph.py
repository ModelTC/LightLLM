import os
import torch
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class CudaGraph:
    # CudaGraph forward pass for the decoding stage.

    def __init__(self, max_batch_size=8, max_len_in_batch=8192):
        self.graph = {}
        self.mempool = torch.cuda.graph_pool_handle() if torch.cuda.is_available() else None
        self.max_batch_size = max_batch_size
        self.graph_max_len_in_batch = max_len_in_batch

    def can_run(self, batch_size, max_len_in_batch):
        return batch_size <= self.max_batch_size and max_len_in_batch <= self.graph_max_len_in_batch

    def need_capture(self, batch_size):
        return batch_size not in self.graph

    def capture_decode(self, decode_func, input_ids, infer_state):
        graph_obj = torch.cuda.CUDAGraph()
        batch_size = input_ids.shape[0]
        infer_state.max_len_in_batch = self.graph_max_len_in_batch
        infer_state.total_token_num = self.graph_max_len_in_batch * batch_size
        # warmup
        for _ in range(1):
            torch.cuda.synchronize()
            decode_func(input_ids, infer_state)
            torch.cuda.synchronize()
        with torch.cuda.graph(graph_obj, pool=self.mempool):
            predict_logics = decode_func(input_ids, infer_state)
        self.graph[batch_size] = (graph_obj, input_ids, infer_state, predict_logics)
        graph_obj.replay()
        return predict_logics

    def replay(self, input_ids, infer_state):
        batch_size = input_ids.shape[0]
        graph_obj, graph_input_ids, graph_infer_state, graph_predict_logics = self.graph[batch_size]
        graph_input_ids.copy_(input_ids)
        graph_infer_state.copy_for_cuda_graph(infer_state)
        graph_obj.replay()
        return graph_predict_logics

    def warmup(self, model):
        logger.info("Begin capture cudagraph, use the --disable_cudagraph to disable it.")
        for batch_size in range(self.max_batch_size, 0, -1):
            # dummy prefill
            prefill_input_len = 1
            dummy_input_ids = torch.ones((batch_size,), dtype=torch.int32, device="cuda")
            b_req_idx = model.req_manager.alloc(batch_size).int()
            b_seq_len = torch.ones(batch_size, dtype=torch.int32, device="cuda")
            b_ready_cache_len = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
            b_start_loc = torch.arange(0, batch_size, dtype=torch.int32, device="cuda")
            total_token_num = prefill_input_len * batch_size
            logics = model.forward(
                batch_size,
                total_token_num,
                prefill_input_len,
                dummy_input_ids,
                b_req_idx,
                b_start_loc,
                b_seq_len,
                b_ready_cache_len=b_ready_cache_len,
                is_prefill=True,
                multimodal_params=[],
            )
            prob_out = torch.softmax(logics, dim=-1)
            predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
            predict_ids = predict_ids.detach().cpu().numpy()

            # dummy decoding, capture the cudagraph
            b_start_loc = b_start_loc + torch.arange(0, batch_size, dtype=torch.int32, device="cuda")
            total_token_num += batch_size
            b_seq_len += 1
            logics = model.forward(
                batch_size,
                total_token_num,
                prefill_input_len + 1,
                torch.from_numpy(predict_ids).cuda().reshape(-1),
                b_req_idx,
                b_start_loc,
                b_seq_len,
                is_prefill=False,
            )
            model.mem_manager.free_all()
            model.req_manager.free_all()
        logger.info(
            f"Capture cudagraph success, batch_size <={self.max_batch_size} "
            f"and max_len_in_batch <= {self.graph_max_len_in_batch} will infer with cudagraph."
        )
