import os
import torch
import copy
from lightllm.utils.log_utils import init_logger
from lightllm.distributed import custom_comm_ops

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
        # 因为有些推理过程的代码，会通过判断infer_state中是否存在某些属性来在一层上
        # 做一些初始化的操作，后续层可以复用这些计算的结果，如
        # lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding.py
        # 中做的一些操作，所以在 warmup 的时候，需要调用infer_state的copy函数做一个
        # 浅拷贝，不然后续传入到cuda graph捕获过程中后，infer_state因为提前拥有了这些属性，
        # 导致不会重新初始化，这样捕获过程中会不能捕获这些临时添加到 infer_state 管理对象
        # 中的 tensor。
        for _ in range(1):
            torch.cuda.synchronize()
            decode_func(input_ids, copy.copy(infer_state))  # infer_state must copy()
            torch.cuda.synchronize()

        with custom_comm_ops.lightllm_capture_graph():
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

    @torch.no_grad()
    def warmup(self, model):
        logger.info("Begin capture cudagraph, use the --disable_cudagraph to disable it.")
        for batch_size in range(self.max_batch_size, 0, -1):
            # dummy prefill
            prefill_input_len = 1
            dummy_input_ids = torch.ones((batch_size,), dtype=torch.int32, device="cuda")
            b_req_idx = model.req_manager.alloc(batch_size).int()
            mem_indexes = model.mem_manager.alloc(len(dummy_input_ids))
            b_seq_len = torch.ones(batch_size, dtype=torch.int32, device="cuda")
            b_ready_cache_len = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
            b_start_loc = torch.arange(0, batch_size, dtype=torch.int32, device="cuda")
            total_token_num = prefill_input_len * batch_size
            logics = model.forward(
                batch_size,
                total_token_num,
                prefill_input_len,
                dummy_input_ids,
                mem_indexes,
                b_req_idx,
                b_start_loc,
                b_seq_len,
                b_ready_cache_len=b_ready_cache_len,
                is_prefill=True,
                multimodal_params=[],
            )
            mem_indexes = None
            prob_out = torch.softmax(logics, dim=-1)
            logics = None
            predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
            prob_out = None
            predict_ids = predict_ids.detach().cpu().numpy()
            torch.cuda.empty_cache()

            # dummy decoding, capture the cudagraph
            b_start_loc = b_start_loc + torch.arange(0, batch_size, dtype=torch.int32, device="cuda")
            total_token_num += batch_size
            b_seq_len += 1
            mem_indexes = model.mem_manager.alloc(len(predict_ids))
            logics = model.forward(
                batch_size,
                total_token_num,
                prefill_input_len + 1,
                torch.from_numpy(predict_ids).cuda().reshape(-1),
                mem_indexes,
                b_req_idx,
                b_start_loc,
                b_seq_len,
                is_prefill=False,
            )
            mem_indexes = None
            model.mem_manager.free_all()
            model.req_manager.free_all()
            # release local tensors
            for var_name, var_value in list(locals().items()):
                if isinstance(var_value, torch.Tensor):
                    del locals()[var_name]
            torch.cuda.empty_cache()
        logger.info(
            f"Capture cudagraph success, batch_size <={self.max_batch_size} "
            f"and max_len_in_batch <= {self.graph_max_len_in_batch} will infer with cudagraph."
        )
