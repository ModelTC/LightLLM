import os
import torch
import copy
from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.distributed import dist_group_manager, lightllm_capture_graph, CustomProcessGroup
from lightllm.common.basemodel.microbatch_overlap_objs import DecodeMicroBatch

logger = init_logger(__name__)


class CudaGraph:
    # CudaGraph forward pass for the decoding stage.

    def __init__(self, max_batch_size=8, max_len_in_batch=8192):
        self.graph = {}
        self.mempool = torch.cuda.graph_pool_handle() if torch.cuda.is_available() else None
        self.max_batch_size = max_batch_size
        self.graph_max_len_in_batch = max_len_in_batch
        self.enable_decode_microbatch_overlap = get_env_start_args().enable_decode_microbatch_overlap

    def can_run(self, batch_size, max_len_in_batch):
        return batch_size <= self.max_batch_size and max_len_in_batch <= self.graph_max_len_in_batch

    def need_capture(self, batch_size):
        return batch_size not in self.graph

    def _capture_decode(self, decode_func, input_ids, infer_state):
        dist_group: CustomProcessGroup = infer_state.dist_group
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
            decode_func(input_ids, copy.copy(infer_state))
            torch.cuda.synchronize()

        with lightllm_capture_graph(dist_group):
            with torch.cuda.graph(graph_obj, pool=self.mempool):
                predict_logics = decode_func(input_ids, infer_state)
        self.graph[batch_size] = (graph_obj, input_ids, infer_state, predict_logics)
        graph_obj.replay()
        return predict_logics

    def _capture_decode_overlap(self, decode_func, input_ids, infer_state, input_ids1, infer_state1):
        dist_group: CustomProcessGroup = infer_state.dist_group
        dist_group1 = infer_state1.dist_group
        graph_obj = torch.cuda.CUDAGraph()
        batch_size = input_ids.shape[0]
        infer_state.max_len_in_batch = self.graph_max_len_in_batch
        infer_state.total_token_num = self.graph_max_len_in_batch * batch_size
        infer_state1.max_len_in_batch = self.graph_max_len_in_batch
        infer_state1.total_token_num = self.graph_max_len_in_batch * batch_size
        # warmup
        for _ in range(1):
            torch.cuda.synchronize()
            decode_func(input_ids, copy.copy(infer_state), input_ids1, copy.copy(infer_state1))
            torch.cuda.synchronize()
        with lightllm_capture_graph(dist_group1):
            with lightllm_capture_graph(dist_group):
                with torch.cuda.graph(graph_obj, pool=self.mempool):
                    predict_logics, predict_logics1 = decode_func(input_ids, infer_state, input_ids1, infer_state1)
        self.graph[batch_size] = (
            graph_obj,
            input_ids,
            infer_state,
            input_ids1,
            infer_state1,
            predict_logics,
            predict_logics1,
        )
        graph_obj.replay()
        return predict_logics, predict_logics1

    def capture_decode(self, decode_func, input_ids, infer_state, input_ids1=None, infer_state1=None):
        """
        Capture the cuda graph for the decoding stage.
        input_ids1 and infer_state1 is used for the overlap.
        """
        if self.enable_decode_microbatch_overlap:
            return self._capture_decode_overlap(decode_func, input_ids, infer_state, input_ids1, infer_state1)
        else:
            assert input_ids1 is None and infer_state1 is None
            return self._capture_decode(decode_func, input_ids, infer_state)

    def _replay(self, input_ids, infer_state):
        batch_size = input_ids.shape[0]
        graph_obj, graph_input_ids, graph_infer_state, graph_predict_logics = self.graph[batch_size]
        graph_input_ids.copy_(input_ids)
        graph_infer_state.copy_for_cuda_graph(infer_state)
        graph_obj.replay()
        return graph_predict_logics

    def _replay_overlap(self, input_ids, infer_state, input_ids1, infer_state1):
        batch_size = input_ids.shape[0]
        (
            graph_obj,
            graph_input_ids,
            graph_infer_state,
            graph_input_ids1,
            graph_infer_state1,
            graph_predict_logics,
            graph_predict_logics1,
        ) = self.graph[batch_size]
        graph_input_ids.copy_(input_ids)
        graph_infer_state.copy_for_cuda_graph(infer_state)
        graph_input_ids1.copy_(input_ids1)
        graph_infer_state1.copy_for_cuda_graph(infer_state1)
        graph_obj.replay()
        return graph_predict_logics, graph_predict_logics1

    def replay(self, input_ids, infer_state, input_ids1=None, infer_state1=None):
        if self.enable_decode_microbatch_overlap:
            return self._replay_overlap(input_ids, infer_state, input_ids1, infer_state1)
        else:
            assert input_ids1 is None and infer_state1 is None
            return self._replay(input_ids, infer_state)

    @torch.no_grad()
    def warmup(self, model):
        logger.info("Begin capture cudagraph, use the --disable_cudagraph to disable it.")
        for batch_size in range(self.max_batch_size, 0, -1):
            # dummy prefill
            prefill_input_len = 1
            dummy_input_ids = torch.ones((batch_size,), dtype=torch.int32, device="cuda")
            b_req_idx = torch.tensor(
                [model.req_manager.alloc() for _ in range(batch_size)], dtype=torch.int32, device="cuda"
            )
            mem_indexes = model.mem_manager.alloc(len(dummy_input_ids)).cuda()
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
            mem_indexes = model.mem_manager.alloc(len(predict_ids)).cuda()
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

    @torch.no_grad()
    def warmup_overlap(self, model):
        logger.info("Begin capture overlap cudagraph, use the --disable_cudagraph to disable it.")
        for batch_size in range(self.max_batch_size, 0, -1):
            decode_batches = []
            for micro_batch_index in [0, 1]:
                # dummy prefill
                prefill_input_len = 1
                dummy_input_ids = torch.ones((batch_size,), dtype=torch.int32, device="cuda")
                b_req_idx = torch.tensor(
                    [model.req_manager.alloc() for _ in range(batch_size)], dtype=torch.int32, device="cuda"
                )
                mem_indexes = model.mem_manager.alloc(len(dummy_input_ids)).cuda()
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
                mem_indexes = model.mem_manager.alloc(len(predict_ids)).cuda()

                micro_batch = DecodeMicroBatch(
                    batch_size=batch_size,
                    total_token_num=total_token_num,
                    max_len_in_batch=prefill_input_len + 1,
                    input_ids=torch.from_numpy(predict_ids).cuda().reshape(-1),
                    mem_indexes=mem_indexes,
                    b_req_idx=b_req_idx,
                    b_start_loc=b_start_loc,
                    b_seq_len=b_seq_len,
                )
                decode_batches.append(micro_batch)

                for var_name, var_value in list(locals().items()):
                    if isinstance(var_value, torch.Tensor):
                        del locals()[var_name]
                torch.cuda.empty_cache()
            _, _ = model.microbatch_overlap_decode(decode_batches[0], decode_batches[1])

            model.mem_manager.free_all()
            model.req_manager.free_all()

            # release local tensors
            for var_name, var_value in list(locals().items()):
                if isinstance(var_value, torch.Tensor):
                    del locals()[var_name]
            torch.cuda.empty_cache()

        logger.info(
            f"Capture overlap cudagraph success, batch_size <={self.max_batch_size} "
            f"and max_len_in_batch <= {self.graph_max_len_in_batch} will infer with cudagraph."
        )
