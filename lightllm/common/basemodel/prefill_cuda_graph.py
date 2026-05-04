import os
import torch
import copy
import bisect
from typing import List, Tuple
from typing import Optional
from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.tensor_utils import tensor_to_no_ref_tensor
from lightllm.distributed import dist_group_manager, lightllm_capture_graph, CustomProcessGroup
from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from .infer_struct import InferStateInfo
from .cuda_graph import CudaGraph

logger = init_logger(__name__)


class PrefillCudaGraph:
    # CudaGraph forward pass for the decoding stage.

    def __init__(self, decode_cuda_graph: CudaGraph):
        self.graph = {}

        if decode_cuda_graph is not None:
            self.mempool = decode_cuda_graph.mempool  # prefill 和 decode 共享一个 mempool
        else:
            self.mempool = torch.cuda.graph_pool_handle() if torch.cuda.is_available() else None

        self.args = get_env_start_args()
        self.enable_prefill_microbatch_overlap = self.args.enable_prefill_microbatch_overlap
        self.max_handle_token_num = self.args.prefll_cudagraph_max_handle_token

        graph_handle_token_nums = []
        for i in range(2048):
            token_num = int(2 ** (2 * i))
            if 1 < token_num < self.max_handle_token_num:
                graph_handle_token_nums.append(token_num)
        graph_handle_token_nums.append(self.max_handle_token_num)
        self.graph_handle_token_nums = graph_handle_token_nums
        logger.info(f"prefill cuda graph graph_handle_token_nums: {self.graph_handle_token_nums}")

    def can_run(self, handle_token_num: int):
        return handle_token_num <= self.max_handle_token_num

    def need_capture(self, handle_token_num: int):
        finded_handle_token_num = self.find_closest_graph_handle_token_num(handle_token_num=handle_token_num)
        if finded_handle_token_num is not None:
            return finded_handle_token_num not in self.graph
        else:
            assert False, "dead code"

    def find_closest_graph_handle_token_num(self, handle_token_num: int):
        index = bisect.bisect_left(self.graph_handle_token_nums, handle_token_num)
        if index < len(self.graph_handle_token_nums):
            find_handle_token_num = self.graph_handle_token_nums[index]
            return find_handle_token_num
        else:
            return None

    def _capture_prefill(
        self, prefill_func, input_tensors: List[torch.Tensor], infer_state: InferStateInfo
    ) -> List[torch.Tensor]:
        handle_token_num = infer_state.total_token_num - infer_state.prefix_total_token_num
        dist_group: CustomProcessGroup = infer_state.dist_group
        with lightllm_capture_graph(dist_group):
            infer_state.mem_pool = self.mempool
            infer_state.prefill_cuda_graph_create_graph_obj()
            infer_state.prefill_cuda_graph_get_current_capture_graph().__enter__()
            graph_input_tensors: List[torch.Tensor] = [torch.empty_like(e) for e in input_tensors]
            graph_out_tensors: List[torch.Tensor] = prefill_func(graph_input_tensors, infer_state)
            graph_out_tensors = [e.contiguous() for e in graph_out_tensors]
            infer_state.prefill_cuda_graph_get_current_capture_graph().__exit__(None, None, None)

        graph_input_tensors = [tensor_to_no_ref_tensor(e) for e in graph_input_tensors]
        graph_out_tensors = [tensor_to_no_ref_tensor(e) for e in graph_out_tensors]

        self.graph[handle_token_num] = (infer_state, graph_input_tensors, graph_out_tensors)
        self.replay(input_tensors, infer_state)

        return graph_out_tensors

    def _capture_prefill_overlap(
        self,
        prefill_func,
        input_tensors: List[torch.Tensor],
        infer_state: InferStateInfo,
        input_tensors1: List[torch.Tensor],
        infer_state1: InferStateInfo,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        # TODO
        raise NotImplementedError("not impl")

    def capture_prefill(
        self,
        prefill_func,
        input_tensors: List[torch.Tensor],
        infer_state: InferStateInfo,
        input_tensors1: Optional[List[torch.Tensor]] = None,
        infer_state1: Optional[InferStateInfo] = None,
    ):
        """
        Capture the cuda graph for the prefill stage.
        input_tensor1 and infer_state1 is used for the overlap.
        """
        if self.enable_prefill_microbatch_overlap:
            return self._capture_prefill_overlap(
                prefill_func=prefill_func,
                input_tensors=input_tensors,
                infer_state=infer_state,
                input_tensors1=input_tensors1,
                infer_state1=infer_state1,
            )
        else:
            assert input_tensors1 is None and infer_state1 is None
            return self._capture_prefill(
                prefill_func=prefill_func, input_tensors=input_tensors, infer_state=infer_state
            )

    def _replay(self, input_tensors: List[torch.Tensor], infer_state: InferStateInfo) -> List[torch.Tensor]:
        handle_token_num = infer_state.total_token_num - infer_state.prefix_total_token_num
        graph_infer_state, graph_input_tensors, graph_output_tensors = self.graph[handle_token_num]
        graph_infer_state: InferStateInfo = graph_infer_state
        for graph_in_tensor, in_tensor in zip(graph_input_tensors, input_tensors):
            graph_in_tensor.copy_(in_tensor)

        graph_infer_state.copy_for_prefill_cuda_graph(new_infer_state=infer_state)
        graph_infer_state.prefill_replay(infer_state)

        return graph_output_tensors

    def _replay_overlap(
        self,
        input_tensors: List[torch.Tensor],
        infer_state: InferStateInfo,
        input_tensors1: List[torch.Tensor],
        infer_state1: InferStateInfo,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        raise NotImplementedError("not impl")

    def replay(self, input_tensors, infer_state, input_tensor1=None, infer_state1=None):
        if self.enable_prefill_microbatch_overlap:
            return self._replay_overlap(input_tensors, infer_state, input_tensor1, infer_state1)
        else:
            assert input_tensor1 is None and infer_state1 is None
            return self._replay(input_tensors, infer_state)

    @torch.no_grad()
    def warmup(self, model):
        logger.info("Begin capture prefill cudagraph, remove the --enable_prefill_cudagraph to disable it.")
        # for typing easy
        from .basemodel import TpPartBaseModel

        model: TpPartBaseModel = model

        # prefill cuda graph init
        for handle_token_num in self.graph_handle_token_nums[::-1]:
            logger.info(f"Capture prefill cudagraph, handle_token_num: {handle_token_num}")
            total_token_num = handle_token_num
            input_ids = torch.tensor([1 for _ in range(total_token_num)], dtype=torch.int32, device="cuda")
            mem_indexes = model.mem_manager.alloc(len(input_ids)).cuda()
            b_req_idx = torch.tensor([model.req_manager.HOLD_REQUEST_ID], dtype=torch.int32, device="cuda")
            b_seq_len = torch.empty(1, dtype=torch.int32, device="cuda")
            b_seq_len.fill_(total_token_num)
            b_mtp_index = torch.zeros(1, dtype=torch.int32, device="cuda")
            b_ready_cache_len = torch.zeros(1, dtype=torch.int32, device="cuda")
            b_prefill_start_loc = torch.zeros(1, dtype=torch.int32, device="cuda")

            model_input = ModelInput(
                batch_size=1,
                total_token_num=total_token_num,
                max_q_seq_len=total_token_num,
                max_kv_seq_len=total_token_num,
                max_cache_len=0,
                input_ids=input_ids,
                mem_indexes=mem_indexes,
                b_req_idx=b_req_idx,
                b_mtp_index=b_mtp_index,
                b_seq_len=b_seq_len,
                b_ready_cache_len=b_ready_cache_len,
                b_prefill_start_loc=b_prefill_start_loc,
                is_prefill=True,
                b_prefill_has_output_cpu=[False],
                prefix_total_token_num=0,
                multimodal_params=[{"images": [], "audios": []}],
                **model._gen_special_model_input(token_num=total_token_num),
            )
            model_output: ModelOutput = model.forward(model_input)
            del model_output
            del input_ids
            del mem_indexes
            del b_req_idx
            del b_seq_len

            model.mem_manager.free_all()
            model.req_manager.free_all()
            # release local tensors
            for var_name, var_value in list(locals().items()):
                if isinstance(var_value, torch.Tensor):
                    del locals()[var_name]
            torch.cuda.empty_cache()

        logger.info(
            f"Capture repfill cudagraph success, token_num <={self.max_handle_token_num} " f"will infer with cudagraph."
        )

    @torch.no_grad()
    def warmup_overlap(self, model):
        logger.info("Begin capture prefill overlap cudagraph, remove the --enable_prefill_cudagraph to disable it.")
        # for typing easy
        from .basemodel import TpPartBaseModel

        model: TpPartBaseModel = model

        for handle_token_num in self.graph_handle_token_nums[::-1]:
            prefill_batches = []
            for micro_batch_index in [0, 1]:
                # dummy prefill, capture the cudagraph
                total_token_num = handle_token_num
                input_ids = torch.tensor([1 for _ in range(total_token_num)], dtype=torch.int32, device="cuda")
                mem_indexes = model.mem_manager.alloc(len(input_ids)).cuda()
                b_req_idx = torch.tensor([model.req_manager.HOLD_REQUEST_ID], dtype=torch.int32, device="cuda")
                b_seq_len = torch.empty(1, dtype=torch.int32, device="cuda")
                b_seq_len.fill_(total_token_num)
                b_mtp_index = torch.zeros(1, dtype=torch.int32, device="cuda")
                b_ready_cache_len = torch.zeros(1, dtype=torch.int32, device="cuda")
                b_prefill_start_loc = torch.zeros(1, dtype=torch.int32, device="cuda")

                micro_batch = ModelInput(
                    batch_size=1,
                    total_token_num=total_token_num,
                    max_q_seq_len=total_token_num,
                    max_kv_seq_len=total_token_num,
                    max_cache_len=0,
                    input_ids=input_ids,
                    mem_indexes=mem_indexes,
                    b_req_idx=b_req_idx,
                    b_mtp_index=b_mtp_index,
                    b_seq_len=b_seq_len,
                    b_ready_cache_len=b_ready_cache_len,
                    b_prefill_start_loc=b_prefill_start_loc,
                    is_prefill=True,
                    b_prefill_has_output_cpu=[False],
                    prefix_total_token_num=0,
                    multimodal_params=[{"images": [], "audios": []}],
                    **model._gen_special_model_input(token_num=total_token_num),
                )

                prefill_batches.append(micro_batch)
                del micro_batch

                for var_name, var_value in list(locals().items()):
                    if isinstance(var_value, torch.Tensor):
                        del locals()[var_name]
                torch.cuda.empty_cache()

            _, _ = model.microbatch_overlap_prefill(prefill_batches[0], prefill_batches[1])

            model.mem_manager.free_all()
            model.req_manager.free_all()

            del prefill_batches

            # release local tensors
            for var_name, var_value in list(locals().items()):
                if isinstance(var_value, torch.Tensor):
                    del locals()[var_name]
            torch.cuda.empty_cache()

        logger.info(
            f"Capture overlap cudagraph success, handle_token_num <={self.max_handle_token_num} "
            f" will infer with cudagraph."
        )
