from __future__ import annotations

import torch
import bisect
import triton
from typing import List, Tuple
from typing import Optional, TYPE_CHECKING
from lightllm.utils.log_utils import init_logger, log_info_on_leader
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.core.device.cuda import CudaRuntime
from lightllm.core.runner.graph_program import GraphProgram
from lightllm.core.runner.decode_graph import CudaGraph

if TYPE_CHECKING:
    from lightllm.common.basemodel.infer_struct import InferStateInfo

logger = init_logger(__name__)


class PrefillCudaGraph:
    # CudaGraph forward pass for the decoding stage.

    def __init__(self, decode_cuda_graph: CudaGraph, tp_world_size: int, runtime=CudaRuntime):
        self.runtime = runtime
        self.graph = {}
        self.tp_world_size = tp_world_size
        if decode_cuda_graph is not None:
            self.mempool = decode_cuda_graph.mempool  # prefill 和 decode 共享一个 mempool
        else:
            self.mempool = self.runtime.graph_pool()

        self.args = get_env_start_args()
        self.enable_prefill_microbatch_overlap = self.args.enable_prefill_microbatch_overlap
        self.max_handle_token_num = self.args.prefill_cudagraph_max_handle_token
        if self.args.batch_max_tokens is not None:
            self.max_handle_token_num = min(self.max_handle_token_num, self.args.batch_max_tokens)

        graph_handle_token_nums = (
            list(range(4, 33, 4))
            + list(range(48, 257, 16))
            + list(range(288, 513, 32))
            + list(range(576, 1024 + 1, 64))
            + list(range(1280, 4096 + 1, 256))
            + list(range(4608, self.max_handle_token_num + 1, 512))
        )
        graph_handle_token_nums = [e for e in graph_handle_token_nums if e <= self.max_handle_token_num]
        graph_handle_token_nums.append(self.max_handle_token_num)

        graph_handle_token_nums = list(set[int](graph_handle_token_nums))
        graph_handle_token_nums.sort()
        if self.args.enable_tpsp_mix_mode:
            graph_handle_token_nums = [
                triton.cdiv(e, self.tp_world_size) * self.tp_world_size for e in graph_handle_token_nums
            ]
            graph_handle_token_nums = list(set(graph_handle_token_nums))
            graph_handle_token_nums.sort()

        self.graph_handle_token_nums = graph_handle_token_nums
        log_info_on_leader(logger, f"prefill cuda graph graph_handle_token_nums: {self.graph_handle_token_nums}")

    def close(self):
        for captured in self.graph.values():
            captured[0].prefill_program.close()
        self.graph.clear()
        self.mempool = None

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
        handle_token_num = infer_state.input_ids.shape[0]
        program = GraphProgram(self.mempool, self.runtime)
        infer_state.prefill_program = program
        with program.capture():
            graph_input_tensors = [torch.empty_like(e) for e in input_tensors]
            graph_out_tensors = prefill_func(graph_input_tensors, infer_state)
            graph_out_tensors = [e.contiguous() for e in graph_out_tensors]

        graph_input_tensors = [self.runtime.borrow_tensor(e) for e in graph_input_tensors]
        graph_out_tensors = [self.runtime.borrow_tensor(e) for e in graph_out_tensors]

        graph_hidden_collector = infer_state.hidden_collector
        hidden_tensor_count, hidden_tensor_nbytes = graph_hidden_collector.release_graph_tensor_ownership(
            self.runtime.borrow_tensor
        )
        log_info_on_leader(
            logger,
            f"Prefill CUDA Graph hidden collector 已完成 capture 状态托管："
            f"handle_token_num={handle_token_num}, "
            f"collector={graph_hidden_collector.__class__.__name__}, "
            f"no_ref_tensor_count={hidden_tensor_count}, "
            f"no_ref_tensor_nbytes={hidden_tensor_nbytes}。"
            f"这些 tensor 的固定地址由 graph memory pool 管理，collector 仅保留无所有权引用供 replay 使用。",
        )
        self.graph[handle_token_num] = (
            infer_state,
            graph_input_tensors,
            graph_out_tensors,
            graph_hidden_collector,
        )
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
        handle_token_num = infer_state.input_ids.shape[0]
        graph_infer_state, graph_input_tensors, graph_output_tensors, graph_hidden_collector = self.graph[
            handle_token_num
        ]
        graph_infer_state: InferStateInfo = graph_infer_state
        for graph_in_tensor, in_tensor in zip(graph_input_tensors, input_tensors):
            graph_in_tensor.copy_(in_tensor)

        graph_infer_state.copy_for_prefill_cuda_graph(new_infer_state=infer_state)
        # 首次 capture 后 replay 时，infer_state 与 graph_infer_state 是同一对象，
        # 需要先创建运行时实例，避免 finish_output 清空 graph 中保存的 capture collector。
        if infer_state.hidden_collector is graph_hidden_collector:
            infer_state.hidden_collector = graph_hidden_collector.new_instance()
        infer_state.hidden_collector.restore_graph_state(graph_hidden_collector)
        graph_infer_state.prefill_program.replay(infer_state)

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
