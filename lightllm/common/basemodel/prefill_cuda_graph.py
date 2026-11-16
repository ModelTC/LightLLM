import bisect
from threading import Lock
from typing import List, Optional, Tuple

import torch
import triton

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.platform import get_device_runtime
from lightllm.common.attention.api.graph import TensorCopyPlan, tensor_input_copy
from lightllm.common.execution import ExecutionScope
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger
from lightllm.utils.tensor_utils import tensor_to_no_ref_tensor

from .cuda_graph import CudaGraph
from .infer_struct import InferStateInfo

logger = init_logger(__name__)


class PrefillGraphExecution:
    """分段图拥有输入/输出和 collector；attention 仍通过段间回调使用本批次 Plan。"""

    def __init__(self, infer_state, inputs, outputs, collector, runtime, execution):
        self.state, self.inputs, self.outputs, self.collector = infer_state, inputs, outputs, collector
        self.runtime = runtime
        self.execution = execution
        self.binding = infer_state.attention.binding
        self.generation = infer_state.att_state_manager.storage_generation
        self.invalid = False
        self.lock = Lock()
        self.copy_plan = TensorCopyPlan(
            [target for target, _ in infer_state.prefill_input_copies(infer_state)] + list(inputs)
        )
        self.manifest = infer_state.capture_manifest(segmented=True)
        self.manifest.add("segment.inputs", "copy", inputs)
        self.manifest.add("segment.outputs", "generated", outputs)
        self.manifest.add("collector", "generated", collector)
        infer_state.att_state_manager.register_graph(self)

    def record_output_use(self, event):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("cannot register an output reader during prefill Graph submission")
        try:
            if self.state is None or self.invalid:
                raise RuntimeError("prefill Graph execution is closed or invalid")
            self.execution.record_reader(event)
        finally:
            self.lock.release()

    def close(self):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("cannot close prefill Graph during an active submission")
        try:
            if self.state is None:
                return
            with self.execution.join():
                self.state.close_prefill_graphs()
                self.state = self.collector = self.copy_plan = self.manifest = self.binding = None
                self.inputs = self.outputs = ()
        finally:
            self.lock.release()


class PrefillCudaGraph:
    # CudaGraph forward pass for the decoding stage.

    def __init__(self, decode_cuda_graph: CudaGraph, tp_world_size: int, runtime=None, execution=None):
        self.graph = {}
        self.sealed = False
        self.runtime = runtime or (decode_cuda_graph.runtime if decode_cuda_graph is not None else get_device_runtime())
        self.tp_world_size = tp_world_size
        if decode_cuda_graph is not None:
            self.mempool = decode_cuda_graph.mempool  # prefill 和 decode 共享一个 mempool
        else:
            self.mempool = self.runtime.graph_pool()
        self.execution = (
            decode_cuda_graph.execution
            if decode_cuda_graph is not None
            else execution
            if execution is not None
            else ExecutionScope(self.runtime)
        )

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
        logger.info(f"prefill cuda graph graph_handle_token_nums: {self.graph_handle_token_nums}")

    def can_run(self, handle_token_num: int):
        if handle_token_num > self.max_handle_token_num:
            return False
        if self.sealed:
            execution = self.graph.get(self.find_closest_graph_handle_token_num(handle_token_num))
            return execution is not None and execution.state is not None and not execution.invalid
        return True

    def seal(self):
        self.sealed = True

    def need_capture(self, handle_token_num: int):
        finded_handle_token_num = self.find_closest_graph_handle_token_num(handle_token_num=handle_token_num)
        if finded_handle_token_num is not None:
            return finded_handle_token_num not in self.graph or self.graph[finded_handle_token_num].state is None
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
        with self.execution.join(retain=(infer_state, input_tensors)):
            return self._capture_prefill_body(prefill_func, input_tensors, infer_state)

    def _capture_prefill_body(self, prefill_func, input_tensors, infer_state):
        handle_token_num = infer_state.input_ids.shape[0]
        infer_state.mem_pool = self.mempool
        infer_state.device_runtime = self.runtime
        manager = infer_state.att_state_manager
        snapshot = manager.snapshot_capture_state(infer_state.b_req_idx, infer_state.att_state_role)
        restore_pending = True
        try:
            infer_state.prefill_cuda_graph_create_graph_obj()
            infer_state.prefill_cuda_graph_get_current_capture_graph().__enter__()
            graph_input_tensors: List[torch.Tensor] = [torch.empty_like(e) for e in input_tensors]
            graph_out_tensors: List[torch.Tensor] = prefill_func(graph_input_tensors, infer_state)
            graph_out_tensors = [e.contiguous() for e in graph_out_tensors]
            infer_state.prefill_cuda_graph_get_current_capture_graph().__exit__(None, None, None)
            manager.restore_capture_state(snapshot)
            restore_pending = False

            graph_input_tensors = [tensor_to_no_ref_tensor(e) for e in graph_input_tensors]
            graph_out_tensors = [tensor_to_no_ref_tensor(e) for e in graph_out_tensors]

            graph_hidden_collector = infer_state.hidden_collector
            hidden_tensor_count, hidden_tensor_nbytes = graph_hidden_collector.release_graph_tensor_ownership()
            logger.info(
                f"Prefill CUDA Graph hidden collector 已完成 capture 状态托管："
                f"handle_token_num={handle_token_num}, "
                f"collector={graph_hidden_collector.__class__.__name__}, "
                f"no_ref_tensor_count={hidden_tensor_count}, "
                f"no_ref_tensor_nbytes={hidden_tensor_nbytes}。"
                f"这些 tensor 的固定地址由 graph memory pool 管理，collector 仅保留无所有权引用供 replay 使用。"
            )
            self.graph[handle_token_num] = PrefillGraphExecution(
                infer_state,
                graph_input_tensors,
                graph_out_tensors,
                graph_hidden_collector,
                self.runtime,
                self.execution,
            )
            self.replay(input_tensors, infer_state)

            return graph_out_tensors
        except BaseException:
            self.graph.pop(handle_token_num, None)
            infer_state.close_prefill_graphs()
            if restore_pending:
                manager.restore_capture_state(snapshot)
            raise

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
        if self.sealed:
            raise RuntimeError("prefill Graph admission is sealed; capture requires runtime initialization")
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
        execution = self.graph[handle_token_num]
        if not execution.lock.acquire(blocking=False):
            raise RuntimeError("prefill Graph already has an active submission")
        try:
            with self.execution.join(retain=(infer_state, input_tensors)):
                return self._replay_execution(execution, input_tensors, infer_state)
        finally:
            execution.lock.release()

    def _replay_execution(self, execution, input_tensors, infer_state):
        graph_infer_state = execution.state
        if graph_infer_state is None:
            raise RuntimeError("prefill Graph execution is closed")
        if execution.invalid:
            raise RuntimeError("prefill Graph execution is invalid after a failed submission")
        graph_infer_state.att_state_manager.check_execution_health()
        if graph_infer_state.att_state_manager is not infer_state.att_state_manager:
            raise ValueError("prefill Graph input belongs to a different attention state manager")
        if graph_infer_state.att_state_manager.storage_generation != execution.generation:
            raise RuntimeError("attention storage changed; recapture prefill Graph")
        if execution.binding is not None:
            execution.binding.check()
        if execution.binding is not infer_state.attention.binding:
            raise ValueError("prefill Graph model binding changed")
        execution.manifest.check()
        graph_input_tensors, graph_output_tensors, graph_hidden_collector = (
            execution.inputs,
            execution.outputs,
            execution.collector,
        )
        if len(graph_input_tensors) != len(input_tensors):
            raise ValueError("prefill Graph input count changed")
        copies = graph_infer_state.prefill_input_copies(infer_state)
        for index, (target, source) in enumerate(zip(graph_input_tensors, input_tensors)):
            copies.extend(tensor_input_copy(f"prefill input {index}", target, source))
        copies = execution.copy_plan.validate(copies)
        infer_state.attention.retain(input_tensors)
        try:
            for target, source in copies:
                target.copy_(source, non_blocking=True)
            if infer_state.hidden_collector is graph_hidden_collector:
                infer_state.hidden_collector = graph_hidden_collector.new_instance()
            infer_state.hidden_collector.restore_graph_state(graph_hidden_collector)
            graph_infer_state.prefill_replay(infer_state)
        except BaseException:
            execution.invalid = True
            raise

        return graph_output_tensors

    def close(self):
        for execution in self.graph.values():
            execution.close()
        self.graph.clear()

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
            input_ids = torch.tensor([1 for _ in range(total_token_num)], dtype=torch.int64, device="cuda")
            b_req_idx = torch.tensor([model.req_manager.HOLD_REQUEST_ID], dtype=torch.int32, device="cuda")
            b_seq_len = torch.empty(1, dtype=torch.int32, device="cuda")
            b_seq_len.fill_(total_token_num)
            b_mtp_index = torch.zeros(1, dtype=torch.int32, device="cuda")
            b_is_decode_req = torch.zeros(1, dtype=torch.bool, device="cuda")
            b_ready_cache_len = torch.zeros(1, dtype=torch.int32, device="cuda")
            b_prefill_start_loc = torch.zeros(1, dtype=torch.int32, device="cuda")

            model_input = ModelInput(
                batch_size=1,
                total_token_num=total_token_num,
                max_q_seq_len=total_token_num,
                max_kv_seq_len=total_token_num,
                max_cache_len=0,
                input_ids=input_ids,
                b_req_idx=b_req_idx,
                b_mtp_index=b_mtp_index,
                b_seq_len=b_seq_len,
                b_is_decode_req=b_is_decode_req,
                b_ready_cache_len=b_ready_cache_len,
                b_prefill_start_loc=b_prefill_start_loc,
                is_prefill=True,
                b_prefill_has_output_cpu=[False],
                multimodal_params=[{"images": [], "audios": []}],
                **model._gen_special_model_input(token_num=total_token_num),
            )
            model_output: ModelOutput = model.forward(model_input)
            del model_output
            del input_ids
            del b_req_idx
            del b_seq_len

            model.att_state_manager.full_att_allocator.free_all()
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
                input_ids = torch.tensor([1 for _ in range(total_token_num)], dtype=torch.int64, device="cuda")
                b_req_idx = torch.tensor([model.req_manager.HOLD_REQUEST_ID], dtype=torch.int32, device="cuda")
                b_seq_len = torch.empty(1, dtype=torch.int32, device="cuda")
                b_seq_len.fill_(total_token_num)
                b_mtp_index = torch.zeros(1, dtype=torch.int32, device="cuda")
                b_is_decode_req = torch.zeros(1, dtype=torch.bool, device="cuda")
                b_ready_cache_len = torch.zeros(1, dtype=torch.int32, device="cuda")
                b_prefill_start_loc = torch.zeros(1, dtype=torch.int32, device="cuda")

                micro_batch = ModelInput(
                    batch_size=1,
                    total_token_num=total_token_num,
                    max_q_seq_len=total_token_num,
                    max_kv_seq_len=total_token_num,
                    max_cache_len=0,
                    input_ids=input_ids,
                    b_req_idx=b_req_idx,
                    b_mtp_index=b_mtp_index,
                    b_seq_len=b_seq_len,
                    b_is_decode_req=b_is_decode_req,
                    b_ready_cache_len=b_ready_cache_len,
                    b_prefill_start_loc=b_prefill_start_loc,
                    is_prefill=True,
                    b_prefill_has_output_cpu=[False],
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

            model.att_state_manager.full_att_allocator.free_all()
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
