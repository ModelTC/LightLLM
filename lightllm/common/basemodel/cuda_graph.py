import os
import torch
import torch.distributed as dist
import copy
import bisect
import math
import triton
from typing import Optional
from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.distributed import dist_group_manager
from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.common.triton_utils.autotuner import Autotuner, AutotuneKernelType
from lightllm.utils.torch_memory_saver_utils import (
    TorchMemorySaverWrapper,
    MemoryTag,
)
from .infer_struct import InferStateInfo
from .graph import GraphExecution
from lightllm.platform import get_device_runtime


logger = init_logger(__name__)


class CudaGraph:
    # CudaGraph forward pass for the decoding stage.

    @staticmethod
    def gen_cuda_graph_batch_sizes(
        batch_step_size_before_split: int,
        split_batch_size: int,
        batch_step_size_after_split: int,
        max_batch_size: int,
        tp_world_size: int = 1,
    ):
        args = get_env_start_args()

        # Generate CUDA Graph batch sizes in two phases with independent steps:
        # use batch_step_size_before_split up to split_batch_size, then use
        # batch_step_size_after_split above it. For example, given
        # batch_step_size_before_split=8, split_batch_size=32,
        # batch_step_size_after_split=16, and max_batch_size=80, the result is
        # [8, 16, 24, 32, 48, 64, 80]. max_batch_size is always included.

        batch_sizes = list(range(batch_step_size_before_split, split_batch_size + 1, batch_step_size_before_split))
        batch_sizes.extend(
            range(split_batch_size + batch_step_size_after_split, max_batch_size, batch_step_size_after_split)
        )
        batch_sizes = sorted({size for size in batch_sizes if size < max_batch_size} | {max_batch_size})

        if args.enable_tpsp_mix_mode:
            # Keep complete fixed-layout MTP groups as well as TP/SP shards.
            alignment = math.lcm(batch_step_size_before_split, tp_world_size)
            batch_sizes = sorted({triton.cdiv(size, alignment) * alignment for size in batch_sizes})
        assert batch_sizes[-1] == max_batch_size
        return batch_sizes

    def __init__(
        self,
        batch_step_size_before_split: int,
        split_batch_size: int,
        batch_step_size_after_split: int,
        max_batch_size=8,
        max_len_in_batch=8192,
        tp_world_size: int = 1,
        capture_infer_cost: bool = False,
        runtime=None,
    ):
        self.graph = {}
        self.tp_world_size = tp_world_size
        self.runtime = runtime if runtime is not None else get_device_runtime()
        self.mempool = self.runtime.graph_pool()
        self.args = get_env_start_args()
        self.max_batch_size = max_batch_size
        self.graph_max_len_in_batch = max_len_in_batch
        self.enable_decode_microbatch_overlap = self.args.enable_decode_microbatch_overlap
        self.torch_memory_saver = TorchMemorySaverWrapper(self.args.enable_torch_memory_saver)
        self.capture_infer_cost = capture_infer_cost
        self.infer_cost_ms_by_batch_size = {}

        self.cuda_graph_batch_sizes = self.gen_cuda_graph_batch_sizes(
            batch_step_size_before_split=batch_step_size_before_split,
            split_batch_size=split_batch_size,
            batch_step_size_after_split=batch_step_size_after_split,
            max_batch_size=self.max_batch_size,
            tp_world_size=self.tp_world_size,
        )
        logger.info(f"cuda graph batch_sizes: {self.cuda_graph_batch_sizes}")

    def can_run(self, batch_size, max_len_in_batch):
        return batch_size <= self.max_batch_size and max_len_in_batch <= self.graph_max_len_in_batch

    def need_capture(self, batch_size):
        find_batch_size = self.find_closest_graph_batch_size(batch_size)
        if find_batch_size is not None:
            return find_batch_size not in self.graph or self.graph[find_batch_size].graph is None
        else:
            assert False, "dead code"

    def find_closest_graph_batch_size(self, batch_size):
        index = bisect.bisect_left(self.cuda_graph_batch_sizes, batch_size)
        if index < len(self.cuda_graph_batch_sizes):
            find_batch_size = self.cuda_graph_batch_sizes[index]
            return find_batch_size
        else:
            return None

    def _capture_decode(self, decode_func, infer_state: InferStateInfo):
        graph_obj = self.runtime.create_graph()
        input_ids = infer_state.input_ids
        batch_size = input_ids.shape[0]
        infer_state.max_kv_seq_len = self.graph_max_len_in_batch
        infer_state.total_token_num = self.graph_max_len_in_batch * batch_size
        snapshot = infer_state.att_state_manager.snapshot_capture_state(
            infer_state.b_req_idx, infer_state.att_state_role
        )
        restore_state = lambda: infer_state.att_state_manager.restore_capture_state(snapshot)
        try:
            self.runtime.synchronize()
            with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
                decode_func(copy.copy(infer_state))
            self.runtime.synchronize()
        finally:
            restore_state()

        try:
            with self.runtime.capture_graph(graph_obj, pool=self.mempool, memory_saver=self.torch_memory_saver):
                model_output = decode_func(infer_state)
        except BaseException:
            try:
                GraphExecution(graph_obj, (infer_state,), runtime=self.runtime).close()
            finally:
                restore_state()
            raise
        try:
            execution = GraphExecution(graph_obj, (infer_state,), (model_output,), runtime=self.runtime)
        except BaseException:
            restore_state()
            raise
        self.graph[batch_size] = execution
        execution.replay()
        self._measure_replay_cost(graph_obj=execution, batch_size=batch_size, restore_state=restore_state)
        return model_output

    def _capture_decode_overlap(
        self,
        decode_func,
        infer_state: InferStateInfo,
        infer_state1: InferStateInfo,
    ):
        graph_obj = self.runtime.create_graph()
        input_ids = infer_state.input_ids
        batch_size = input_ids.shape[0]
        infer_state.max_kv_seq_len = self.graph_max_len_in_batch
        infer_state.total_token_num = self.graph_max_len_in_batch * batch_size
        infer_state1.max_kv_seq_len = self.graph_max_len_in_batch
        infer_state1.total_token_num = self.graph_max_len_in_batch * batch_size
        snapshot = infer_state.att_state_manager.snapshot_capture_state(
            torch.cat((infer_state.b_req_idx, infer_state1.b_req_idx)), infer_state.att_state_role
        )
        restore_state = lambda: infer_state.att_state_manager.restore_capture_state(snapshot)
        try:
            self.runtime.synchronize()
            with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
                decode_func(copy.copy(infer_state), copy.copy(infer_state1))
            self.runtime.synchronize()
        finally:
            restore_state()

        try:
            with self.runtime.capture_graph(graph_obj, pool=self.mempool, memory_saver=self.torch_memory_saver):
                model_output, model_output1 = decode_func(infer_state, infer_state1)
        except BaseException:
            try:
                GraphExecution(graph_obj, (infer_state, infer_state1), runtime=self.runtime).close()
            finally:
                restore_state()
            raise
        try:
            execution = GraphExecution(
                graph_obj, (infer_state, infer_state1), (model_output, model_output1), runtime=self.runtime
            )
        except BaseException:
            restore_state()
            raise
        self.graph[batch_size] = execution
        execution.replay()
        self._measure_replay_cost(graph_obj=execution, batch_size=batch_size, restore_state=restore_state)
        return model_output, model_output1

    def _measure_replay_cost(
        self, graph_obj: torch.cuda.CUDAGraph, batch_size: int, restore_state=lambda: None
    ) -> None:
        if not self.capture_infer_cost:
            return

        dist.barrier(group=dist.group.WORLD)
        start_event = self.runtime.create_event(enable_timing=True)
        end_event = self.runtime.create_event(enable_timing=True)
        restore_state()
        graph_obj.replay()
        restore_state()
        start_event.record()
        graph_obj.replay()
        end_event.record()
        end_event.synchronize()
        infer_cost_ms_tensor = torch.tensor(
            [start_event.elapsed_time(end_event)],
            dtype=torch.float32,
            device="cuda",
        )
        dist.all_reduce(infer_cost_ms_tensor, op=dist.ReduceOp.MIN, group=dist.group.WORLD)
        if self.enable_decode_microbatch_overlap:
            # overlap graph 每次 replay 同时处理两个等容量 microbatch。
            batch_size *= 2
        self.infer_cost_ms_by_batch_size[batch_size] = float(infer_cost_ms_tensor.item())

    def capture_decode(
        self,
        decode_func,
        infer_state: InferStateInfo,
        infer_state1: Optional[InferStateInfo] = None,
    ):
        """
        Capture the cuda graph for the decoding stage.
        input_ids1 and infer_state1 is used for the overlap.
        """
        if self.enable_decode_microbatch_overlap:
            return self._capture_decode_overlap(decode_func, infer_state, infer_state1)
        else:
            assert infer_state1 is None
            return self._capture_decode(decode_func, infer_state)

    def replay(self, infer_state, infer_state1=None):
        batch_size = infer_state.input_ids.shape[0]
        states = (infer_state, infer_state1) if self.enable_decode_microbatch_overlap else (infer_state,)
        if any(state is None for state in states):
            raise ValueError("Graph replay requires all microbatch lanes")
        return self.graph[batch_size].replay(*states)

    def close(self):
        for execution in self.graph.values():
            execution.close()
        self.graph.clear()

    @torch.no_grad()
    def warmup(self, model):
        logger.info("Begin capture cudagraph, use the --disable_cudagraph to disable it.")
        # for typing easy
        from .basemodel import TpPartBaseModel

        model: TpPartBaseModel = model
        # decode cuda graph init
        for batch_size in self.cuda_graph_batch_sizes[::-1]:
            seq_len = 2
            total_token_num = batch_size * seq_len
            max_len_in_batch = self.graph_max_len_in_batch
            input_ids = torch.tensor([1 for _ in range(batch_size)], dtype=torch.int64, device="cuda")
            b_req_idx = torch.tensor(
                [model.req_manager.HOLD_REQUEST_ID for _ in range(batch_size)], dtype=torch.int32, device="cuda"
            )
            b_seq_len = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda")
            b_mtp_index = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
            b_shared_seq_len = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
            b_shared_radix_node_id = torch.full((batch_size,), -1, dtype=torch.int64, device="cuda")

            model_input = ModelInput(
                batch_size=batch_size,
                total_token_num=total_token_num,
                max_q_seq_len=1,
                max_kv_seq_len=max_len_in_batch,
                input_ids=input_ids,
                b_req_idx=b_req_idx,
                b_seq_len=b_seq_len,
                b_mtp_index=b_mtp_index,
                b_shared_seq_len=b_shared_seq_len,
                b_shared_radix_node_id=b_shared_radix_node_id,
                b_position_delta=torch.zeros(batch_size, dtype=torch.int32, device="cuda"),
                is_prefill=False,
                multimodal_params=[{"images": [], "audios": []} for _ in range(batch_size)],
                **model._gen_special_model_input(batch_size),
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
            f"Capture cudagraph success, batch_size <={self.max_batch_size} "
            f"and max_len_in_batch <= {self.graph_max_len_in_batch} will infer with cudagraph."
        )

    @torch.no_grad()
    def warmup_overlap(self, model):
        logger.info("Begin capture overlap cudagraph, use the --disable_cudagraph to disable it.")
        # for typing easy
        from .basemodel import TpPartBaseModel

        model: TpPartBaseModel = model
        for batch_size in self.cuda_graph_batch_sizes[::-1]:
            decode_batches = []
            for micro_batch_index in [0, 1]:
                # dummy decoding, capture the cudagraph
                seq_len = 2
                total_token_num = batch_size * seq_len
                max_len_in_batch = self.graph_max_len_in_batch
                input_ids = torch.tensor([1 for _ in range(batch_size)], dtype=torch.int64, device="cuda")
                b_req_idx = torch.tensor(
                    [model.req_manager.HOLD_REQUEST_ID for _ in range(batch_size)], dtype=torch.int32, device="cuda"
                )
                b_seq_len = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda")
                b_mtp_index = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
                b_shared_seq_len = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
                b_shared_radix_node_id = torch.full((batch_size,), -1, dtype=torch.int64, device="cuda")

                micro_batch = ModelInput(
                    is_prefill=False,
                    batch_size=batch_size,
                    total_token_num=total_token_num,
                    max_q_seq_len=1,
                    max_kv_seq_len=max_len_in_batch,
                    input_ids=input_ids,
                    b_mtp_index=b_mtp_index,
                    b_req_idx=b_req_idx,
                    b_seq_len=b_seq_len,
                    b_shared_seq_len=b_shared_seq_len,
                    b_shared_radix_node_id=b_shared_radix_node_id,
                    b_position_delta=torch.zeros(batch_size, dtype=torch.int32, device="cuda"),
                    multimodal_params=[{"images": [], "audios": []} for _ in range(batch_size)],
                    **model._gen_special_model_input(batch_size),
                )
                decode_batches.append(micro_batch)
                del micro_batch

                for var_name, var_value in list(locals().items()):
                    if isinstance(var_value, torch.Tensor):
                        del locals()[var_name]
                torch.cuda.empty_cache()

            _, _ = model.microbatch_overlap_decode(decode_batches[0], decode_batches[1])

            model.att_state_manager.full_att_allocator.free_all()
            model.req_manager.free_all()

            del decode_batches

            # release local tensors
            for var_name, var_value in list(locals().items()):
                if isinstance(var_value, torch.Tensor):
                    del locals()[var_name]
            torch.cuda.empty_cache()

        logger.info(
            f"Capture overlap cudagraph success, batch_size <={self.max_batch_size} "
            f"and max_len_in_batch <= {self.graph_max_len_in_batch} will infer with cudagraph."
        )
