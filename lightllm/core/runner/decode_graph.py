from __future__ import annotations

import torch
import torch.distributed as dist
import copy
import bisect
import math
import triton
from typing import Optional, TYPE_CHECKING
from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.common.triton_utils.autotuner import Autotuner, AutotuneKernelType
from lightllm.utils.torch_memory_saver_utils import (
    TorchMemorySaverWrapper,
    MemoryTag,
)
from lightllm.core.device.cuda import CudaRuntime

if TYPE_CHECKING:
    from lightllm.common.basemodel.infer_struct import InferStateInfo


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
        runtime=CudaRuntime,
    ):
        self.runtime = runtime
        self.graph = {}
        self.tp_world_size = tp_world_size
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

    def close(self):
        for captured in self.graph.values():
            captured[0].reset()
        self.graph.clear()
        self.mempool = None

    def can_run(self, batch_size, max_len_in_batch):
        return batch_size <= self.max_batch_size and max_len_in_batch <= self.graph_max_len_in_batch

    def need_capture(self, batch_size):
        find_batch_size = self.find_closest_graph_batch_size(batch_size)
        if find_batch_size is not None:
            return find_batch_size not in self.graph
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
        # warmup
        # 因为有些推理过程的代码，会通过判断infer_state中是否存在某些属性来在一层上
        # 做一些初始化的操作，后续层可以复用这些计算的结果，如
        # lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding.py
        # 中做的一些操作，所以在 warmup 的时候，需要调用infer_state的copy函数做一个
        # 浅拷贝，不然后续传入到cuda graph捕获过程中后，infer_state因为提前拥有了这些属性，
        # 导致不会重新初始化，这样捕获过程中会不能捕获这些临时添加到 infer_state 管理对象
        # 中的 tensor。

        for _ in range(1):
            # 记录原始存在的变量
            pure_para_set = set(vars(infer_state).keys())
            self.runtime.synchronize()
            # 在正式捕获前调优 decode attention，退出作用域后再捕获选定的配置。
            with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
                decode_func(copy.copy(infer_state))
            self.runtime.synchronize()
            for param_name in set(vars(infer_state).keys()):
                if param_name not in pure_para_set:
                    delattr(infer_state, param_name)

        with self.runtime.capture_graph(graph_obj, pool=self.mempool, memory_saver=self.torch_memory_saver):
            model_output = decode_func(infer_state)
        self.graph[batch_size] = (graph_obj, infer_state, model_output)
        graph_obj.replay()
        self._measure_replay_cost(graph_obj=graph_obj, batch_size=batch_size)
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
        # warmup
        for _ in range(1):
            # 记录原始存在的变量
            pure_para_set = set(vars(infer_state).keys())
            pure_para_set1 = set(vars(infer_state1).keys())
            self.runtime.synchronize()
            with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
                decode_func(copy.copy(infer_state), copy.copy(infer_state1))
            self.runtime.synchronize()
            for para_name in set(vars(infer_state).keys()):
                if para_name not in pure_para_set:
                    delattr(infer_state, para_name)
            for para_name in set(vars(infer_state1).keys()):
                if para_name not in pure_para_set1:
                    delattr(infer_state1, para_name)

        with self.runtime.capture_graph(graph_obj, pool=self.mempool, memory_saver=self.torch_memory_saver):
            model_output, model_output1 = decode_func(infer_state, infer_state1)
        self.graph[batch_size] = (
            graph_obj,
            infer_state,
            infer_state1,
            model_output,
            model_output1,
        )
        graph_obj.replay()
        self._measure_replay_cost(graph_obj=graph_obj, batch_size=batch_size)
        return model_output, model_output1

    def _measure_replay_cost(self, graph_obj: torch.cuda.CUDAGraph, batch_size: int) -> None:
        if not self.capture_infer_cost:
            return

        dist.barrier(group=dist.group.WORLD)
        start_event = self.runtime.create_event(enable_timing=True)
        end_event = self.runtime.create_event(enable_timing=True)
        graph_obj.replay()
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

    def _replay(self, infer_state: InferStateInfo):
        batch_size = infer_state.input_ids.shape[0]
        graph_obj, graph_infer_state, graph_output = self.graph[batch_size]
        graph_infer_state.copy_for_cuda_graph(infer_state)
        graph_obj.replay()
        return graph_output

    def _replay_overlap(
        self,
        infer_state: InferStateInfo,
        infer_state1: InferStateInfo,
    ):
        batch_size = infer_state.input_ids.shape[0]
        (
            graph_obj,
            graph_infer_state,
            graph_infer_state1,
            graph_model_output,
            graph_model_output1,
        ) = self.graph[batch_size]
        graph_infer_state.copy_for_cuda_graph(infer_state)
        graph_infer_state1.copy_for_cuda_graph(infer_state1)
        graph_obj.replay()
        return graph_model_output, graph_model_output1

    def replay(self, infer_state, infer_state1=None):
        if self.enable_decode_microbatch_overlap:
            return self._replay_overlap(infer_state, infer_state1)
        else:
            assert infer_state1 is None
            return self._replay(infer_state)
