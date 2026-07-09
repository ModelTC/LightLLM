import os
import torch
import torch.distributed as dist
import copy
import bisect
import triton
from typing import Optional
from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import (
    get_env_start_args,
    enable_dynamic_mtp_verify,
    get_diverse_max_batch_shared_group_size,
)
from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from .infer_struct import InferStateInfo


logger = init_logger(__name__)


def _build_mtp_mark_shared_group_values(
    *,
    batch_size: int,
    mtp_group_size: int,
    max_group_size: int,
    split_groups: bool,
) -> list:
    assert mtp_group_size > 0
    assert max_group_size > 0

    group_cap = max_group_size if split_groups else mtp_group_size
    b_mark_shared_group = [0 for _ in range(batch_size)]
    for req_start in range(0, batch_size, mtp_group_size):
        req_end = min(req_start + mtp_group_size, batch_size)
        for group_start in range(req_start, req_end, group_cap):
            group_size = min(group_cap, req_end - group_start)
            b_mark_shared_group[group_start + group_size - 1] = group_size
    return b_mark_shared_group


class CudaGraph:
    # CudaGraph forward pass for the decoding stage.

    def __init__(self, max_batch_size=8, max_len_in_batch=8192, tp_world_size: int = 1):
        self.graph = {}
        self.tp_world_size = tp_world_size
        self.mempool = torch.cuda.graph_pool_handle() if torch.cuda.is_available() else None
        self.args = get_env_start_args()
        self.raw_max_batch_size = max_batch_size
        self.max_batch_size = None
        self.graph_max_len_in_batch = max_len_in_batch
        self.enable_decode_microbatch_overlap = self.args.enable_decode_microbatch_overlap
        self.spec_adapter = None
        self.model = None

        self._refresh_cuda_graph_batch_sizes()
        return

    def set_spec_adapter(self, spec_adapter, model=None):
        self.spec_adapter = spec_adapter
        self.model = model
        self._refresh_cuda_graph_batch_sizes()
        return

    def _refresh_cuda_graph_batch_sizes(self):
        # gen cuda graph batch_sizes
        # cuda graph gen for batch size = [1, 2, 3, ..., graph_split_batch_size]
        # and [graph_split_batch_size + graph_grow_step_size,
        # if the mtp_step is not 0, then the batch_sizes will be multiply of (mtp_step + 1)

        mtp_step = self._get_decode_graph_mtp_step()
        group_size = mtp_step + 1
        self.max_batch_size = (self.raw_max_batch_size // group_size) * group_size
        assert self.max_batch_size > 0, "cuda graph max_batch_size must cover at least one decode group"

        graph_split_batch_size = self.args.graph_split_batch_size * group_size
        graph_grow_step_size = self.args.graph_grow_step_size * group_size

        batch_sizes = [i * group_size for i in range(1, self.args.graph_split_batch_size + 1)]
        for _batch_size in range(
            graph_split_batch_size + graph_grow_step_size,
            self.max_batch_size,
            graph_grow_step_size,
        ):
            batch_sizes.append(_batch_size)

        batch_sizes = list(set([e for e in batch_sizes if e < self.max_batch_size]))
        batch_sizes.append(self.max_batch_size)
        batch_sizes.sort()
        if self.args.enable_tpsp_mix_mode:
            batch_sizes = [triton.cdiv(e, self.tp_world_size) * self.tp_world_size for e in batch_sizes]
            batch_sizes = list(set(batch_sizes))
            batch_sizes.sort()

        self.cuda_graph_batch_sizes = batch_sizes
        assert batch_sizes[-1] == self.max_batch_size
        logger.info(f"cuda graph batch_sizes: {self.cuda_graph_batch_sizes}")
        return

    def _get_decode_graph_mtp_step(self) -> int:
        if self.spec_adapter is None:
            return self.args.mtp_step
        return self.spec_adapter.get_decode_graph_mtp_step(self.model)

    def _is_block_draft_model(self) -> bool:
        if self.spec_adapter is None or self.model is None:
            return False
        is_block_draft_model = getattr(self.spec_adapter, "is_block_draft_model", None)
        return is_block_draft_model is not None and is_block_draft_model(self.model)

    def can_run(self, batch_size, max_len_in_batch):
        return batch_size <= self.max_batch_size and max_len_in_batch <= self.graph_max_len_in_batch

    def _graph_key(
        self,
        batch_size: int,
        model_context=None,
        model_context1=None,
    ):
        spec_key = None
        spec_key1 = None
        if self.spec_adapter is not None and model_context is not None:
            spec_key = self.spec_adapter.graph_cache_key(model_context, model=self.model)
        if self.spec_adapter is not None and model_context1 is not None:
            spec_key1 = self.spec_adapter.graph_cache_key(model_context1, model=self.model)
        if spec_key is None and spec_key1 is None:
            return batch_size
        return (batch_size, spec_key, spec_key1)

    def _export_spec_capture(self, infer_state: InferStateInfo):
        if self.spec_adapter is None:
            return None
        return self.spec_adapter.export_graph_capture()

    def _restore_spec_capture(self, infer_state: InferStateInfo, captured_hiddens) -> None:
        if self.spec_adapter is not None:
            self.spec_adapter.restore_graph_capture(captured_hiddens)
        return

    def need_capture(
        self,
        batch_size,
        model_context: Optional[ModelInput] = None,
        model_context1: Optional[ModelInput] = None,
    ):
        find_batch_size = self.find_closest_graph_batch_size(batch_size)
        return (
            find_batch_size is not None
            and self._graph_key(find_batch_size, model_context, model_context1) not in self.graph
        )

    def find_closest_graph_batch_size(self, batch_size):
        index = bisect.bisect_left(self.cuda_graph_batch_sizes, batch_size)
        if index < len(self.cuda_graph_batch_sizes):
            find_batch_size = self.cuda_graph_batch_sizes[index]
            return find_batch_size
        else:
            return None

    def _make_warmup_mtp_index(self, batch_size: int) -> torch.Tensor:
        mtp_step = self._get_decode_graph_mtp_step()
        if mtp_step <= 0:
            return torch.zeros(batch_size, dtype=torch.int32, device="cuda")
        return torch.arange(batch_size, dtype=torch.int32, device="cuda") % (mtp_step + 1)

    def _make_warmup_seq_len(self, batch_size: int) -> torch.Tensor:
        mtp_step = self._get_decode_graph_mtp_step()
        if mtp_step > 0:
            group_size = mtp_step + 1
            return torch.arange(batch_size, dtype=torch.int32, device="cuda") % group_size + 2
        return torch.full((batch_size,), 2, dtype=torch.int32, device="cuda")

    def _make_warmup_mtp_mark_shared_group(self, batch_size: int) -> Optional[torch.Tensor]:
        mtp_step = self._get_decode_graph_mtp_step()
        if mtp_step <= 0:
            return None

        mtp_group_size = mtp_step + 1
        max_group_size = get_diverse_max_batch_shared_group_size()
        b_mark_shared_group = _build_mtp_mark_shared_group_values(
            batch_size=batch_size,
            mtp_group_size=mtp_group_size,
            max_group_size=max_group_size,
            split_groups=not self._is_block_draft_model(),
        )
        return torch.tensor(b_mark_shared_group, dtype=torch.int32, device="cuda")

    def _capture_decode(self, decode_func, infer_state: InferStateInfo):
        graph_obj = torch.cuda.CUDAGraph()
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
            torch.cuda.synchronize()
            decode_func(copy.copy(infer_state))
            torch.cuda.synchronize()
            for param_name in set(vars(infer_state).keys()):
                if param_name not in pure_para_set:
                    delattr(infer_state, param_name)

        with torch.cuda.graph(graph_obj, pool=self.mempool):
            model_output = decode_func(infer_state)
        spec_capture = self._export_spec_capture(infer_state)
        self.graph[self._graph_key(batch_size, infer_state)] = (graph_obj, infer_state, model_output, spec_capture)
        graph_obj.replay()
        self._record_capture_replay_infer_cost_ms(
            graph_obj=graph_obj,
            batch_size=batch_size,
            is_draft_model=self._is_draft_model_capture(infer_state),
        )
        return model_output

    def _record_capture_replay_infer_cost_ms(
        self,
        graph_obj: torch.cuda.CUDAGraph,
        batch_size: int,
        is_draft_model: bool,
    ) -> None:
        if not enable_dynamic_mtp_verify():
            return
        if is_draft_model and self.args.mtp_mode == "dspark":
            # DSpark's planner uses target verify cost plus confidence-derived
            # capacity estimates. Draft block cost is not part of the decision,
            # so avoid adding a runtime barrier/synchronize on lazy draft graph
            # capture.
            return

        from lightllm.server.router.model_infer.infer_batch import g_infer_context

        dist.barrier(group=dist.group.WORLD)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        graph_obj.replay()
        start_event.record()
        graph_obj.replay()
        end_event.record()
        end_event.synchronize()
        infer_cost_ms_tensor = torch.tensor([start_event.elapsed_time(end_event)], dtype=torch.float32, device="cuda")
        dist.all_reduce(infer_cost_ms_tensor, op=dist.ReduceOp.MIN, group=dist.group.WORLD)
        infer_cost_ms = float(infer_cost_ms_tensor.item())
        g_infer_context.record_dynamic_mtp_infer_cost(
            batch_size=batch_size,
            infer_cost_ms=infer_cost_ms,
            is_draft_model=is_draft_model,
        )
        return

    def _is_draft_model_capture(self, infer_state: InferStateInfo) -> bool:
        if infer_state.mtp_draft_input_hiddens is not None or getattr(infer_state, "is_draft_model", False):
            return True
        if self.spec_adapter is None or self.model is None:
            return False
        return self.spec_adapter.is_draft_model(self.model)

    def _capture_decode_overlap(
        self,
        decode_func,
        infer_state: InferStateInfo,
        infer_state1: InferStateInfo,
    ):
        graph_obj = torch.cuda.CUDAGraph()
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
            torch.cuda.synchronize()
            decode_func(copy.copy(infer_state), copy.copy(infer_state1))
            torch.cuda.synchronize()
            for para_name in set(vars(infer_state).keys()):
                if para_name not in pure_para_set:
                    delattr(infer_state, para_name)
            for para_name in set(vars(infer_state1).keys()):
                if para_name not in pure_para_set1:
                    delattr(infer_state1, para_name)

        with torch.cuda.graph(graph_obj, pool=self.mempool):
            model_output, model_output1 = decode_func(infer_state, infer_state1)
        spec_capture = self._export_spec_capture(infer_state)
        spec_capture1 = self._export_spec_capture(infer_state1)
        self.graph[self._graph_key(batch_size, infer_state, infer_state1)] = (
            graph_obj,
            infer_state,
            infer_state1,
            model_output,
            model_output1,
            spec_capture,
            spec_capture1,
        )
        graph_obj.replay()
        self._record_capture_replay_infer_cost_ms(
            graph_obj=graph_obj,
            batch_size=batch_size,
            is_draft_model=self._is_draft_model_capture(infer_state),
        )
        return model_output, model_output1

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
        graph_obj, graph_infer_state, graph_output, spec_capture = self.graph[self._graph_key(batch_size, infer_state)]
        graph_infer_state.copy_for_cuda_graph(infer_state)
        graph_obj.replay()
        self._restore_spec_capture(infer_state, spec_capture)
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
            spec_capture,
            spec_capture1,
        ) = self.graph[self._graph_key(batch_size, infer_state, infer_state1)]
        graph_infer_state.copy_for_cuda_graph(infer_state)
        graph_infer_state1.copy_for_cuda_graph(infer_state1)
        graph_obj.replay()
        self._restore_spec_capture(infer_state, spec_capture)
        self._restore_spec_capture(infer_state1, spec_capture1)
        return graph_model_output, graph_model_output1

    def replay(self, infer_state, infer_state1=None):
        if self.enable_decode_microbatch_overlap:
            return self._replay_overlap(infer_state, infer_state1)
        else:
            assert infer_state1 is None
            return self._replay(infer_state)

    @torch.no_grad()
    def warmup(self, model):
        logger.info("Begin capture cudagraph, use the --disable_cudagraph to disable it.")
        # for typing easy
        from .basemodel import TpPartBaseModel

        model: TpPartBaseModel = model
        # decode cuda graph init
        for batch_size in self.cuda_graph_batch_sizes[::-1]:
            max_len_in_batch = self.graph_max_len_in_batch
            input_ids = torch.tensor([1 for _ in range(batch_size)], dtype=torch.int32, device="cuda")
            mem_indexes = model.mem_manager.alloc(len(input_ids)).cuda()
            b_req_idx = torch.tensor(
                [model.req_manager.HOLD_REQUEST_ID for _ in range(batch_size)], dtype=torch.int32, device="cuda"
            )
            b_seq_len = self._make_warmup_seq_len(batch_size)
            total_token_num = int(b_seq_len.sum().item())
            b_mtp_index = self._make_warmup_mtp_index(batch_size)
            b_mark_shared_group = self._make_warmup_mtp_mark_shared_group(batch_size)

            model_input = ModelInput(
                batch_size=batch_size,
                total_token_num=total_token_num,
                max_q_seq_len=1,
                max_kv_seq_len=max_len_in_batch,
                input_ids=input_ids,
                mem_indexes=mem_indexes,
                b_req_idx=b_req_idx,
                b_seq_len=b_seq_len,
                b_mtp_index=b_mtp_index,
                b_mark_shared_group=b_mark_shared_group,
                b_position_delta=torch.zeros(batch_size, dtype=torch.int32, device="cuda"),
                is_prefill=False,
                multimodal_params=[{"images": [], "audios": []} for _ in range(batch_size)],
                **model._gen_special_model_input(batch_size),
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
                max_len_in_batch = self.graph_max_len_in_batch
                input_ids = torch.tensor([1 for _ in range(batch_size)], dtype=torch.int32, device="cuda")
                mem_indexes = model.mem_manager.alloc(len(input_ids)).cuda()
                b_req_idx = torch.tensor(
                    [model.req_manager.HOLD_REQUEST_ID for _ in range(batch_size)], dtype=torch.int32, device="cuda"
                )
                b_seq_len = self._make_warmup_seq_len(batch_size)
                total_token_num = int(b_seq_len.sum().item())
                b_mtp_index = self._make_warmup_mtp_index(batch_size)
                b_mark_shared_group = self._make_warmup_mtp_mark_shared_group(batch_size)

                micro_batch = ModelInput(
                    is_prefill=False,
                    batch_size=batch_size,
                    total_token_num=total_token_num,
                    max_q_seq_len=1,
                    max_kv_seq_len=max_len_in_batch,
                    input_ids=input_ids,
                    b_mtp_index=b_mtp_index,
                    b_mark_shared_group=b_mark_shared_group,
                    mem_indexes=mem_indexes,
                    b_req_idx=b_req_idx,
                    b_seq_len=b_seq_len,
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

            model.mem_manager.free_all()
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
