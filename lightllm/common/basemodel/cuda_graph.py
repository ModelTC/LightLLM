import os
import torch
import copy
import bisect
import math
import triton
from typing import Optional
from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.distributed import dist_group_manager
from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.utils.torch_memory_saver_utils import (
    TorchMemorySaverWrapper,
    MemoryTag,
)
from .infer_struct import InferStateInfo


logger = init_logger(__name__)


class CudaGraph:
    # CudaGraph forward pass for the decoding stage.

    def __init__(self, max_batch_size=8, max_len_in_batch=8192, tp_world_size: int = 1):
        self.graph = {}
        self.tp_world_size = tp_world_size
        self.mempool = torch.cuda.graph_pool_handle() if torch.cuda.is_available() else None
        self.args = get_env_start_args()
        self.mtp_step = self.args.mtp_step
        self.max_batch_size = max_batch_size
        self.graph_max_len_in_batch = max_len_in_batch
        self.enable_decode_microbatch_overlap = self.args.enable_decode_microbatch_overlap
        self.torch_memory_saver = TorchMemorySaverWrapper(self.args.enable_torch_memory_saver)

        # With MTP enabled, both the main-model verify forward and the draft (MTP) forward run over
        # the (mtp_step+1)-expanded decode layout, so every decode batch size is a multiple of
        # (mtp_step+1) and there is a single decode layout — the graph is keyed by batch size alone.
        batch_size_multiple = self.mtp_step + 1 if self.mtp_step > 0 else 1
        self.cuda_graph_batch_sizes = self._build_cuda_graph_batch_sizes(batch_size_multiple=batch_size_multiple)
        logger.info(f"cuda graph batch_sizes: {self.cuda_graph_batch_sizes}")

    def _build_cuda_graph_batch_sizes(self, batch_size_multiple: int):
        graph_split_batch_size = self.args.graph_split_batch_size * batch_size_multiple
        graph_grow_step_size = self.args.graph_grow_step_size * batch_size_multiple

        batch_sizes = [i * batch_size_multiple for i in range(1, self.args.graph_split_batch_size + 1)]
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
            padding_unit = math.lcm(self.tp_world_size, batch_size_multiple)
            batch_sizes = [triton.cdiv(e, padding_unit) * padding_unit for e in batch_sizes]
            batch_sizes = list(set(batch_sizes))
            batch_sizes.sort()

        assert batch_sizes[-1] == self.max_batch_size
        return batch_sizes

    def can_run(self, batch_size, max_len_in_batch):
        return batch_size <= self.max_batch_size and max_len_in_batch <= self.graph_max_len_in_batch

    def _decode_graph_key(self, infer_state: InferStateInfo):
        return infer_state.input_ids.shape[0]

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

    def _build_warmup_decode_model_input(
        self,
        model,
        batch_size: int,
        device: str = "cuda",
    ) -> ModelInput:
        mtp_size = self.mtp_step + 1
        input_ids = torch.ones(batch_size, dtype=torch.int32, device=device)
        mem_indexes = model.mem_manager.alloc(batch_size).to(device)
        b_req_idx = torch.full(
            (batch_size,),
            fill_value=model.req_manager.HOLD_REQUEST_ID,
            dtype=torch.int32,
            device=device,
        )

        b_num_accepted_tokens = None
        if self.mtp_step > 0:
            assert batch_size % mtp_size == 0, "MTP decode CUDA graph batch size must be a multiple of mtp_step + 1"
            real_batch_size = batch_size // mtp_size
            b_mtp_index = torch.arange(mtp_size, dtype=torch.int32, device=device).repeat(real_batch_size)
            b_seq_len = torch.arange(2, mtp_size + 2, dtype=torch.int32, device=device).repeat(real_batch_size)
            # b_num_accepted_tokens 不再随 model_input 传入：GDN 的 init_mtp_verify_extra_state 会按
            # req_first(全 HOLD，槽恒为 1) gather，warmup/capture 自然得到全 1，等价旧的 torch.ones。
            total_token_num = real_batch_size * (mtp_size * (mtp_size + 3) // 2)
        else:
            seq_len = 2
            total_token_num = batch_size * seq_len
            b_mtp_index = torch.zeros(batch_size, dtype=torch.int32, device=device)
            b_seq_len = torch.empty(batch_size, dtype=torch.int32, device=device)
            b_seq_len.fill_(seq_len)

        return ModelInput(
            batch_size=batch_size,
            total_token_num=total_token_num,
            max_q_seq_len=1,
            max_kv_seq_len=self.graph_max_len_in_batch,
            input_ids=input_ids,
            mem_indexes=mem_indexes,
            b_req_idx=b_req_idx,
            b_seq_len=b_seq_len,
            b_mtp_index=b_mtp_index,
            b_num_accepted_tokens=b_num_accepted_tokens,
            is_prefill=False,
            multimodal_params=[{"images": [], "audios": []} for _ in range(batch_size)],
            **model._gen_special_model_input(batch_size),
        )

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

        with self.torch_memory_saver.cuda_graph(graph_obj, pool=self.mempool):
            model_output = decode_func(infer_state)
        self.graph[self._decode_graph_key(infer_state)] = (
            graph_obj,
            infer_state,
            model_output,
        )
        graph_obj.replay()
        return model_output

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

        with self.torch_memory_saver.cuda_graph(graph_obj, pool=self.mempool):
            model_output, model_output1 = decode_func(infer_state, infer_state1)
        self.graph[self._decode_graph_key(infer_state)] = (
            graph_obj,
            infer_state,
            infer_state1,
            model_output,
            model_output1,
        )
        graph_obj.replay()
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
        graph_obj, graph_infer_state, graph_output = self.graph[self._decode_graph_key(infer_state)]
        graph_infer_state.copy_for_cuda_graph(infer_state)
        graph_obj.replay()
        return graph_output

    def _replay_overlap(
        self,
        infer_state: InferStateInfo,
        infer_state1: InferStateInfo,
    ):
        (
            graph_obj,
            graph_infer_state,
            graph_infer_state1,
            graph_model_output,
            graph_model_output1,
        ) = self.graph[self._decode_graph_key(infer_state)]
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

    @torch.no_grad()
    def warmup(self, model):
        logger.info("Begin capture cudagraph, use the --disable_cudagraph to disable it.")
        # for typing easy
        from .basemodel import TpPartBaseModel

        model: TpPartBaseModel = model

        # decode cuda graph init
        for batch_size in self.cuda_graph_batch_sizes[::-1]:
            model_input = self._build_warmup_decode_model_input(model, batch_size)
            model_output: ModelOutput = model.forward(model_input)
            del model_output

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
                micro_batch = self._build_warmup_decode_model_input(model, batch_size)
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
