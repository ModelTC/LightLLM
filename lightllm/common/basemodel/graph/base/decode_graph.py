import bisect
import copy
import triton
import torch
from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.platform import get_backend
from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import get_env_start_args
from typing import Optional

logger = init_logger(__name__)


class DecodeGraph:

    PLATFORM_CLASS_MAP: dict[str, type["DecodeGraph"]] = {}

    def __new__(
        cls,
        max_batch_size: int,
        max_len_in_batch: int,
        tp_world_size: int = 1,
        platform_backend: str = "cuda",
    ):
        if cls is not DecodeGraph:
            return object.__new__(cls)
        impl_cls = cls.PLATFORM_CLASS_MAP[platform_backend]
        return object.__new__(impl_cls)

    def __init__(
        self,
        max_batch_size: int,
        max_len_in_batch: int,
        tp_world_size: int = 1,
        platform_backend: str = "cuda",
    ):
        self.hardware_platform = platform_backend
        self._init_decode_graph(max_batch_size, max_len_in_batch, tp_world_size)
        self._init_decode_graph_extra()

    def _init_decode_graph_extra(self):
        pass

    def _init_decode_graph(self, max_batch_size: int, max_len_in_batch: int, tp_world_size: int):
        self.graph: dict[int, tuple] = {}

        args = get_env_start_args()
        mtp_step = args.mtp_step
        self.max_batch_size = max_batch_size
        self.graph_max_len_in_batch = max_len_in_batch
        self.enable_decode_microbatch_overlap = args.enable_decode_microbatch_overlap

        self.platform_backend = get_backend()        
        self.target_device = self.platform_backend.runtime.target_device()

        self.mempool = self.platform_backend.graph.graph_pool_handle()

        graph_split_batch_size = args.graph_split_batch_size * (mtp_step + 1)
        graph_grow_step_size = args.graph_grow_step_size * (mtp_step + 1)
        batch_sizes = [i * (mtp_step + 1) for i in range(1, graph_split_batch_size + 1)]
        for _batch_size in range(graph_split_batch_size + graph_grow_step_size, max_batch_size, graph_grow_step_size):
            batch_sizes.append(_batch_size)
        batch_sizes = list(set([e for e in batch_sizes if e < max_batch_size]))
        batch_sizes.append(max_batch_size)
        batch_sizes.sort()

        if args.enable_tpsp_mix_mode:
            batch_sizes = [triton.cdiv(e, tp_world_size) * tp_world_size for e in batch_sizes]
            batch_sizes = list(set(batch_sizes))
            batch_sizes.sort()
        
        self.graph_batch_sizes = batch_sizes
        assert batch_sizes[-1] == self.max_batch_size
        logger.info(f"decode graph batch_sizes: {self.graph_batch_sizes}")

    def can_run(self, batch_size: int, max_len_in_batch: int) -> bool:
        return batch_size <= self.max_batch_size and max_len_in_batch <= self.graph_max_len_in_batch

    def need_capture(self, batch_size: int) -> bool:
        find_batch_size = self.find_closest_graph_batch_size(batch_size)
        if find_batch_size is not None:
            return find_batch_size not in self.graph
        else:
            assert False, "dead code"

    def find_closest_graph_batch_size(self, batch_size: int) -> Optional[int]:
        index = bisect.bisect_left(self.graph_batch_sizes, batch_size)
        if index < len(self.graph_batch_sizes):
            find_batch_size = self.graph_batch_sizes[index]
            return find_batch_size
        else:
            return None

    def _capture_decode(self, decode_func, infer_state: InferStateInfo) -> ModelOutput:
        graph_obj = self.platform_backend.graph.create_graph()
        batch_size = infer_state.input_ids.shape[0]
        infer_state.max_kv_seq_len = self.graph_max_len_in_batch
        infer_state.total_token_num = self.graph_max_len_in_batch * batch_size
        # warmup
        for _ in range(1):
            pure_para_set = set(vars(infer_state).keys())
            self.platform_backend.runtime.synchronize()
            decode_func(copy.copy(infer_state))
            self.platform_backend.runtime.synchronize()
            for param_name in set(vars(infer_state).keys()):
                if param_name not in pure_para_set:
                    delattr(infer_state, param_name)

        with self.platform_backend.graph.graph(graph_obj, pool=self.mempool):
            model_output = decode_func(infer_state)
        self.graph[batch_size] = (graph_obj, infer_state, model_output)

        return model_output

    def _capture_decode_overlap(
        self,
        decode_func,
        infer_state: InferStateInfo,
        infer_state1: InferStateInfo,
    ) -> tuple[ModelOutput, ModelOutput]:
        graph_obj = self.platform_backend.graph.create_graph()
        batch_size = infer_state.input_ids.shape[0]
        infer_state.max_kv_seq_len = self.graph_max_len_in_batch
        infer_state.total_token_num = self.graph_max_len_in_batch * batch_size
        infer_state1.max_kv_seq_len = self.graph_max_len_in_batch
        infer_state1.total_token_num = self.graph_max_len_in_batch * batch_size
        # warmup
        for _ in range(1):
            pure_para_set = set(vars(infer_state).keys())
            pure_para_set1 = set(vars(infer_state1).keys())
            self.platform_backend.runtime.synchronize()
            decode_func(copy.copy(infer_state), copy.copy(infer_state1))
            self.platform_backend.runtime.synchronize()
            for param_name in set(vars(infer_state).keys()):
                if param_name not in pure_para_set:
                    delattr(infer_state, param_name)
            for param_name in set(vars(infer_state1).keys()):
                if param_name not in pure_para_set1:
                    delattr(infer_state1, param_name)

        with self.platform_backend.graph.graph(graph_obj, pool=self.mempool):
            model_output, model_output1 = decode_func(infer_state, infer_state1)
        self.graph[batch_size] = (graph_obj, infer_state, infer_state1, model_output, model_output1)

        return model_output, model_output1

    def capture_decode(
        self,
        decode_func,
        infer_state: InferStateInfo,
        infer_state1: Optional[InferStateInfo] = None,
    ) -> tuple[ModelOutput, ModelOutput]:
        if self.enable_decode_microbatch_overlap:
            return self._capture_decode_overlap(decode_func, infer_state, infer_state1)
        else:
            assert infer_state1 is None
            return self._capture_decode(decode_func, infer_state)

    def _replay(
        self,
        infer_state: InferStateInfo,
        b1_cu_q_seq_len_cpu: list[int],
        b_cu_kv_seq_len_cpu: list[int],
    ) -> ModelOutput:
        batch_size = infer_state.input_ids.shape[0]
        graph_obj, graph_infer_state, graph_output = self.graph[batch_size]
        graph_infer_state.copy_for_cuda_graph(infer_state)
        self.platform_backend.graph.replay_graph(graph_obj)

        return graph_output

    def _replay_overlap(
        self,
        infer_state: InferStateInfo,
        infer_state1: InferStateInfo,
        b1_cu_q_seq_len_cpu: list[int],
        b_cu_kv_seq_len_cpu: list[int],
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
        self.platform_backend.graph.replay_graph(graph_obj)

        return graph_model_output, graph_model_output1

    def replay(self, infer_state, b1_cu_q_seq_len_cpu: list[int], b_cu_kv_seq_len_cpu: list[int], infer_state1=None):
        if self.enable_decode_microbatch_overlap:
            return self._replay_overlap(infer_state, infer_state1, b1_cu_q_seq_len_cpu, b_cu_kv_seq_len_cpu)
        else:
            assert infer_state1 is None
            return self._replay(infer_state, b1_cu_q_seq_len_cpu, b_cu_kv_seq_len_cpu)

    @torch.no_grad()
    def warmup(self, model):
        logger.info("Begin capture cudagraph, use the --disable_cudagraph to disable it.")
        # for typing easy
        from lightllm.common.basemodel.basemodel import TpPartBaseModel

        model: TpPartBaseModel = model

        # decode cuda graph init
        for batch_size in self.graph_batch_sizes[::-1]:
            seq_len = self.graph_max_len_in_batch
            total_token_num = batch_size * seq_len
            max_len_in_batch = self.graph_max_len_in_batch
            input_ids = torch.tensor([1 for _ in range(batch_size)], dtype=torch.int32, device=self.target_device)
            mem_indexes = model.mem_manager.alloc(len(input_ids)).to(self.target_device)
            b_req_idx = torch.tensor(
                [model.req_manager.HOLD_REQUEST_ID for _ in range(batch_size)], dtype=torch.int32, device=self.target_device
            )
            b_seq_len = torch.empty(batch_size, dtype=torch.int32, device=self.target_device)
            b_seq_len.fill_(seq_len)
            b_mtp_index = torch.zeros(batch_size, dtype=torch.int32, device=self.target_device)

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
            self.platform_backend.runtime.empty_cache()

        logger.info(
            f"Capture cudagraph success, batch_size <={self.max_batch_size} "
            f"and max_len_in_batch <= {self.graph_max_len_in_batch} will infer with cudagraph."
        )

    @torch.no_grad()
    def warmup_overlap(self, model):
        logger.info("Begin capture overlap cudagraph, use the --disable_cudagraph to disable it.")
        # for typing easy
        from lightllm.common.basemodel.basemodel import TpPartBaseModel

        model: TpPartBaseModel = model

        for batch_size in self.graph_batch_sizes[::-1]:
            decode_batches = []
            for micro_batch_index in [0, 1]:
                # dummy decoding, capture the cudagraph
                seq_len = self.graph_max_len_in_batch
                total_token_num = batch_size * seq_len
                max_len_in_batch = self.graph_max_len_in_batch
                input_ids = torch.tensor([1 for _ in range(batch_size)], dtype=torch.int32, device=self.target_device)
                mem_indexes = model.mem_manager.alloc(len(input_ids)).to(self.target_device)
                b_req_idx = torch.tensor(
                    [model.req_manager.HOLD_REQUEST_ID for _ in range(batch_size)], dtype=torch.int32, device=self.target_device
                )
                b_seq_len = torch.empty(batch_size, dtype=torch.int32, device=self.target_device)
                b_seq_len.fill_(seq_len)
                b_mtp_index = torch.zeros(batch_size, dtype=torch.int32, device=self.target_device)

                micro_batch = ModelInput(
                    is_prefill=False,
                    batch_size=batch_size,
                    total_token_num=total_token_num,
                    max_q_seq_len=1,
                    max_kv_seq_len=max_len_in_batch,
                    input_ids=input_ids,
                    b_mtp_index=b_mtp_index,
                    mem_indexes=mem_indexes,
                    b_req_idx=b_req_idx,
                    b_seq_len=b_seq_len,
                    multimodal_params=[{"images": [], "audios": []} for _ in range(batch_size)],
                    **model._gen_special_model_input(batch_size),
                )
                decode_batches.append(micro_batch)
                del micro_batch

                for var_name, var_value in list(locals().items()):
                    if isinstance(var_value, torch.Tensor):
                        del locals()[var_name]
                self.platform_backend.runtime.empty_cache()

            _, _ = model.microbatch_overlap_decode(decode_batches[0], decode_batches[1])

            model.mem_manager.free_all()
            model.req_manager.free_all()

            del decode_batches

            # release local tensors
            for var_name, var_value in list(locals().items()):
                if isinstance(var_value, torch.Tensor):
                    del locals()[var_name]
            self.platform_backend.runtime.empty_cache()

        logger.info(
            f"Capture overlap cudagraph success, batch_size <={self.max_batch_size} "
            f"and max_len_in_batch <= {self.graph_max_len_in_batch} will infer with cudagraph."
        )
