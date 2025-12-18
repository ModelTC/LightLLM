import torch
import triton
import collections
from lightllm.common.kv_cache_mem_manager import MemoryManager
from lightllm.common.req_manager import ReqManager
from lightllm.distributed import CustomProcessGroup
from typing import Tuple, Any, Optional, List
from .triton_kernel.gen_prefill_params import gen_prefill_params
from .triton_kernel.gen_decode_params import gen_decode_params
from .triton_kernel.multimodal_emb import mark_multimodal_obj
from .batch_objs import ModelInput
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.dist_utils import get_global_dp_rank


class InferStateInfo:
    """
    推理时用的信息结构体
    """

    def __init__(self):
        self.batch_size: int = None
        self.total_token_num: int = None
        self.b_req_idx: torch.Tensor = None
        self.b_start_loc: torch.Tensor = None
        self.b_ready_cache_len: torch.Tensor = None  # only for prefill prompt cache used.

        self.b_shared_seq_len: torch.Tensor = None  # only for diverse mode used in decode phase.
        self.b_mark_shared_group: torch.Tensor = None  # only for diverse mode used in decode phase.

        self.b_seq_len: torch.Tensor = None
        # max_len_in_batch prefill 和 decode 阶段含义不同
        # prefill 阶段指每个req 输入token的长度（不包括已经cache的部分）最大值
        # decode 阶段指的是每个req的总长 最大值
        self.max_len_in_batch: int = None
        # max_cache_len 用于 prefill 阶段标识请求中最大 cache的kv 的长度
        self.max_cache_len: int = None
        # prefix_total_token_num 用于 prefill 阶段标识当前请求中所有已经ready的kv的长度
        # 的sum值, 其值等于 sum(b_ready_cache_len)
        self.prefix_total_token_num: int = None
        self.is_prefill: bool = None

        self.mem_manager: MemoryManager = None
        self.req_manager: ReqManager = None

        self.mem_index: torch.Tensor = None

        self.is_token_healing: bool = False
        self.return_all_prompt_logics: bool = False
        self.multimodal_params: dict = None
        self.is_cuda_graph: bool = False  # 标记是否是cuda graph的捕获推理
        self.dist_group: CustomProcessGroup = None

        # 在microbatch overlap的运行模式下，用于标记当前 microbatch 的 index 序号
        # 在一些细节场景下需要有该信息区分一些资源的申请和管理。
        self.microbatch_index: int = 0

        # 衍生使用的管理变量，为了方便扩展接入其他的高性能attention推理算子，在
        # inferstate 基类上添加下面的标记变量，用于扩展。
        # b 开头的tensor变量其shape为[batch_size,]
        # b1 开头的tensor变量其shape为[batch_size + 1,]
        self.b_q_seq_len: torch.Tensor = None
        self.b1_cu_q_seq_len: torch.Tensor = None
        self.b_kv_seq_len: torch.Tensor = None
        self.b1_cu_kv_seq_len: torch.Tensor = None
        self.position_ids: torch.Tensor = None
        self.max_q_seq_len: int = None
        self.max_kv_seq_len: int = None

        # 一些特殊模型，特殊模式使用的输入变量，本身这些变量不适合放在
        # inferstate的基类中，但是为了代码的简洁和方便，都放在基类中
        # 进行管理。注意这些成员变量只会在特定的模型和模式下才会生效。

        # deepseekv3 mtp draft model 使用的额外输入参数,
        # 在开启 mtp_mode == deepseekv3 时，mtp draft model
        # 的输入会用到，其他模型和场景都不会用到
        self.deepseekv3_mtp_draft_input_hiddens: Optional[torch.Tensor] = None

        # 在单节点多dp的运行模式下，在进行prefill的阶段，如果出现了dp之间数据不平衡的现象，
        # 可以将推理的数据，进行重新分配到各个dp，在做 att 之前，重新 all to all 到各自的
        # dp，计算完成后，再 all to all 回去，这样可以使，各个dp 间处理的数据比较均衡，提升
        # prefill时候的计算效率。下面的变量，都是在这种场景下才会被使用的变量，普通情况下
        # 下面的变量不会被使用。
        self.need_dp_prefill_balance: bool = False
        self.dp_origin_lens: List[int] = None
        self.dp_handle_lens: List[int] = None
        # self.dp_input_lens: torch.Tensor = None
        self.dp_output_split_sizes: List[List[int]] = None
        self.dp_input_split_sizes: List[List[int]] = None

    def init_some_extra_state(self, model, input_ids: torch.Tensor):
        if self.is_prefill:
            (
                self.b_q_seq_len,
                self.b1_cu_q_seq_len,
                self.b_kv_seq_len,
                self.b1_cu_kv_seq_len,
                self.position_ids,
            ) = gen_prefill_params(
                input_token_num=input_ids.shape[0],
                b_ready_cache_len=self.b_ready_cache_len,
                b_seq_len=self.b_seq_len,
            )
            self.b_start_loc = self.b1_cu_q_seq_len[0:-1]
        else:
            (
                self.b_q_seq_len,
                self.b1_cu_q_seq_len,
                self.b_kv_seq_len,
                self.b1_cu_kv_seq_len,
                self.position_ids,
            ) = gen_decode_params(self.b_seq_len)
            # TODO: check the correctness
            self.max_kv_seq_len = self.max_len_in_batch
            self.b_start_loc = self.b1_cu_kv_seq_len[0:-1]

    def copy_for_cuda_graph(self, new_infer_state: "InferStateInfo"):
        for attr_name, attr_value in vars(new_infer_state).items():
            if isinstance(attr_value, torch.Tensor):
                attr_ = getattr(self, attr_name, None)
                if attr_ is not None and attr_.data_ptr() != attr_value.data_ptr():
                    attr_.copy_(attr_value, non_blocking=True)
        return

    def prefill_dp_balance(self, input_ids: torch.Tensor):
        """
        在prefill的时候, 对于处于 dp 模式下的时候，对输入的数据进行重新的调整和分配，降低各个dp处理数据量过于不一致的时候,导致
        的prefill 推理性能下降
        """
        assert self.is_prefill
        import torch.distributed as dist

        self.need_dp_prefill_balance = True

        args = get_env_start_args()

        dp_input_lens = torch.empty(size=(args.dp,), device="cuda", dtype=torch.int32)
        input_len = torch.empty(size=(1,), device="cuda", dtype=torch.int32)
        input_len.fill_(len(input_ids))
        dist.all_gather_into_tensor(
            output_tensor=dp_input_lens,
            input_tensor=input_len,
            group=self.dist_group.dp_prefill_balance_group,
            async_op=False,
        )
        dp_input_lens = dp_input_lens.detach().cpu()
        self.dp_origin_lens = dp_input_lens.tolist()
        sum_input_len = dp_input_lens.sum().item()
        dp_handle_lens = [sum_input_len // args.dp for _ in range(args.dp)]
        for i in range(sum_input_len % args.dp):
            dp_handle_lens[i] += 1

        self.dp_handle_lens = dp_handle_lens.copy()

        dest_dp_inputs = [[] for _ in range(args.dp)]
        # 分配每个dp 的原始输入和分配后的原始输入
        origin_datas = collections.deque()
        for origin_dp_index, origin_dp_input_len in enumerate(dp_input_lens.numpy()):
            handle_len = dp_handle_lens[origin_dp_index]
            if origin_dp_input_len > handle_len:
                origin_datas.append((origin_dp_index, handle_len, origin_dp_input_len))
                dp_handle_lens[origin_dp_index] = 0
                dest_dp_inputs[origin_dp_index].append((origin_dp_index, 0, handle_len))
            else:
                dp_handle_lens[origin_dp_index] -= origin_dp_input_len
                dest_dp_inputs[origin_dp_index].append((origin_dp_index, 0, origin_dp_input_len))

        for dest_dp_index in range(args.dp):
            need_size = dp_handle_lens[dest_dp_index]
            if need_size == 0:
                continue
            while len(origin_datas) != 0:
                origin_data = origin_datas.popleft()
                origin_dp_index, start, end = origin_data
                if end - start > need_size:
                    dest_dp_inputs[dest_dp_index].append((origin_dp_index, start, start + need_size))
                    origin_datas.appendleft((origin_dp_index, start + need_size, end))
                    break
                else:
                    dest_dp_inputs[dest_dp_index].append((origin_dp_index, start, end))
                    need_size -= end - start
                    if need_size == 0:
                        break

        dp_output_split_sizes = [[0 for _ in range(args.dp)] for _ in range(args.dp)]
        for dest_dp_index, dest_dp_data in enumerate(dest_dp_inputs):
            for origin_dp_index, start, end in dest_dp_data:
                dp_output_split_sizes[dest_dp_index][origin_dp_index] += end - start
        dp_input_split_sizes = [[0 for _ in range(args.dp)] for _ in range(args.dp)]
        for dest_dp_index, dest_dp_data in enumerate(dest_dp_inputs):
            for origin_dp_index, start, end in dest_dp_data:
                dp_input_split_sizes[origin_dp_index][dest_dp_index] += end - start

        self.dp_input_split_sizes = dp_input_split_sizes
        self.dp_output_split_sizes = dp_output_split_sizes

        new_input_ids = self._all_to_all_balance_get(input_ids)
        if hasattr(self, "position_ids") and self.position_ids is not None:
            # deepseekv2 mla 特殊模型需要保留原始的 position_ids, 用于减少通信量
            self._unbalance_position_ids = self.position_ids

            self.position_ids = self._all_to_all_balance_get(self.position_ids)
        if hasattr(self, "position_cos") and self.position_cos is not None:
            # deepseekv2 mla 特殊模型需要保留原始的 position_cos, 用于减少通信量
            self._unbalance_position_cos = self.position_cos

            self.position_cos = self._all_to_all_balance_get(self.position_cos)
        if hasattr(self, "position_sin") and self.position_sin is not None:
            # deepseekv2 mla 特殊模型需要保留原始的 position_sin, 用于减少通信量
            self._unbalance_position_sin = self.position_sin

            self.position_sin = self._all_to_all_balance_get(self.position_sin)

        return new_input_ids

    def _all_to_all_balance_get(self, data: torch.Tensor):
        dp_rank = get_global_dp_rank()
        import torch.distributed as dist
        from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager

        old_shape = data.shape
        data = data.view(-1)

        origin_len = self.dp_origin_lens[dp_rank]
        assert data.shape[0] % origin_len == 0
        scale_size = data.shape[0] // origin_len
        handle_len = self.dp_handle_lens[dp_rank]

        dest_data = g_cache_manager.alloc_tensor(
            shape=(handle_len * scale_size,),
            data_type=data.dtype,
            device="cuda",
        )
        dist.all_to_all_single(
            output=dest_data.view(-1),
            input=data.view(-1),
            output_split_sizes=[e * scale_size for e in self.dp_output_split_sizes[dp_rank]],
            input_split_sizes=[e * scale_size for e in self.dp_input_split_sizes[dp_rank]],
            group=self.dist_group.dp_prefill_balance_group,
            async_op=False,
        )
        return dest_data.view(-1, *old_shape[1:])

    def _all_to_all_unbalance_get(self, data: torch.Tensor):
        dp_rank = get_global_dp_rank()
        import torch.distributed as dist
        from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager

        old_shape = data.shape
        data = data.view(-1)

        handle_len = self.dp_handle_lens[dp_rank]
        scale_size = data.shape[0] // handle_len
        assert data.shape[0] % handle_len == 0
        origin_len = self.dp_origin_lens[dp_rank]
        origin_data = g_cache_manager.alloc_tensor(
            shape=(origin_len * scale_size,),
            data_type=data.dtype,
            device="cuda",
        )
        dist.all_to_all_single(
            output=origin_data.view(-1),
            input=data,
            output_split_sizes=[e * scale_size for e in self.dp_input_split_sizes[dp_rank]],
            input_split_sizes=[e * scale_size for e in self.dp_output_split_sizes[dp_rank]],
            group=self.dist_group.dp_prefill_balance_group,
            async_op=False,
        )
        return origin_data.view(-1, *old_shape[1:])

    # 用于 prefll cuda graph 的专用功能接口
    def prefill_cuda_graph_create_graph_obj(self):
        if not hasattr(self, "prefill_cuda_graph_exe_list"):
            self.prefill_cuda_graph_exe_list = []
        graph_obj = torch.cuda.CUDAGraph()
        capture_graph = torch.cuda.graph(graph_obj, pool=self.mem_pool)
        self.prefill_cuda_graph_exe_list.append((graph_obj, capture_graph))
        return

    def prefill_cuda_graph_get_current_capture_graph(self) -> torch.cuda.graph:
        assert len(self.prefill_cuda_graph_exe_list) > 0, "no cuda graph exe obj found"
        if isinstance(self.prefill_cuda_graph_exe_list[-1], tuple):
            return self.prefill_cuda_graph_exe_list[-1][1]
        else:
            return self.prefill_cuda_graph_exe_list[-2][1]

    def prefill_cuda_graph_add_cpu_runnning_func(self, func, after_graph: torch.cuda.graph):
        if not hasattr(self, "prefill_cuda_graph_exe_list"):
            self.prefill_cuda_graph_exe_list = []
        if after_graph is None:
            self.prefill_cuda_graph_exe_list.append(func)
            return

        for i, e in enumerate(self.prefill_cuda_graph_exe_list):
            if isinstance(e, tuple) and e[1] == after_graph:
                self.prefill_cuda_graph_exe_list.insert(i + 1, func)
                return
        assert False, "after_graph not found in prefill_cuda_graph_exe_list"

    def prefill_replay(self, new_infer_state: "InferStateInfo"):
        for func in self.prefill_cuda_graph_exe_list:
            if isinstance(func, tuple):
                graph_obj, _ = func
                graph_obj.replay()
            else:
                func(new_infer_state)
        return

    def copy_for_prefill_cuda_graph(self, new_infer_state: "InferStateInfo"):
        for attr_name, attr_value in vars(new_infer_state).items():
            if isinstance(attr_value, torch.Tensor):
                attr_ = getattr(self, attr_name, None)
                if attr_ is not None and attr_.data_ptr() != attr_value.data_ptr() and attr_.shape == attr_value.shape:
                    attr_.copy_(attr_value, non_blocking=True)
        return
