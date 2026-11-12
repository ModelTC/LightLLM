import collections
from contextlib import contextmanager
from dataclasses import replace
from typing import List, Optional

import torch

from lightllm.common.attention.api import AttentionPhase, ContextCapacity
from lightllm.common.attention_state.manager import AttentionStateManager
from lightllm.common.req_manager import ReqManager
from lightllm.distributed import CustomProcessGroup
from lightllm.utils.dist_utils import get_dp_world_size, get_global_dp_rank
from lightllm.utils.envs_utils import get_env_start_args

from .attention_batch import AttentionBatchBuilder
from .triton_kernel.gen_decode_params import gen_decode_params
from .triton_kernel.gen_prefill_params import gen_prefill_params


class InferStateInfo:
    """
    推理时用的信息结构体
    """

    def __init__(self):
        self.attention_groups = ()
        self.attention_contexts = {}
        self.attention_batch = None

        self.input_ids: torch.Tensor = None
        self.batch_size: int = None
        self.total_token_num: int = None
        self.b_req_idx: torch.Tensor = None
        self.b_ready_cache_len: torch.Tensor = None  # only for prefill prompt cache used.

        self.b_shared_seq_len: torch.Tensor = None  # raw decode radix-cache shared lengths.
        self.b_shared_radix_node_id: torch.Tensor = None  # raw decode radix-node ids.

        self.b_mtp_index: torch.Tensor = None
        # MRoPE position offset propagated from ModelInput.
        self.b_position_delta: torch.Tensor = None

        self.b_seq_len: torch.Tensor = None
        # max_cache_len 用于 prefill 阶段标识请求中最大 cache的kv 的长度
        self.max_cache_len: int = None
        self.is_prefill: bool = None
        self.is_draft_model: bool = False

        self.att_state_manager: AttentionStateManager = None
        self.att_state_role: str = "main"  # main 或 draft0 .. draft35，区分共享管理器中的 State。
        self.req_manager: ReqManager = None

        self.mem_index: torch.Tensor = None

        self.return_all_prompt_logics: bool = False
        # 在开启 return_all_prompt_logics 模式时，保存整个 prefill 阶段每一个
        # token 位置的 logits，供后续回传 prompt logprobs 信息使用。
        # 仅在 prefill 阶段且需要返回 prompt logprobs 时才会被填充。
        self.prompt_logics: Optional[torch.Tensor] = None
        self.multimodal_params: dict = None
        self.is_cuda_graph_capture: bool = False  # 标记是否用于 decode CUDA Graph 捕获
        self.is_cuda_graph_replay: bool = False  # 标记是否为 decode CUDA Graph 回放准备输入
        self.dist_group: CustomProcessGroup = None

        # 在microbatch overlap的运行模式下，用于标记当前 microbatch 的 index 序号
        # 在一些细节场景下需要有该信息区分一些资源的申请和管理。
        self.microbatch_index: int = 0

        # 当前 forward 独占的 hidden state 收集器。普通推理使用短生命周期实例；
        # Prefill CUDA Graph 使用随 graph infer state 长期保存的实例。
        self.hidden_collector = None

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

        # prefill 用
        self.b_q_start_loc: torch.Tensor = None
        # decode 用
        self.b_kv_start_loc: torch.Tensor = None

        # 一些特殊模型，特殊模式使用的输入变量，本身这些变量不适合放在
        # inferstate的基类中，但是为了代码的简洁和方便，都放在基类中
        # 进行管理。注意这些成员变量只会在特定的模型和模式下才会生效。

        # mtp draft model 使用的额外输入参数,
        # 在开启 mtp_mode 时，mtp draft model
        # 的输入会用到，其他模型和场景都不会用到
        self.mtp_draft_input_hiddens: Optional[torch.Tensor] = None

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

    def init_some_extra_state(self, model):
        if self.is_prefill:
            (
                self.b_q_seq_len,
                self.b1_cu_q_seq_len,
                self.b_kv_seq_len,
                self.b1_cu_kv_seq_len,
                self.position_ids,
            ) = gen_prefill_params(
                input_token_num=self.input_ids.shape[0],
                b_ready_cache_len=self.b_ready_cache_len,
                b_seq_len=self.b_seq_len,
            )
            self.b_q_start_loc = self.b1_cu_q_seq_len[0:-1]
        else:
            (
                self.b_q_seq_len,
                self.b1_cu_q_seq_len,
                self.b_kv_seq_len,
                self.b1_cu_kv_seq_len,
                self.position_ids,
            ) = gen_decode_params(self.b_seq_len)
            self.b_kv_start_loc = self.b1_cu_kv_seq_len[0:-1]

    def prepare_attention(self, model):
        from lightllm.common.basemodel.attention import get_attention_workspace

        self.device_runtime = model.device_runtime
        if model._attention_storage_generation != self.att_state_manager.storage_generation:
            raise RuntimeError("attention storage changed; rebind layers and recapture CUDA graphs")
        phase = AttentionPhase.PREFILL if self.is_prefill else AttentionPhase.DECODE
        self.attention_batch = AttentionBatchBuilder.from_infer_state(self)
        if self.is_cuda_graph_replay:
            return
        capacity = ContextCapacity(
            phase,
            self.batch_size,
            self.input_ids.shape[0],
            model.graph_max_len_in_batch if self.is_cuda_graph_capture else self.max_kv_seq_len,
            self.max_q_seq_len,
        )
        for group_id, group in enumerate(self.attention_groups):
            workspace = get_attention_workspace(model, group_id, self.microbatch_index, self.input_ids.device)
            workspace.begin()
            backend = group.backend_for(phase)
            if self.is_cuda_graph_capture:
                workspace = workspace.fork()
            workspace.reserve(backend.resource_requirements(capacity, capture=self.is_cuda_graph_capture))
            context = backend.allocate_context(capacity, capture=self.is_cuda_graph_capture, workspace=workspace)
            backend.prepare(context, self.attention_batch)
            self.attention_contexts[group_id] = context

    def attention_context(self, binding):
        context = self.attention_contexts[binding.group_id]
        if context.backend_identity is not binding.backend or context.capacity.phase is not binding.spec.phase:
            raise ValueError("layer binding does not belong to this attention execution")
        return context

    def update_attention(self):
        for group_id, context in self.attention_contexts.items():
            self.attention_groups[group_id].backend_for(context.capacity.phase).update(context)

    def finish_attention(self):
        for context in self.attention_contexts.values():
            context.workspace.finish()

    def copy_for_cuda_graph(self, new_infer_state: "InferStateInfo"):
        if (
            self.att_state_role != new_infer_state.att_state_role
            or len(self.attention_groups) != len(new_infer_state.attention_groups)
            or any(
                current is not incoming
                for current, incoming in zip(self.attention_groups, new_infer_state.attention_groups)
            )
        ):
            raise ValueError("captured attention belongs to a different model role or group")
        copies = []
        batch_tensor_names = {
            name for name, value in vars(self.attention_batch).items() if isinstance(value, torch.Tensor)
        }
        for attr_name, attr_value in vars(new_infer_state).items():
            if isinstance(attr_value, torch.Tensor) and attr_name not in batch_tensor_names:
                target = getattr(self, attr_name, None)
                if target is not None and target.data_ptr() != attr_value.data_ptr():
                    if (target.shape, target.dtype, target.device) != (
                        attr_value.shape,
                        attr_value.dtype,
                        attr_value.device,
                    ):
                        raise ValueError(f"captured input {attr_name} changed shape, dtype or device")
                    copies.append((target, attr_value))
        for context in self.attention_contexts.values():
            context.workspace.begin()
        self.attention_batch.copy_inputs(new_infer_state.attention_batch)
        for target, source in copies:
            target.copy_(source, non_blocking=True)
        batch = replace(
            self.attention_batch,
            max_kv_seq_len=new_infer_state.max_kv_seq_len,
            max_q_seq_len=new_infer_state.max_q_seq_len,
            total_token_num=new_infer_state.total_token_num,
        )
        for group_id, context in self.attention_contexts.items():
            self.attention_groups[group_id].decode_backend.prepare(context, batch)
        return

    def call_overlap_hook(self):
        """
        overlap_hook 是在 microbatch overlap 的运行模式下，用于调用绑定在inferstate上的hook函数，用于
        实现折叠通信和计算的效果，普通模式并不调用这个函数。这是个快速调用函数，用于减少重复代码。
        """
        if getattr(self, "hook", None) is not None:
            self.hook()
            self.hook = None
        return

    ##### prefill dp balance 相关的函数 #####

    def prepare_prefill_dp_balance(self):
        """
        在prefill的时候, 对于处于 dp 模式下的时候，对输入的数据进行重新的调整和分配，降低各个dp处理数据量过于不一致的时候,导致
        的prefill 推理性能下降
        """
        assert self.is_prefill
        import torch.distributed as dist

        input_ids = self.input_ids  # 原始输入的input_ids

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
        dp_input_lens = dp_input_lens.detach().cpu().tolist()
        self.dp_origin_lens = dp_input_lens.copy()
        sum_input_len = sum(dp_input_lens)
        if not args.enable_tpsp_mix_mode:
            dp_handle_lens = [sum_input_len // args.dp for _ in range(args.dp)]
            for i in range(sum_input_len % args.dp):
                dp_handle_lens[i] += 1
        else:
            # tpsp mix mode 需要让每个dp 的处理长度是 tp_world_size 的整数倍
            tp_world_size = get_dp_world_size()
            assert all(e % tp_world_size == 0 for e in dp_input_lens)
            _dp_input_lens = [e // tp_world_size for e in dp_input_lens]
            _sum_input_len = sum(_dp_input_lens)
            _dp_input_lens = [_sum_input_len // args.dp for _ in range(args.dp)]
            for i in range(_sum_input_len % args.dp):
                _dp_input_lens[i] += 1
            dp_handle_lens = [_dp_input_len * tp_world_size for _dp_input_len in _dp_input_lens]
            assert sum(dp_handle_lens) == sum_input_len

        self.dp_handle_lens = dp_handle_lens.copy()

        dest_dp_inputs = [[] for _ in range(args.dp)]
        # 分配每个dp 的原始输入和分配后的原始输入
        origin_datas = collections.deque()
        for origin_dp_index, origin_dp_input_len in enumerate(dp_input_lens):
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

        if hasattr(self, "position_ids") and self.position_ids is not None:
            # deepseekv2 mla 特殊模型需要保留原始的 position_ids, 用于减少通信量
            self._unbalance_position_ids = self.position_ids
            self._balance_position_ids = self._all_to_all_balance_get(self.position_ids, change_state=False)

        if hasattr(self, "position_cos") and self.position_cos is not None:
            # deepseekv2 mla 特殊模型需要保留原始的 position_cos, 用于减少通信量
            self._unbalance_position_cos = self.position_cos
            self._balance_position_cos = self._all_to_all_balance_get(self.position_cos, change_state=False)

        if hasattr(self, "position_sin") and self.position_sin is not None:
            # deepseekv2 mla 特殊模型需要保留原始的 position_sin, 用于减少通信量
            self._unbalance_position_sin = self.position_sin
            self._balance_position_sin = self._all_to_all_balance_get(self.position_sin, change_state=False)

        self._unbalance_input_ids = self.input_ids
        self._balance_input_ids = self._all_to_all_balance_get(input_ids, change_state=False)

        return

    def __change_to_unbalance(self):
        self.input_ids = self._unbalance_input_ids
        if hasattr(self, "position_ids"):
            self.position_ids = self._unbalance_position_ids
        if hasattr(self, "position_cos"):
            self.position_cos = self._unbalance_position_cos
        if hasattr(self, "position_sin"):
            self.position_sin = self._unbalance_position_sin
        return

    def __change_to_balance(self):
        self.input_ids = self._balance_input_ids
        if hasattr(self, "position_ids"):
            self.position_ids = self._balance_position_ids
        if hasattr(self, "position_cos"):
            self.position_cos = self._balance_position_cos
        if hasattr(self, "position_sin"):
            self.position_sin = self._balance_position_sin
        return

    def _all_to_all_balance_get(self, data: torch.Tensor, change_state: bool = True):
        if change_state:
            self.__change_to_balance()
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

    def _all_to_all_unbalance_get(self, data: torch.Tensor, change_state: bool = True):
        if change_state:
            self.__change_to_unbalance()

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

    # 用于 prefill cuda graph 的专用功能接口
    @contextmanager
    def _capture_prefill_segment(self, graph_obj):
        with self.device_runtime.capture_graph(graph_obj, pool=self.mem_pool):
            self._prefill_capture_active = True
            try:
                yield
            finally:
                self._prefill_capture_active = False

    def prefill_cuda_graph_create_graph_obj(self):
        if not hasattr(self, "prefill_cuda_graph_exe_list"):
            self.prefill_cuda_graph_exe_list = []
        graph_obj = self.device_runtime.create_graph()
        capture_graph = self._capture_prefill_segment(graph_obj)
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

    def close_prefill_graphs(self):
        try:
            # 错误可能发生在某一段内部，也可能发生在段间的 CPU 回调中。
            # 只退出仍在捕获的上下文，避免重复结束已经完成的段。
            if getattr(self, "_prefill_capture_active", False):
                self.prefill_cuda_graph_get_current_capture_graph().__exit__(None, None, None)
        finally:
            self.device_runtime.synchronize()
            for item in getattr(self, "prefill_cuda_graph_exe_list", ()):
                if isinstance(item, tuple):
                    item[0].reset()
            self.prefill_cuda_graph_exe_list = []

    def copy_for_prefill_cuda_graph(self, new_infer_state: "InferStateInfo"):
        for attr_name, attr_value in vars(new_infer_state).items():
            if isinstance(attr_value, torch.Tensor):
                attr_ = getattr(self, attr_name, None)
                if attr_ is not None and attr_.data_ptr() != attr_value.data_ptr() and attr_.shape == attr_value.shape:
                    attr_.copy_(attr_value, non_blocking=True)
        return
