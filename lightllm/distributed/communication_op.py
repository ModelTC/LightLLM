# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.6.3.post1/vllm/distributed/communication_op.py
# of the vllm-project/vllm GitHub repository.
#
# Copyright 2023 ModelTC Team
# Copyright 2023 vLLM Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.distributed as dist
from torch.distributed import ReduceOp, ProcessGroup
from typing import List, Dict, Optional, Set, Union
from lightllm.utils.log_utils import init_logger
from lightllm.utils.device_utils import has_nvlink
from lightllm.utils.envs_utils import (
    get_env_start_args,
    get_deepep_num_max_dispatch_tokens_per_rank,
)
from lightllm.utils.dist_utils import (
    get_global_world_size,
    get_dp_world_size,
    create_new_group_for_current_dp,
    create_dp_special_inter_group,
)
from lightllm.utils.device_utils import get_device_sm_count, is_sm100_gpu
from lightllm.utils.torch_dtype_utils import get_torch_dtype

logger = init_logger(__name__)


try:
    import deep_ep

    HAS_DEEPEP = True
except:
    HAS_DEEPEP = False
    logger.info("deep_ep is not installed, you can't use the api of it.")


class CustomProcessGroup:
    def __init__(self):
        self.symm_mem_reduce = None
        self.flashinfer_reduce = None
        self.dp_world_size = get_dp_world_size()
        self.device_group = create_new_group_for_current_dp("nccl")
        if get_env_start_args().enable_dp_prefill_balance:
            self.dp_prefill_balance_group = create_dp_special_inter_group("nccl")
        else:
            self.dp_prefill_balance_group = None

        self.autotune_group = dist.new_group([i for i in range(get_global_world_size())], backend="gloo")

    def _support_custom_allreduce(self) -> bool:
        return has_nvlink() and self.dp_world_size in [2, 4, 6, 8]

    def init_symm_mem_reduce(self) -> None:
        if not self._support_custom_allreduce():
            return
        from .symm_mem_all_reduce import SymmMemAllreduce

        data_type = get_torch_dtype(get_env_start_args().data_type)
        symm = SymmMemAllreduce(self.device_group, torch.cuda.current_device(), dtype=data_type)
        if not symm.disabled:
            self.symm_mem_reduce = symm
            logger.info("Enable SymmMem ALLReduce.")

    def init_flashinfer_reduce(self) -> None:
        if not self._support_custom_allreduce():
            return
        from .flashinfer_all_reduce import FlashInferAllReduce

        fi_cpu_group = create_new_group_for_current_dp("gloo")
        fi = FlashInferAllReduce(fi_cpu_group, torch.cuda.current_device())
        if not fi.disabled:
            self.flashinfer_reduce = fi
            logger.info("Enable FlashInfer ALLReduce.")

    def all_reduce(self, input_: torch.Tensor) -> None:
        # Dispatch chain: FlashInfer -> SymmMem -> NCCL.
        if self.flashinfer_reduce is not None and self.flashinfer_reduce.should_use(input_):
            input_.data = self.flashinfer_reduce.all_reduce(input_)
            return
        if self.symm_mem_reduce is not None and self.symm_mem_reduce.should_use(input_):
            self.symm_mem_reduce.all_reduce(input_)
            return
        return dist.all_reduce(input_, group=self.device_group)

    def all_gather_into_tensor(self, output_: torch.Tensor, input_: torch.Tensor, async_op: bool = False) -> None:
        return dist.all_gather_into_tensor(output_, input_, group=self.device_group, async_op=async_op)


class DistributeGroupManager:
    def __init__(self):
        self.groups = []
        self.ep_buffer = None
        self.ep_mega_moe_buffer = None
        self.ep_num_sms = None
        self.ep_non_overlap_num_sms = None

    def __len__(self):
        return len(self.groups)

    def create_groups(self, group_size: int):
        args = get_env_start_args()
        for i in range(group_size):
            group = CustomProcessGroup()
            if not args.disable_symm_mem_allreduce:
                group.init_symm_mem_reduce()
            if not args.disable_flashinfer_allreduce:
                group.init_flashinfer_reduce()
            self.groups.append(group)
        return

    def get_default_group(self) -> CustomProcessGroup:
        return self.groups[0]

    def get_group(self, group_index: int) -> CustomProcessGroup:
        return self.groups[group_index]

    @staticmethod
    def get_moe_quant_methods(layer_weights: List) -> Set[str]:
        """收集实际绑定到各 MoE 层 expert weight 上的量化方法名称。

        expert 量化类型可能分别来自启动参数、quant_cfg 和模型 config。调用本函数
        时 layer weights 已经构造完成，每层 ``experts.quant_method`` 保存的是按照
        既定优先级解析后的最终结果，因此这里不再重复解析配置。

        返回方法名称集合是为了去重并支持混合量化。例如部分 MoE 层使用 FP4、
        其余层使用 FP8 时，可以据此同时初始化两条执行路径所需的 buffer；普通
        dense 层没有 ``experts``，会被自然跳过。
        """
        quant_method_names = set()
        for layer_weight in layer_weights:
            # dense 层没有 experts；这里只关心真正参与 MoE 计算的层。
            experts = getattr(layer_weight, "experts", None)
            quant_method = getattr(experts, "quant_method", None)
            method_name = getattr(quant_method, "method_name", None)
            if method_name is not None:
                quant_method_names.add(method_name)
        return quant_method_names

    def new_deepep_group(
        self,
        n_routed_experts,
        hidden_size,
        expert_quant_method_names: Set[str],
        num_experts_per_tok: int = 1,
        moe_intermediate_size: Optional[int] = None,
    ):
        """初始化 DeepEP 通信组以及当前模型实际需要的 MoE buffer。

        ``expert_quant_method_names`` 是各 MoE 层最终绑定的 quant method 名称集合。
        同一个模型可能逐层混用 FP4 和 FP8：SM100 FP4 层走 Mega MoE，其他层走
        DeepEP Elastic 路径。这里只为实际存在的执行路径分配 buffer，
        避免为未使用的路径长期占用显存。
        """
        enable_ep_moe = get_env_start_args().enable_ep_moe
        if not enable_ep_moe:
            self.ep_buffer = None
            self.ep_mega_moe_buffer = None
            self.ep_num_sms = None
            self.ep_non_overlap_num_sms = None
            return
        assert HAS_DEEPEP, "deep_ep is required for expert parallelism"

        # ElasticBuffer 按当前服务真正会执行的阶段选择容量。
        num_max_dispatch_tokens_per_rank = get_deepep_num_max_dispatch_tokens_per_rank()

        import deep_gemm
        from lightllm.common.basemodel.triton_kernel.fused_moe import grouped_fused_moe_ep

        # LightLLM 的 prefill expanded layout 当前按 128 行构造 expert 分段。
        # 在初始化阶段同步设置 DeepGEMM 的全局 M 对齐值，后续 prefill 和
        # decode 都通过 getter 读取它，避免通信布局与 GEMM 分块不一致。
        grouped_fused_moe_ep.set_mk_alignment_for_contiguous_layout(128)

        global_world_size = get_global_world_size()
        deepep_group = dist.new_group(list(range(global_world_size)))
        # DeepEP reuses this group's NCCL communicator via _comm_ptr(). Because the
        # group is created without device_id, warm it up first to avoid reading a null
        # communicator. The default process group's warmup does not cover this group.
        dist.barrier(
            group=deepep_group,
            device_ids=[torch.cuda.current_device()],
        )
        self.ll_num_tokens = num_max_dispatch_tokens_per_rank
        self.ll_hidden = hidden_size
        total_redundant_experts = get_env_start_args().eplb_num_redundant_experts_per_rank * global_world_size
        self.ll_num_experts = n_routed_experts + total_redundant_experts
        self.ep_buffer = deep_ep.ElasticBuffer(
            deepep_group,
            num_max_tokens_per_rank=self.ll_num_tokens,
            hidden=self.ll_hidden,
            num_topk=num_experts_per_tok,
            use_fp8_dispatch=True,
            allow_multiple_reduction=True,
        )
        self.ep_mega_moe_buffer = None

        if not expert_quant_method_names:
            raise ValueError("No valid MoE quant method was found while initializing DeepEP buffers")

        mega_moe_quant_method = "fp4fp8-b32-deepgemm"
        # SM100 FP4 层额外通过 DeepGEMM Mega MoE 完成通信和计算。
        enable_mega_moe_buffer = is_sm100_gpu() and mega_moe_quant_method in expert_quant_method_names

        if enable_mega_moe_buffer:
            if moe_intermediate_size is None:
                raise ValueError("SM100 Mega MoE requires moe_intermediate_size or intermediate_size in model config")

            self.ep_mega_moe_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
                deepep_group,
                self.ll_num_experts,
                self.ll_num_tokens,
                num_experts_per_tok,
                self.ll_hidden,
                moe_intermediate_size,
            )
        logger.info(
            "Initialize DeepEP MoE buffers: elastic=True, mega_moe=%s, "
            "num_max_dispatch_tokens_per_rank=%s, ll_num_experts=%s, expert_quant_method_names=%s",
            enable_mega_moe_buffer,
            self.ll_num_tokens,
            self.ll_num_experts,
            sorted(expert_quant_method_names),
        )
        theoretical_sms = self.ep_buffer.get_theoretical_num_sms(self.ll_num_experts, num_experts_per_tok)
        self._set_num_sms_for_deep_gemm(theoretical_sms)

        # 串行路径会在通信后立即等待，不需要为并行计算预留大部分 SM。
        # DeepEP 要求 num_sms 为偶数，因此将设备总 SM 数的 60% 向下取偶数。
        device_sms = get_device_sm_count()
        non_overlap_num_sms = int(device_sms * 0.6) // 2 * 2
        self.set_ep_num_sms_for_non_overlap(non_overlap_num_sms)
        logger.info(
            "DeepEP SM allocation: overlap=%s, non_overlap=%s, device=%s",
            self.ep_num_sms,
            self.ep_non_overlap_num_sms,
            device_sms,
        )

    def set_ep_num_sms_for_non_overlap(self, num_sms: int) -> None:
        """设置非 overlap 通信使用的 SM 数量，不改变 DeepGEMM 的 SM 配额。"""
        device_sms = get_device_sm_count()
        assert 4 <= num_sms <= device_sms
        assert num_sms % 2 == 0
        self.ep_non_overlap_num_sms = num_sms

    def _set_num_sms_for_deep_gemm(self, deepep_sms: int):
        try:
            try:
                from deep_gemm.jit_kernels.utils import set_num_sms
            except:
                from deep_gemm import set_num_sms

            device_sms = get_device_sm_count()
            deepep_sms = max(0, min(deepep_sms, max(device_sms - 2, 0)))
            self.ep_num_sms = deepep_sms
            set_num_sms(max(device_sms - deepep_sms, 2))
        except BaseException as e:
            logger.warning(f"set num sms for deep_gemm failed: {e}")

    def clear_deepep_buffer(self) -> None:
        """清理通信组件在当前请求中产生的临时状态。"""
        # 当前通信组件不需要显式清理。保留统一入口，后续接入需要清理的
        # 通信实现时，无需再修改模型执行流程。
        pass


def all_reduce(
    input_: torch.Tensor,
    group: Optional[Union[ProcessGroup, CustomProcessGroup]] = None,
    op: ReduceOp = ReduceOp.SUM,
    async_op: bool = False,
) -> None:
    if _is_single_group(group=group):
        return
    if isinstance(group, CustomProcessGroup):
        if op == ReduceOp.SUM:
            return group.all_reduce(input_)
        return dist.all_reduce(input_, op, group.device_group, async_op)
    return dist.all_reduce(input_, op, group, async_op)


def all_gather_into_tensor(
    output_: torch.Tensor,
    input_: torch.Tensor,
    group: Optional[Union[ProcessGroup, CustomProcessGroup]] = None,
    async_op: bool = False,
) -> None:
    if _is_single_group(group=group):
        output_.copy_(input_)
        return
    if isinstance(group, CustomProcessGroup):
        return group.all_gather_into_tensor(output_, input_)
    else:
        return dist.all_gather_into_tensor(output_, input_, group, async_op)


def all_gather(
    output_: List[torch.Tensor],
    input_: torch.Tensor,
    group: Optional[Union[ProcessGroup, CustomProcessGroup]] = None,
    async_op: bool = False,
) -> None:
    if _is_single_group(group=group):
        if len(output_) > 0:
            output_[0].copy_(input_)
        return
    # todo 目前还没有定制算子的支持。
    if isinstance(group, CustomProcessGroup):
        return dist.all_gather(output_, input_, group.device_group, async_op)
    else:
        return dist.all_gather(output_, input_, group, async_op)


def reduce_scatter_tensor(
    output: torch.Tensor,
    input: torch.Tensor,
    op: ReduceOp = ReduceOp.SUM,
    group: Optional[Union[ProcessGroup, CustomProcessGroup]] = None,
    async_op=False,
):
    if _is_single_group(group=group):
        output.copy_(input)
        return
    # 目前还没有定制算子实现。
    if isinstance(group, CustomProcessGroup):
        return dist.reduce_scatter_tensor(output, input, op=op, group=group.device_group, async_op=async_op)
    else:
        return dist.reduce_scatter_tensor(output, input, op=op, group=group, async_op=async_op)


def broadcast(
    tensor: torch.Tensor,
    src: int,
    group: Optional[Union[ProcessGroup, CustomProcessGroup]] = None,
    async_op: bool = False,
) -> None:
    if _is_single_group(group=group):
        return
    if isinstance(group, CustomProcessGroup):
        return dist.broadcast(tensor, src=src, group=group.device_group, async_op=async_op)
    else:
        return dist.broadcast(tensor, src=src, group=group, async_op=async_op)


def _is_single_group(group: Optional[Union[ProcessGroup, CustomProcessGroup]]) -> bool:
    if isinstance(group, CustomProcessGroup):
        return group.dp_world_size == 1
    else:
        return dist.get_world_size(group=group) == 1


dist_group_manager = DistributeGroupManager()
