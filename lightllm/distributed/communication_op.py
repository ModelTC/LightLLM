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


import os
import torch
import triton
import torch.distributed as dist
from torch.distributed import ReduceOp, ProcessGroup
from typing import List, Dict, Optional, Set, Union
from lightllm.utils.log_utils import init_logger
from lightllm.utils.device_utils import has_nvlink
from lightllm.utils.envs_utils import (
    get_env_start_args,
    get_deepep_num_max_dispatch_tokens_per_rank_prefill,
    get_deepep_num_max_dispatch_tokens_per_rank_decode,
    get_redundancy_expert_num,
)
from lightllm.utils.dist_utils import (
    get_global_world_size,
    get_dp_world_size,
    create_new_group_for_current_dp,
    create_dp_special_inter_group,
)
from lightllm.utils.device_utils import get_device_sm_count, is_sm90_gpu, is_sm100_gpu
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
        self.symm_mem_out_of_place = os.getenv("LIGHTLLM_SYMM_MEM_OUT_OF_PLACE", "0").upper() in {"1", "ON", "TRUE"}
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
            if self.symm_mem_out_of_place:
                input_.data = self.symm_mem_reduce.all_reduce_out_of_place(input_)
            else:
                self.symm_mem_reduce.all_reduce(input_)
            return
        return dist.all_reduce(input_, group=self.device_group)

    def all_gather_into_tensor(self, output_: torch.Tensor, input_: torch.Tensor, async_op: bool = False) -> None:
        return dist.all_gather_into_tensor(output_, input_, group=self.device_group, async_op=async_op)


class DistributeGroupManager:
    def __init__(self):
        self.groups = []
        self.ep_buffer = None
        self.ep_prefill_uses_legacy_buffer = False
        self.ep_low_latency_buffer = None
        self.ep_mega_moe_buffer = None
        self.ep_num_sms = None

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
        DeepEP legacy low-latency 路径。这里只为实际存在的执行路径分配 buffer，
        避免为未使用的路径长期占用显存。
        """
        enable_ep_moe = get_env_start_args().enable_ep_moe
        prefill_num_max_dispatch_tokens_per_rank = get_deepep_num_max_dispatch_tokens_per_rank_prefill()
        decode_num_max_dispatch_tokens_per_rank = get_deepep_num_max_dispatch_tokens_per_rank_decode()
        if not enable_ep_moe:
            self.ep_buffer = None
            self.ep_prefill_uses_legacy_buffer = False
            self.ep_low_latency_buffer = None
            self.ep_mega_moe_buffer = None
            self.ep_num_sms = None
            return
        assert HAS_DEEPEP, "deep_ep is required for expert parallelism"

        global_world_size = get_global_world_size()
        deepep_group = dist.new_group(list(range(global_world_size)))
        # DeepEP reuses this group's NCCL communicator via _comm_ptr(). Because the
        # group is created without device_id, warm it up first to avoid reading a null
        # communicator. The default process group's warmup does not cover this group.
        dist.barrier(
            group=deepep_group,
            device_ids=[torch.cuda.current_device()],
        )
        self.ll_num_tokens = prefill_num_max_dispatch_tokens_per_rank
        self.ll_decode_num_tokens = decode_num_max_dispatch_tokens_per_rank
        self.ll_hidden = hidden_size
        self.ll_num_experts = n_routed_experts + get_redundancy_expert_num() * global_world_size
        # ElasticBuffer uses NCCL GIN even for an intra-node group.  GIN is not
        # available on many otherwise fully-connected NVLink hosts.  The legacy
        # normal DeepEP buffer is the native intra-node transport and avoids that
        # unnecessary network dependency; keep ElasticBuffer for multi-node EP.
        self.ep_prefill_uses_legacy_buffer = get_env_start_args().nnodes == 1
        if self.ep_prefill_uses_legacy_buffer:
            hidden_bytes = self.ll_hidden * get_torch_dtype(get_env_start_args().data_type).itemsize
            dispatch_config = deep_ep.Buffer.get_dispatch_config(global_world_size)
            combine_config = deep_ep.Buffer.get_combine_config(global_world_size)
            num_nvl_bytes = max(
                dispatch_config.get_nvl_buffer_size_hint(hidden_bytes, global_world_size),
                combine_config.get_nvl_buffer_size_hint(hidden_bytes, global_world_size),
            )
            num_rdma_bytes = max(
                dispatch_config.get_rdma_buffer_size_hint(hidden_bytes, global_world_size),
                combine_config.get_rdma_buffer_size_hint(hidden_bytes, global_world_size),
            )
            self.ep_buffer = deep_ep.Buffer(
                deepep_group,
                num_nvl_bytes=num_nvl_bytes,
                num_rdma_bytes=num_rdma_bytes,
                low_latency_mode=False,
            )
        else:
            self.ep_buffer = deep_ep.ElasticBuffer(
                deepep_group,
                num_max_tokens_per_rank=self.ll_num_tokens,
                hidden=self.ll_hidden,
                num_topk=num_experts_per_tok,
                use_fp8_dispatch=True,
                allow_multiple_reduction=True,
            )
        self.ep_mega_moe_buffer = None
        self.ep_low_latency_buffer = None

        if not expert_quant_method_names:
            raise ValueError("No valid MoE quant method was found while initializing DeepEP buffers")

        mega_moe_quant_method = "fp4fp8-b32-deepgemm"
        sm90_mega_moe_quant_method = "fp8w8a8-b128-deepgemm"
        is_sm100 = is_sm100_gpu()
        enable_sm90_mega_moe = is_sm90_gpu() and os.getenv("LIGHTLLM_ENABLE_SM90_MEGA_MOE", "0").upper() in {
            "1",
            "ON",
            "TRUE",
        }

        # Buffer 选择规则：
        # 1. 非 SM100 不支持 Mega MoE，只初始化 legacy low-latency buffer；
        # 2. SM100 全部 MoE 层为 FP4，只初始化 Mega MoE buffer；
        # 3. SM100 全部 MoE 层为 FP8，只初始化 legacy low-latency buffer；
        # 4. SM100 逐层混合 FP4/FP8，两套 buffer 都要初始化。
        if is_sm100:
            # 只要存在一个 FP4 MoE 层，就需要 Mega MoE buffer；只要存在一个非 FP4
            # MoE 层，就需要 legacy low-latency buffer。FP4/FP8 逐层混用时两者都会初始化。
            has_mega_moe_layer = mega_moe_quant_method in expert_quant_method_names
            has_legacy_moe_layer = any(
                method_name != mega_moe_quant_method for method_name in expert_quant_method_names
            )
            enable_mega_moe_buffer = has_mega_moe_layer
            enable_low_latency_buffer = has_legacy_moe_layer
        elif enable_sm90_mega_moe:
            has_mega_moe_layer = sm90_mega_moe_quant_method in expert_quant_method_names
            has_legacy_moe_layer = any(
                method_name != sm90_mega_moe_quant_method for method_name in expert_quant_method_names
            )
            enable_mega_moe_buffer = has_mega_moe_layer
            enable_low_latency_buffer = has_legacy_moe_layer
        else:
            enable_mega_moe_buffer = False
            enable_low_latency_buffer = True

        if enable_low_latency_buffer:
            # FP8 MoE 的 decode 使用 legacy low-latency buffer；prefill 阶段还会将其
            # 空闲的本地 RDMA storage 复用为分块 grouped GEMM 的临时 workspace。
            decode_size_hint = deep_ep.Buffer.get_low_latency_rdma_size_hint(
                self.ll_decode_num_tokens, self.ll_hidden, global_world_size, self.ll_num_experts
            )
            microbatch_count = len(self.groups)
            min_prefill_reuse_buffer_bytes = _calculate_min_chunked_expanded_moe_reuse_buffer_bytes(
                global_world_size=global_world_size,
                prefill_tokens_per_rank=prefill_num_max_dispatch_tokens_per_rank,
                hidden_size=hidden_size,
                moe_intermediate_size=moe_intermediate_size,
                hidden_dtype=get_torch_dtype(get_env_start_args().data_type),
                microbatch_count=microbatch_count,
            )
            # Decode 和 prefill 不会同时使用 local RDMA storage，容量取两条路径的较大值。
            num_rdma_bytes = max(decode_size_hint, min_prefill_reuse_buffer_bytes)
            # DeepEP 返回的 decode hint 不保证能被 microbatch 均分；最终再对齐一次，
            # 确保每个 workspace slice 的容量及起始位置仍保持 256-byte 对齐。
            rdma_alignment = microbatch_count * 256
            num_rdma_bytes = triton.cdiv(num_rdma_bytes, rdma_alignment) * rdma_alignment
            self.ep_low_latency_buffer = deep_ep.Buffer(
                deepep_group,
                num_rdma_bytes=num_rdma_bytes,
                low_latency_mode=True,
                num_qps_per_rank=(self.ll_num_experts // global_world_size),
            )

        if enable_mega_moe_buffer:
            # SM100 FP4 层通过 DeepGEMM Mega MoE 完成通信和计算，不使用 legacy
            # low-latency buffer，因此纯 FP4 模型无需承担后者的大块 RDMA 显存。
            if moe_intermediate_size is None:
                raise ValueError("SM100 Mega MoE requires moe_intermediate_size or intermediate_size in model config")

            import deep_gemm

            mega_moe_kwargs = {}
            if enable_sm90_mega_moe:
                mega_moe_kwargs.update(use_fp8_dispatch=True, activation="swiglu")
            self.ep_mega_moe_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
                deepep_group,
                self.ll_num_experts,
                self.ll_num_tokens,
                num_experts_per_tok,
                self.ll_hidden,
                moe_intermediate_size,
                **mega_moe_kwargs,
            )
        logger.info(
            "Initialize DeepEP MoE buffers: legacy_prefill=%s, low_latency=%s, mega_moe=%s, "
            "expert_quant_method_names=%s",
            self.ep_prefill_uses_legacy_buffer,
            enable_low_latency_buffer,
            enable_mega_moe_buffer,
            sorted(expert_quant_method_names),
        )
        theoretical_sms = (
            0
            if enable_mega_moe_buffer and not enable_low_latency_buffer
            else (
                deep_ep.Buffer.num_sms
                if self.ep_prefill_uses_legacy_buffer
                else self.ep_buffer.get_theoretical_num_sms(self.ll_num_experts, num_experts_per_tok)
            )
        )
        self._set_num_sms_for_deep_gemm(theoretical_sms)

    def _set_num_sms_for_deep_gemm(self, deepep_sms: int):
        try:
            try:
                from deep_gemm.jit_kernels.utils import set_num_sms
            except:
                from deep_gemm import set_num_sms

            device_sms = get_device_sm_count()
            deepep_sms = max(0, min(deepep_sms, max(device_sms - 2, 0)))
            self.ep_num_sms = deepep_sms
            if self.ep_low_latency_buffer is not None:
                deep_ep.Buffer.set_num_sms(deepep_sms - deepep_sms % 2)
            set_num_sms(max(device_sms - deepep_sms, 2))
        except BaseException as e:
            logger.warning(f"set num sms for deep_gemm failed: {e}")

    def get_deep_ep_prefill_moe_workspace(self, microbatch_index: int = 0) -> torch.Tensor:
        """Return a slice of the workspace reused by DeepEP prefill MoE kernels.

        DeepEP's low-latency RDMA buffer is idle during prefill, so its local
        storage is reused as temporary workspace for the expanded MoE compute
        path to reduce peak GPU memory. With one communication group, the
        default ``microbatch_index=0`` receives the whole workspace. With
        multiple groups, the workspace is split into ``len(self.groups)``
        equal slices and each in-flight microbatch uses the slice matching its
        group index.

        Args:
            microbatch_index: Zero-based microbatch and communication-group
                index assigned to this in-flight prefill computation.

        Returns:
            A one-dimensional uint8 tensor view over the selected workspace
            slice; no additional GPU memory is allocated.

        This workspace is only valid after the DeepEP group has been
        initialized. The same returned slice must not be used concurrently by
        overlapping calls.
        """
        assert self.ep_low_latency_buffer is not None, "DeepEP low-latency buffer is not initialized"
        workspace = self.ep_low_latency_buffer.get_local_buffer_tensor(torch.uint8, use_rdma_buffer=True)
        microbatch_count = len(self.groups)
        assert 0 <= microbatch_index < microbatch_count
        workspace_size = workspace.numel() // microbatch_count
        return workspace.narrow(0, microbatch_index * workspace_size, workspace_size)

    def clear_deepep_buffer(self):
        """
        Prefill MoE compute reuses the low-latency RDMA buffer as workspace.
        Clean it before the buffer is used by low-latency decode kernels.
        """
        if self.ep_low_latency_buffer is not None:
            self.ep_low_latency_buffer.clean_low_latency_buffer(
                self.ll_decode_num_tokens, self.ll_hidden, self.ll_num_experts
            )


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


def _calculate_min_chunked_expanded_moe_reuse_buffer_bytes(
    global_world_size: int,
    prefill_tokens_per_rank: int,
    hidden_size: int,
    moe_intermediate_size: int,
    hidden_dtype: torch.dtype,
    microbatch_count: int,
) -> int:
    """计算 ``chunked_expanded_moe_forward`` 极端情况下所需的最小复用 buffer。

    Prefill 阶段会将空闲的 legacy DeepEP local RDMA storage 复用为 expanded
    grouped GEMM 的 workspace。该函数只计算这条 prefill 路径的容量下限，不包含
    low-latency decode 自身所需的 RDMA 容量。

    每个 prefill workspace 由全程常驻的 dense gather 输出和分块计算的临时峰值组成。
    gather 按 ``global_world_size * prefill_tokens_per_rank`` 估算最大可能接收行数，
    并向 1024 行对齐；这与运行期 workspace 探测使用的保守分桶保持一致。临时计算
    按最小 128 行 chunk 估算：W1 阶段同时需要 ``gemm_out_a`` 和 ``silu_out``；后续
    量化/W2 阶段涉及 ``silu_out``、FP8 量化结果、scale 及 ``gemm_out_b``。这里有意
    沿用保守公式，避免因生命周期估计不足而低估空间。

    多个 microbatch/communication group 并行时，每组独占一个 workspace slice，故
    prefill 容量乘以 ``microbatch_count``。最后按 ``microbatch_count * 256`` 对齐，
    保证均分后每个 slice 仍满足 256-byte 对齐。
    """

    def align(value: int, alignment: int) -> int:
        return triton.cdiv(value, alignment) * alignment

    def tensor_bytes(rows: int, columns: int, itemsize: int) -> int:
        # 沿用TensorBufferManager的256对齐规则
        return align(rows * columns * itemsize, 256)

    chunk_rows = 128
    hidden_itemsize = hidden_dtype.itemsize
    scale_cols = moe_intermediate_size // 128

    silu_bytes = tensor_bytes(chunk_rows, moe_intermediate_size, hidden_itemsize)
    gemm_out_a_bytes = tensor_bytes(chunk_rows, 2 * moe_intermediate_size, hidden_itemsize)
    quant_bytes = tensor_bytes(chunk_rows, moe_intermediate_size, 1)
    scale_bytes = tensor_bytes(chunk_rows, scale_cols, torch.float32.itemsize)
    gemm_out_b_bytes = tensor_bytes(chunk_rows, hidden_size, hidden_itemsize)

    w1_peak_bytes = silu_bytes + gemm_out_a_bytes

    # silu 释放后，B 较小时可复用其 first-fit 空洞；B 更大则需另行预留完整空间。
    quant_w2_peak_bytes = (
        silu_bytes + quant_bytes + scale_bytes + (gemm_out_b_bytes if gemm_out_b_bytes > silu_bytes else 0)
    )
    temporary_peak_bytes = max(w1_peak_bytes, quant_w2_peak_bytes)

    # 按 _get_max_chunk_rows 的 1024 行对齐规则
    gather_bytes = tensor_bytes(align(global_world_size * prefill_tokens_per_rank, 1024), hidden_size, hidden_itemsize)
    per_workspace_bytes = gather_bytes + temporary_peak_bytes

    reuse_buffer_bytes = per_workspace_bytes * microbatch_count
    return align(reuse_buffer_bytes, microbatch_count * 256)


dist_group_manager = DistributeGroupManager()
