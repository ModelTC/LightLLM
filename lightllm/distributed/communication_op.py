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


import math
import os
import torch
import torch.distributed as dist
from dataclasses import dataclass
from torch.distributed import ReduceOp, ProcessGroup
from typing import List, Dict, Optional, Set, Union
from lightllm.utils.log_utils import init_logger
from lightllm.utils.device_utils import has_nvlink
from lightllm.utils.envs_utils import (
    get_env_start_args,
    get_deepep_num_max_dispatch_tokens_per_rank_prefill,
    get_deepep_num_max_dispatch_tokens_per_rank_decode,
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


_TENSOR_BUFFER_ALIGNMENT_BYTES = 256
_DEEPEP_PREFILL_CHUNK_ROWS = 128
_DEEPEP_GATHER_ROWS_CACHE_ALIGNMENT = 1024


@dataclass(frozen=True)
class _LegacyLowLatencyRdmaSizing:
    """Capacity required to reuse each legacy DeepEP RDMA slice in prefill."""

    num_rdma_bytes: int
    per_workspace_bytes: int
    prefill_required_total_bytes: int
    max_dense_rows: int
    microbatch_count: int


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _calculate_legacy_low_latency_rdma_sizing(
    decode_size_hint: int,
    global_world_size: int,
    prefill_tokens_per_rank: int,
    hidden_size: int,
    moe_intermediate_size: Optional[int],
    hidden_dtype: torch.dtype,
    microbatch_count: int,
) -> _LegacyLowLatencyRdmaSizing:
    """Return RDMA storage large enough for one prefill chunk in every slice.

    The grouped FP8 expanded-MoE path keeps the dense gather output alive,
    then executes a 128-row chunk through W1, quantization, and W2. The byte
    accounting mirrors TensorBufferManager's 256-byte first-fit allocation and
    the actual tensor lifetimes in ``chunked_expanded_moe_forward``.
    """
    if moe_intermediate_size is None:
        raise ValueError("Legacy DeepEP low-latency buffer requires moe_intermediate_size")
    if global_world_size <= 0 or prefill_tokens_per_rank <= 0 or hidden_size <= 0:
        raise ValueError("DeepEP workspace dimensions must be positive")
    if moe_intermediate_size <= 0 or moe_intermediate_size % _DEEPEP_PREFILL_CHUNK_ROWS:
        raise ValueError("moe_intermediate_size must be a positive multiple of 128 for legacy DeepEP FP8 MoE")
    if microbatch_count <= 0:
        raise ValueError("DeepEP microbatch_count must be positive")

    def tensor_bytes(*shape: int, itemsize: int) -> int:
        return _align_up(math.prod(shape) * itemsize, _TENSOR_BUFFER_ALIGNMENT_BYTES)

    chunk_rows = _DEEPEP_PREFILL_CHUNK_ROWS
    hidden_itemsize = hidden_dtype.itemsize
    block_size_k = 128
    scale_cols = moe_intermediate_size // block_size_k
    silu_bytes = tensor_bytes(chunk_rows, moe_intermediate_size, itemsize=hidden_itemsize)
    gemm_out_a_bytes = tensor_bytes(chunk_rows, 2 * moe_intermediate_size, itemsize=hidden_itemsize)
    # Legacy expanded MoE quantizes activations to FP8, i.e. one byte per value.
    quant_bytes = tensor_bytes(chunk_rows, moe_intermediate_size, itemsize=1)
    # HAS_SGL_KERNEL uses [scale_cols, chunk_rows], the fallback [chunk_rows,
    # scale_cols]; their element counts are identical.
    scale_bytes = tensor_bytes(chunk_rows, scale_cols, itemsize=torch.float32.itemsize)
    gemm_out_b_bytes = tensor_bytes(chunk_rows, hidden_size, itemsize=hidden_itemsize)

    w1_peak_bytes = silu_bytes + gemm_out_a_bytes
    quant_peak_bytes = silu_bytes + quant_bytes + scale_bytes
    # A is freed before Q/L. S is freed before B; first-fit can reuse S only
    # when B fits there, otherwise B must fit as a contiguous tail allocation.
    if silu_bytes >= gemm_out_b_bytes:
        temp_bytes = max(w1_peak_bytes, quant_peak_bytes)
    else:
        temp_bytes = max(w1_peak_bytes, quant_peak_bytes + gemm_out_b_bytes)

    max_dense_rows = _align_up(
        global_world_size * prefill_tokens_per_rank,
        _DEEPEP_GATHER_ROWS_CACHE_ALIGNMENT,
    )
    gather_bytes = tensor_bytes(max_dense_rows, hidden_size, itemsize=hidden_itemsize)
    per_workspace_bytes = gather_bytes + temp_bytes
    prefill_required_total_bytes = per_workspace_bytes * microbatch_count
    num_rdma_bytes = _align_up(
        max(decode_size_hint, prefill_required_total_bytes),
        microbatch_count * _TENSOR_BUFFER_ALIGNMENT_BYTES,
    )
    return _LegacyLowLatencyRdmaSizing(
        num_rdma_bytes=num_rdma_bytes,
        per_workspace_bytes=per_workspace_bytes,
        prefill_required_total_bytes=prefill_required_total_bytes,
        max_dense_rows=max_dense_rows,
        microbatch_count=microbatch_count,
    )


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

    def all_reduce_residual_rmsnorm(self, inp, residual, rms_weight, eps, alloc_func):
        # Fused AR + residual-add + RMSNorm via flashinfer when the message is small enough
        # for the oneshot-lamport fast path; otherwise return None so the caller falls back to
        # a plain all_reduce + a separate (fused_add) rmsnorm.
        if self.flashinfer_reduce is not None and self.flashinfer_reduce.should_use(inp):
            return self.flashinfer_reduce.allreduce_residual_rmsnorm(inp, residual, rms_weight, eps, alloc_func)
        return None

    def all_gather_into_tensor(self, output_: torch.Tensor, input_: torch.Tensor, async_op: bool = False) -> None:
        return dist.all_gather_into_tensor(output_, input_, group=self.device_group, async_op=async_op)


class DistributeGroupManager:
    def __init__(self):
        self.groups = []
        self.ep_balance_monitor_group = None
        self.ep_buffer = None
        self.ep_low_latency_buffer = None
        self.ep_prefill_moe_workspace = None
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
        if (
            getattr(args, "enable_ep_moe", False)
            and not getattr(args, "disable_ep_balance_monitor", False)
            and getattr(args, "run_mode", "normal") != "decode"
            and not getattr(args, "enable_prefill_cudagraph", False)
            and not is_sm100_gpu()
        ):
            self.ep_balance_monitor_group = dist.new_group(ranks=list(range(get_global_world_size())), backend="gloo")
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
            if experts is None:
                experts = getattr(layer_weight, "experts_", None)
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
        args = get_env_start_args()
        enable_ep_moe = args.enable_ep_moe
        prefill_num_max_dispatch_tokens_per_rank = get_deepep_num_max_dispatch_tokens_per_rank_prefill()
        decode_num_max_dispatch_tokens_per_rank = (
            None if args.run_mode == "prefill" else get_deepep_num_max_dispatch_tokens_per_rank_decode()
        )
        if not enable_ep_moe:
            self.ep_buffer = None
            self.ep_low_latency_buffer = None
            self.ep_prefill_moe_workspace = None
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
        total_redundant_experts = (
            get_env_start_args().eplb_num_redundant_experts_per_rank * global_world_size
            if get_env_start_args().enable_prefill_eplb
            else 0
        )
        self.ll_prefill_num_experts = n_routed_experts + total_redundant_experts
        # EPLB's redundant rows are a prefill-only physical layout; decode
        # always routes the logical expert space.
        self.ll_decode_num_experts = n_routed_experts
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
        self.ep_prefill_moe_workspace = None

        if not expert_quant_method_names:
            raise ValueError("No valid MoE quant method was found while initializing DeepEP buffers")

        mega_moe_quant_method = "fp4fp8-b32-deepgemm"
        is_sm100 = is_sm100_gpu()

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
        else:
            enable_mega_moe_buffer = False
            has_legacy_moe_layer = True

        # The legacy buffer is also the only prefill workspace. Pure prefill
        # does not use its low-latency protocol, but still needs its RDMA
        # storage for the bounded grouped-MoE compute path.
        enable_low_latency_buffer = has_legacy_moe_layer

        if enable_low_latency_buffer:
            if self.ll_decode_num_tokens is None:
                decode_size_hint = 0
            else:
                decode_size_hint = deep_ep.Buffer.get_low_latency_rdma_size_hint(
                    self.ll_decode_num_tokens, self.ll_hidden, global_world_size, self.ll_decode_num_experts
                )
            rdma_sizing = _calculate_legacy_low_latency_rdma_sizing(
                decode_size_hint=decode_size_hint,
                global_world_size=global_world_size,
                prefill_tokens_per_rank=prefill_num_max_dispatch_tokens_per_rank,
                hidden_size=hidden_size,
                moe_intermediate_size=moe_intermediate_size,
                hidden_dtype=get_torch_dtype(args.data_type),
                microbatch_count=len(self.groups),
            )
            num_rdma_bytes = rdma_sizing.num_rdma_bytes
            logger.info(
                "Initialize DeepEP legacy RDMA workspace: decode_size_hint=%s, "
                "prefill_required_total_bytes=%s, num_rdma_bytes=%s, "
                "per_workspace_bytes=%s, max_dense_rows=%s, microbatch_count=%s",
                decode_size_hint,
                rdma_sizing.prefill_required_total_bytes,
                num_rdma_bytes,
                rdma_sizing.per_workspace_bytes,
                rdma_sizing.max_dense_rows,
                rdma_sizing.microbatch_count,
            )
            self.ep_low_latency_buffer = deep_ep.Buffer(
                deepep_group,
                num_rdma_bytes=num_rdma_bytes,
                low_latency_mode=True,
                num_qps_per_rank=(self.ll_decode_num_experts // global_world_size),
            )
            self.ep_prefill_moe_workspace = self.ep_low_latency_buffer.get_local_buffer_tensor(
                torch.uint8, use_rdma_buffer=True
            )

        if enable_mega_moe_buffer:
            # SM100 FP4 层通过 DeepGEMM Mega MoE 完成通信和计算，不使用 legacy
            # low-latency buffer，因此纯 FP4 模型无需承担后者的大块 RDMA 显存。
            if moe_intermediate_size is None:
                raise ValueError("SM100 Mega MoE requires moe_intermediate_size or intermediate_size in model config")

            import deep_gemm

            self.ep_mega_moe_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
                deepep_group,
                self.ll_decode_num_experts,
                self.ll_num_tokens,
                num_experts_per_tok,
                self.ll_hidden,
                moe_intermediate_size,
            )
        logger.info(
            "Initialize DeepEP MoE buffers: low_latency=%s, prefill_workspace_bytes=%s, "
            "mega_moe=%s, ll_prefill_num_experts=%s, ll_decode_num_experts=%s, expert_quant_method_names=%s",
            enable_low_latency_buffer,
            self.ep_prefill_moe_workspace.numel() if self.ep_prefill_moe_workspace is not None else 0,
            enable_mega_moe_buffer,
            self.ll_prefill_num_experts,
            self.ll_decode_num_experts,
            sorted(expert_quant_method_names),
        )
        theoretical_sms = self.ep_buffer.get_theoretical_num_sms(self.ll_prefill_num_experts, num_experts_per_tok)
        low_latency_sms = self.ep_buffer.get_theoretical_num_sms(self.ll_decode_num_experts, num_experts_per_tok)
        self._set_num_sms_for_deep_gemm(theoretical_sms, low_latency_sms)

    def _set_num_sms_for_deep_gemm(self, deepep_sms: int, low_latency_sms: int):
        try:
            try:
                from deep_gemm.jit_kernels.utils import set_num_sms
            except:
                from deep_gemm import set_num_sms

            device_sms = get_device_sm_count()
            deepep_sms = max(0, min(deepep_sms, max(device_sms - 2, 0)))
            low_latency_sms = max(0, min(low_latency_sms, max(device_sms - 2, 0)))
            self.ep_num_sms = deepep_sms
            if self.ep_low_latency_buffer is not None:
                # This setting controls the legacy low-latency buffer; keep
                # its SM reservation based on decode's logical expert count.
                deep_ep.Buffer.set_num_sms(low_latency_sms - low_latency_sms % 2)
            set_num_sms(max(device_sms - deepep_sms, 2))
        except BaseException as e:
            logger.warning(f"set num sms for deep_gemm failed: {e}")

    def get_deep_ep_prefill_moe_workspace(self, microbatch_index: int = 0) -> torch.Tensor:
        """Return a slice of the workspace used by DeepEP Prefill MoE kernels.

        All legacy FP8 MoE modes, including pure prefill, use local DeepEP
        low-latency RDMA storage. Pure prefill only treats it as workspace;
        decode-capable modes reuse the same storage after decode has released
        its low-latency layout. With one communication group, the default
        ``microbatch_index=0`` receives the whole workspace. With multiple
        groups, the workspace is split into ``len(self.groups)`` equal slices.

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
        assert self.ep_prefill_moe_workspace is not None, "DeepEP Prefill MoE workspace is not initialized"
        workspace = self.ep_prefill_moe_workspace
        microbatch_count = len(self.groups)
        assert 0 <= microbatch_index < microbatch_count
        workspace_size = workspace.numel() // microbatch_count
        return workspace.narrow(0, microbatch_index * workspace_size, workspace_size)

    def clear_deepep_buffer(self):
        """
        Decode-capable modes reuse the low-latency RDMA buffer during Prefill,
        so clean it before the next low-latency Decode. Pure Prefill uses the
        same RDMA storage only as workspace and has no low-latency layout to
        clean.
        """
        if self.ep_low_latency_buffer is not None and self.ll_decode_num_tokens is not None:
            self.ep_low_latency_buffer.clean_low_latency_buffer(
                self.ll_decode_num_tokens, self.ll_hidden, self.ll_decode_num_experts
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


def all_reduce_residual_rmsnorm(
    inp: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    eps: float,
    group: Optional[Union[ProcessGroup, CustomProcessGroup]],
    alloc_func,
):
    """Fused all-reduce + residual-add + RMSNorm (SGLang #22390).

    Returns ``(norm_out, residual_out)`` when a fused fast path (flashinfer) is available,
    otherwise ``None`` so the caller can fall back to ``all_reduce`` + a separate
    (fused-add) RMSNorm. ``inp`` is the un-reduced tensor; ``residual`` is added after the
    reduction.
    """
    if isinstance(group, CustomProcessGroup):
        return group.all_reduce_residual_rmsnorm(inp, residual, rms_weight, eps, alloc_func)
    return None


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
