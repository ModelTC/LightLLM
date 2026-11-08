"""KDA replay 运行态。

这个文件单独实现 KDA 的 replay history、checkpoint 和 decode 准备。
它不继承 native ``GDNState``，也不和 GDN replay 共享执行器基类。
"""

from __future__ import annotations

import torch
import triton

from lightllm.common.ssm_types import SSMRunConfig
from lightllm.common.basemodel.triton_kernel.linear_att.replayssm import (
    _accept_updates,
    _prepare_fold,
    _fold_history,
    _materialize,
    _replay,
)
from lightllm.common.basemodel.triton_kernel.linear_att.ssm_autotune import resolve_run_config
from collections.abc import Sequence

from lightllm.common.basemodel.triton_kernel.linear_att.ssm_acceptance import commit_ssm_prefix


def upload_indices(values: Sequence[int], device, dtype=torch.int64) -> torch.Tensor:
    """把 Python 索引上传到目标 device。"""

    return torch.tensor(list(values), dtype=dtype, pin_memory=True).to(device, non_blocking=True)


class ReplayKDARuntime:
    """以短 history 重放 decode 的 SSM 执行器。

    ``state`` 是唯一的 canonical checkpoint；keys、deltas、gates、betas 和
    cursor 只属于请求当前的临时 history，不参与 CPU/PD checkpoint。decode
    先从 checkpoint 加载 history，再把新 token 写入 ring buffer；接受或
    checkpoint 导出时，materialize kernel 将已接受前缀折叠回 ``state``。
    """

    _run_config: SSMRunConfig | None = None

    @property
    def run_config(self) -> SSMRunConfig | None:
        """返回最近一次 forward 使用的 kernel 配置。"""

        return self._run_config

    @run_config.setter
    def run_config(self, config: SSMRunConfig | dict | None) -> None:
        """保存 kernel 配置，兼容 autotune 返回的字典。"""

        if config is None or isinstance(config, SSMRunConfig):
            self._run_config = config
        else:
            self._run_config = SSMRunConfig(**config)

    def __init__(
        self,
        state: torch.Tensor,
        capacity: int,
        verify_width: int,
        activation_dtype: torch.dtype = torch.bfloat16,
        *,
        num_key_heads: int | None = None,
        kda: bool = True,
        gate_lower_bound: float = -5.0,
        compute_mode: str = "inline",
        run_config: SSMRunConfig | dict[str, int] | None = None,
    ) -> None:
        """创建 ReplaySSM 的 canonical state、双半区 history 和 cursor。

        ``verify_width`` 决定一次 verify 的最大 token 数，``capacity`` 是
        history ring 的半区长度；``compute_mode`` 选择 projection 的执行方式。
        """

        assert state.dtype in (torch.float32, torch.bfloat16), "ReplaySSM requires FP32 or BF16 SSM state"
        assert verify_width > 0 and capacity >= max(4, verify_width) and capacity & (capacity - 1) == 0
        assert compute_mode in ("inline", "precompute")
        self.state = state
        self.kda = kda
        self.gate_lower_bound = gate_lower_bound
        self.fold_programs = (
            8 * torch.cuda.get_device_properties(state.device).multi_processor_count if state.is_cuda else 0
        )
        self.capacity = capacity
        self.verify_width = verify_width
        self.compute_mode = compute_mode
        self.run_config = run_config
        self._config_is_fixed = run_config is not None
        self.batch_configs = {}
        layers, slots, hv, k, v = state.shape
        # 保留冗余布局，兼容已有调用方和 A/B 测试。
        self.num_key_heads = hv if num_key_heads is None else num_key_heads
        assert self.num_key_heads > 0 and hv % self.num_key_heads == 0
        self.hold_req_id = slots - 1
        self.cursors = torch.zeros(slots, dtype=torch.int32, device=state.device)
        self.forward_cursors = torch.zeros_like(self.cursors)
        # fold 时交替使用两个半区，避免一个 V tile 仍在读取时被其他 CTA 覆盖。
        # cursor 打包保存 phase（2*L）和 count（0..L）。precompute 路径在 BF16
        # 下可以使用更窄的 history buffer；其他形状使用 FP32 作为安全默认值。
        history_dtype = (
            activation_dtype
            if activation_dtype == torch.bfloat16
            and verify_width > 1
            and compute_mode == "precompute"
            and capacity <= 8
            else torch.float32
        )
        key_shape = (layers, slots, self.num_key_heads, 2 * capacity, k)
        self.keys = torch.empty(key_shape, dtype=history_dtype, device=state.device)
        self.deltas = torch.empty((layers, slots, hv, 2 * capacity, v), dtype=history_dtype, device=state.device)
        self.gates = torch.empty(
            (layers, slots, hv, 2 * capacity, k) if kda else (layers, slots, hv, 2 * capacity),
            dtype=torch.float32,
            device=state.device,
        )
        self.raw_keys = torch.empty(key_shape, dtype=activation_dtype, device=state.device)
        self.raw_values = torch.empty((layers, slots, hv, 2 * capacity, v), dtype=activation_dtype, device=state.device)
        self.betas = torch.empty((layers, slots, hv, 2 * capacity), dtype=torch.float32, device=state.device)

    def clear_history(self, req: int | torch.Tensor) -> None:
        """清空复用或恢复请求的临时 history，保留 canonical SSM state。"""
        if type(req) is int:
            # 标量赋值会把 CPU tensor 复制到 CUDA slot。
            self.cursors[req].zero_()
        else:
            self.cursors[req] = 0

    def prepare_decode(self, reqs: torch.Tensor, cu_seqlens: torch.Tensor | None = None) -> torch.Tensor:
        """保存本轮 decode 的 history 起点，并折叠即将溢出的旧半区。"""
        positions = torch.empty_like(reqs)
        layers, _, hv, kd, vd = self.state.shape
        active = torch.empty((reqs.numel() + 1,), dtype=torch.int32, device=reqs.device)
        _prepare_fold[(1,)](
            reqs,
            self.cursors,
            positions,
            self.forward_cursors,
            cu_seqlens,
            active,
            reqs.numel(),
            self.hold_req_id,
            self.capacity,
            cu_seqlens is not None,
            triton.next_power_of_2(reqs.numel()),
            self.verify_width,
            num_warps=4,
        )
        programs = min(self.fold_programs, triton.cdiv(vd, 32) * reqs.numel() * layers * hv)
        _fold_history[(programs,)](
            self.state,
            self.raw_keys,
            self.raw_values,
            self.gates,
            self.betas,
            reqs,
            positions,
            active,
            reqs.numel(),
            layers,
            hv,
            self.num_key_heads,
            kd,
            vd,
            self.capacity,
            triton.next_power_of_2(kd),
            32,
            self.state.stride(0),
            self.raw_keys.stride(0),
            self.raw_values.stride(0),
            self.gates.stride(0),
            self.betas.stride(0),
            self.kda,
            num_warps=2,
            num_stages=1,
        )
        return positions

    def accept_updates(self, reqs: torch.Tensor, accepted_state_indices: torch.Tensor | None = None) -> None:
        """更新 cursor 以保留每个请求本轮已接受的前缀。

        MTP 时 ``accepted_state_indices`` 是每个请求最后接受的 draft
        位置；单 token 路径传 ``None``，每个请求只推进一个位置。
        """
        _accept_updates[(triton.cdiv(reqs.numel(), 256),)](
            self.cursors,
            reqs,
            accepted_state_indices,
            reqs.numel(),
            self.hold_req_id,
            self.verify_width,
            256,
        )

    def merge_accepted_updates(self, reqs: torch.Tensor) -> None:
        """把接受的 history 物化回 canonical state，并清空请求 cursor。"""
        if reqs.numel() == 0:
            return
        self._materialize_accepted_state(reqs, self.state, snapshot=False)
        self.cursors.index_fill_(0, reqs.long(), 0)

    def _materialize_accepted_state(
        self,
        reqs: torch.Tensor,
        output: torch.Tensor,
        snapshot: bool,
        output_strides=None,
        flush_export: bool = False,
    ) -> None:
        """将请求的 accepted prefix 写入 state 或调用方提供的输出页。"""

        assert not (snapshot and flush_export), "flush export must update the canonical state"

        layers, slots, hv, k, v = self.state.shape
        config = self.run_config or SSMRunConfig(32, 4)
        bv = config.BV
        _materialize[(triton.cdiv(v, bv), reqs.numel(), layers * hv)](
            self.state,
            output,
            self.raw_keys,
            self.raw_values,
            self.gates,
            self.betas,
            self.cursors,
            reqs,
            slots,
            hv,
            self.num_key_heads,
            k,
            v,
            self.capacity,
            self.hold_req_id,
            triton.next_power_of_2(k),
            bv,
            snapshot,
            self.kda,
            FLUSH_EXPORT=flush_export,
            OUTPUT_STRIDES=output_strides,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )

    def get_accepted_state(self, req_idx: int, accepted_index: int | None = None) -> torch.Tensor:
        """返回不修改 active state/history 的单请求 canonical 快照。"""
        layers, _, hv, k, v = self.state.shape
        output = torch.empty((layers, hv, k, v), dtype=self.state.dtype, device=self.state.device)
        reqs = torch.tensor([req_idx], dtype=torch.int32, device=self.state.device)
        self._materialize_accepted_state(reqs, output, snapshot=True)
        return output

    def export_accepted_state(
        self,
        req_indices: list[int],
        targets: list[torch.Tensor],
        accepted_indices: torch.Tensor | None = None,
        *,
        flush: bool = False,
    ) -> None:
        """将多个 accepted 快照直接写入调用方目标页，避免中间整批分配。"""

        if not req_indices:
            return
        shape = (self.state.shape[0], *self.state.shape[2:])
        strides = targets[0].stride()
        assert len(req_indices) == len(targets)
        assert all(t.shape == shape and t.stride() == strides and t.dtype == self.state.dtype for t in targets)
        self._materialize_accepted_state(
            upload_indices(req_indices, self.state.device, torch.int32),
            upload_indices([t.data_ptr() for t in targets], self.state.device),
            snapshot=not flush,
            flush_export=flush,
            output_strides=strides,
        )

    def get_run_config(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        a_log: torch.Tensor,
        bias: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
    ) -> SSMRunConfig:
        """按当前 kernel 形状选择或返回固定的 ReplaySSM 调优配置。"""

        return resolve_run_config(self, "replay", q, k, v, a, b, a_log, bias, cu_seqlens)

    def forward(
        self,
        linear_layer_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        a_log: torch.Tensor,
        bias: torch.Tensor,
        reqs: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        run_config: SSMRunConfig | None = None,
        *,
        state_indices: torch.Tensor | None = None,
        accepted_token_counts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, SSMRunConfig]:
        """执行一层 ReplaySSM decode，并返回输出与本轮 run config。"""

        config = run_config or self.get_run_config(q, k, v, a, b, a_log, bias, cu_seqlens)
        hv, kd, vd = self.state.shape[-3:]
        assert self.num_key_heads in (q.shape[-2], hv)
        axis = 1 if cu_seqlens is not None else 0
        out = torch.empty_like(v)
        bv = config.BV
        precompute_projections = self.verify_width > 1 and bv >= 16 and self.compute_mode == "precompute"
        # 在 Python 中解析 tile 维度；Triton JIT 中直接调用 max() 会降级为 tensor
        # 运算，无法定义静态 shape。
        bw = triton.next_power_of_2(self.verify_width)
        _replay[(triton.cdiv(vd, bv), reqs.numel(), hv)](
            q,
            k,
            v,
            a,
            b,
            a_log,
            bias,
            self.state[linear_layer_idx],
            self.keys[linear_layer_idx],
            self.deltas[linear_layer_idx],
            self.gates[linear_layer_idx],
            self.raw_keys[linear_layer_idx],
            self.raw_values[linear_layer_idx],
            self.betas[linear_layer_idx],
            reqs,
            self.forward_cursors,
            cu_seqlens,
            out,
            q.stride(axis),
            k.stride(axis),
            v.stride(axis),
            a.stride(0),
            b.stride(0),
            q.shape[-2],
            hv,
            self.num_key_heads,
            kd,
            vd,
            self.capacity,
            self.hold_req_id,
            triton.next_power_of_2(kd),
            bv,
            cu_seqlens is not None,
            precompute_projections,
            bw,
            self.kda,
            self.gate_lower_bound,
            BQK=max(16, 2 * bw),
            BR=triton.next_power_of_2(max(16, self.capacity + bw)),
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )
        return out, config

    def accept_single_token(self, reqs: torch.Tensor) -> None:
        """非 MTP decode 直接把单 token 写入当前 history。"""
        self.accept_updates(reqs)

    def commit_accepted(self, indices: torch.Tensor, reqs: torch.Tensor, lengths: torch.Tensor) -> None:
        """把接受长度写入 MTP 槽位索引，并推进本请求的 replay cursor。"""

        if self.verify_width > 1:
            commit_ssm_prefix(indices, reqs, lengths, self.cursors)
        else:
            commit_ssm_prefix(indices, reqs, lengths)
            self.accept_updates(reqs, indices)

    def commit(self, acceptance, indices: torch.Tensor, state_name: str | None = None) -> None:
        """兼容直接持有 runtime 的测试：从 acceptance 取出请求和接受长度。"""

        self.commit_accepted(indices, acceptance.req_indices, acceptance.accepted_lengths)

    def export_ssm(self, req_indices: list[int], targets: list[torch.Tensor], *, flush: bool = False) -> None:
        """把已接受 SSM 写入调用方页面。CUDA 目标直接写指针，其他设备先在 GPU 上物化再拷贝。"""

        if not req_indices:
            return
        target = targets[0]
        if target.is_cuda and target.device == self.state.device:
            self.export_accepted_state(req_indices, targets, flush=flush)
            return
        for req, dst in zip(req_indices, targets):
            snap = torch.empty(
                (self.state.shape[0], *self.state.shape[2:]),
                dtype=self.state.dtype,
                device=self.state.device,
            )
            self._materialize_accepted_state(
                torch.tensor([req], dtype=torch.int32, device=self.state.device),
                snap,
                snapshot=True,
            )
            dst.copy_(snap)
            if flush:
                self.state[:, req].copy_(snap)
                self.clear_history(req)


from functools import lru_cache
from typing import TYPE_CHECKING, List
from math import prod

import torch

from lightllm.utils.log_utils import init_logger

from ...utils import deepcopy_result, dtype_name
from ...types import AxisMapping, StateTransferLayout, TensorField, TransferRegion
from ..base_state import BoundedAttentionState
from ..conv_ssm_types import ConvSSMStateConfig
from .kda_pages_pool import KDAPagesPool
from ..transfer_utils import bounded_att_buffer_copies
from ..copy_kernel import copy_bounded_att_tensors

if TYPE_CHECKING:
    from lightllm.common.attention_state.types import PageKind


logger = init_logger(__name__)


class ReplayKDAState(BoundedAttentionState):
    """KDA replay：每个请求一份 canonical SSM，外加一段不进入 checkpoint 的 history。"""

    big_pages_pool: KDAPagesPool | None
    small_pages_pool: KDAPagesPool | None

    # ========================================================================
    # 一、资源定义与初始化
    # ========================================================================

    def __init__(
        self,
        config: ConvSSMStateConfig,
        request_capacity: int,
        mtp_step: int = 0,
        device: str | torch.device = "cuda",
        *,
        ssm_state_mode: str = "replay",
        ssm_replay_config: dict | None = None,
        gate_lower_bound: float = -5.0,
    ) -> None:
        """按请求容量创建 Conv 运行态、单槽 canonical SSM 和 replay history。"""
        if ssm_state_mode != "replay":
            raise ValueError(f"ReplayKDAState only implements replay mode, got {ssm_state_mode}")
        assert isinstance(config, ConvSSMStateConfig), "config must be ConvSSMStateConfig"
        self.config = config
        self.state_name: str | None = None  # 由 AttentionStateManager.register_state() 绑定时设置。
        self.request_capacity = request_capacity  # 包含预留的 HOLD 请求槽位。
        self.mtp_step = mtp_step
        self.gate_lower_bound = gate_lower_bound
        self._logical_to_storage_layer = {
            logical_layer_id: storage_layer_index
            for storage_layer_index, logical_layer_id in enumerate(config.logical_layer_ids)
        }
        storage_layer_count = len(config.logical_layer_ids)
        self.conv_state = torch.zeros(
            (storage_layer_count, request_capacity, config.conv_channels, config.conv_width + mtp_step),
            dtype=config.conv_dtype,
            device=device,
        )
        self.ssm_state = torch.zeros(
            (
                storage_layer_count,
                request_capacity,
                config.num_v_heads,
                config.key_dim,
                config.value_dim,
            ),
            dtype=config.ssm_dtype,
            device=device,
        )
        capacity, compute_mode = _replay_settings(ssm_replay_config, mtp_step)
        self.runtime = ReplayKDARuntime(
            self.ssm_state,
            capacity,
            mtp_step + 1,
            config.conv_dtype,
            num_key_heads=config.num_k_heads,
            kda=True,
            gate_lower_bound=gate_lower_bound,
            compute_mode=compute_mode,
        )
        self.mtp_state_index = torch.zeros(request_capacity, dtype=torch.int32, device=device) if mtp_step else None
        self.big_pages_pool = None
        self.small_pages_pool = None

    @property
    def ssm_slots_per_req(self) -> int:
        """Replay 每个请求只保留一个 canonical SSM 槽位。"""

        return 1

    @property
    def _ssm_executor(self):
        """供既有测试读取 history；实现本身不依赖这个名字。"""

        return self.runtime

    def get_state_type(self) -> str:
        """返回 GDN 算法及 Conv/SSM 存储精度。"""
        return f"kda_conv_{dtype_name(dtype=self.config.conv_dtype)}" f"_ssm_{dtype_name(dtype=self.config.ssm_dtype)}"

    def initialize_page_pools(
        self,
        big_page_capacity: int,
        small_page_capacity: int = 0,
        device: str | torch.device | int = "cpu",
        pin_memory: bool = True,
    ) -> None:
        """容量规划完成后创建本算法的大小页池；容量为零的页池保持 None。"""
        assert self.big_pages_pool is None and self.small_pages_pool is None, "page pools already initialized"
        assert big_page_capacity >= 0 and small_page_capacity >= 0, "page capacities must be nonnegative"
        self.big_pages_pool = (
            KDAPagesPool(
                size=big_page_capacity,
                config=self.config,
                device=device,
                pin_memory=pin_memory,
            )
            if big_page_capacity
            else None
        )
        self.small_pages_pool = (
            KDAPagesPool(
                size=small_page_capacity,
                config=self.config,
                device=device,
                pin_memory=pin_memory,
            )
            if small_page_capacity
            else None
        )

    # ========================================================================
    # 二、Bounded attention：计算视图、运行态重置、大小页与 MTP 提交
    # ========================================================================

    def contains_logical_layer(self, logical_layer_id: int) -> bool:
        """判断指定逻辑层是否属于当前 State 的层映射，共享层同样返回 True。"""
        return logical_layer_id in self._logical_to_storage_layer

    def get_layer_view(self, logical_layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """按逻辑层号映射到物理层，返回 Conv/SSM Tensor 引用；MTP 游标和步数直接从 State 读取。"""
        storage_layer_index = self._logical_to_storage_layer[logical_layer_id]
        return self.conv_state[storage_layer_index], self.ssm_state[storage_layer_index]

    def reset_bounded_att_state(self, req_indices: List[int]) -> None:
        """清零指定请求的 Conv 历史、全部 SSM 候选及 MTP 游标，保留页池快照。"""
        assert isinstance(req_indices, list), "request indices must be a list"
        for req_idx in req_indices:
            self.conv_state[:, req_idx].zero_()
            self.ssm_state[:, req_idx].zero_()
            self.runtime.clear_history(req_idx)
            if self.mtp_state_index is not None:
                self.mtp_state_index[req_idx].fill_(0)

    def save_bounded_att_runtime_to_pages(
        self,
        req_indices: List[int],
        state_token_lens: List[int],
        page_kind: PageKind,
        page_slot_ids: List[int],
    ) -> None:
        """保存各请求指定绝对 token 长度处的基准状态；Conv/SSM 不依赖长度定位。"""
        assert isinstance(req_indices, list) and isinstance(
            page_slot_ids, list
        ), "request and slot indices must be lists"
        assert len(req_indices) == len(state_token_lens) == len(page_slot_ids), "request, length and page counts differ"
        pool = self._pool(page_kind)
        conv_copies = [
            (self._accepted_conv(req_idx), pool.conv_state[slot_id])
            for req_idx, slot_id in zip(req_indices, page_slot_ids)
        ]
        copy_bounded_att_tensors(copies=conv_copies)
        self.runtime.export_ssm(req_indices, [pool.ssm_state[slot_id] for slot_id in page_slot_ids])

    def restore_bounded_att_runtime_from_pages(
        self,
        req_indices: List[int],
        state_token_lens: List[int],
        page_kind: PageKind,
        page_slot_ids: List[int],
    ) -> None:
        """恢复各请求指定绝对 token 长度处的基准状态并归零游标；Conv/SSM 不依赖长度定位。"""
        assert isinstance(req_indices, list) and isinstance(
            page_slot_ids, list
        ), "request and slot indices must be lists"
        assert len(req_indices) == len(state_token_lens) == len(page_slot_ids), "request, length and page counts differ"
        pool = self._pool(page_kind)
        width = self.config.conv_width
        conv_copies, ssm_copies = [], []
        for req_idx, slot_id in zip(req_indices, page_slot_ids):
            self.runtime.clear_history(req_idx)
            conv_copies.append((pool.conv_state[slot_id], self.conv_state[:, req_idx, :, :width]))
            ssm_copies.append((pool.ssm_state[slot_id], self.ssm_state[:, req_idx]))
        copy_bounded_att_tensors(copies=conv_copies)
        copy_bounded_att_tensors(copies=ssm_copies)
        if self.mtp_state_index is not None:
            for req_idx in req_indices:
                self.mtp_state_index[req_idx].fill_(0)

    def commit_bounded_att_state(
        self,
        req_indices: torch.Tensor,
        accepted_steps: torch.Tensor,
    ) -> None:
        """提交已接受前缀：更新 MTP 游标，并把同样的长度加到 replay cursor。"""
        if self.mtp_state_index is None:
            return
        self.runtime.commit_accepted(self.mtp_state_index, req_indices, accepted_steps)

    def merge_accepted_updates(self, reqs: torch.Tensor) -> None:
        """Prefill 前把已接受 history 折回 canonical state。"""

        self.runtime.merge_accepted_updates(reqs)

    def prepare_for_decode(self, reqs: torch.Tensor, cu_seqlens: torch.Tensor | None) -> None:
        """折叠即将溢出的 history。非 MTP decode 同时接受本轮唯一 token。"""

        self.runtime.prepare_decode(reqs, cu_seqlens)
        if cu_seqlens is None:
            self.runtime.accept_single_token(reqs)

    def replay_forward(
        self,
        logical_layer: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        a_log: torch.Tensor,
        bias: torch.Tensor,
        reqs: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """执行一层 KDA replay decode。"""

        output, _ = self.runtime.forward(
            self._logical_to_storage_layer[logical_layer],
            q,
            k,
            v,
            a,
            b,
            a_log,
            bias,
            reqs,
            cu_seqlens,
        )
        return output

    def get_accepted_state(self, req_idx: int):
        """返回已接受 Conv 窗口和物化后的 SSM，不改写 canonical state 或 cursor。"""

        ssm = self.runtime.get_accepted_state(req_idx)
        return _AcceptedTensors(self._accepted_conv(req_idx), ssm)

    def _accepted_conv(self, req_idx: int) -> torch.Tensor:
        width = self.config.conv_width
        conv = self.conv_state[:, req_idx]
        if self.mtp_state_index is None:
            return conv[:, :, :width]
        offset = self.mtp_state_index[req_idx]
        index = torch.arange(width, device=conv.device) + offset
        return conv.index_select(-1, index)

    # ========================================================================
    # 三、CPU Cache 数据交换
    # ========================================================================

    @deepcopy_result
    @lru_cache(maxsize=None)
    def get_cpu_transfer_regions(self) -> tuple[TransferRegion, ...]:
        """返回已接受 Conv/SSM 快照的字段区间与切片，供 CPU 页布局和容量计算使用。

        region.shape 描述 CPU 页中的全局字段；axis_mappings 将本地页池的 TP 分片映射到该字段。
        起止位置相对本 State，由管理器平移到 CPU 联合页；不包含 MTP 候选区或游标。
        """
        return self._get_transfer_regions()

    def write_state_pages_to_cpu(
        self,
        page_kind: PageKind,
        page_slot_ids: List[int],
        cpu_page_buffers: torch.Tensor,
        cpu_page_indices: List[int],
        layout: StateTransferLayout,
        cpu_page_ready_flags: list[bool] | None = None,
    ) -> None:
        """将指定页池槽位的快照写入 CPU 页。"""
        assert layout.state_name == self.state_name, "transfer layout does not describe this resource"
        assert 0 <= layout.byte_start < layout.byte_end, "invalid State byte interval"
        assert cpu_page_buffers.device.type == "cpu", "CPU cache buffer must reside on CPU"
        pool = self._pool(page_kind)
        assert isinstance(page_slot_ids, list), "slot indices must be a list"
        assert isinstance(cpu_page_indices, list), "CPU page indices must be a list"
        assert len(page_slot_ids) == len(cpu_page_indices), "CPU page and slot counts differ"
        if cpu_page_ready_flags is not None:
            assert len(cpu_page_ready_flags) == len(cpu_page_indices), "CPU page and ready counts differ"
        page_fields, page_indices = [], []
        for index, (slot, page) in enumerate(zip(page_slot_ids, cpu_page_indices)):
            # ready 页不参与复制；同一传输区域的所有页面组成一个同形状批次。
            if cpu_page_ready_flags is not None and cpu_page_ready_flags[index]:
                continue
            page_fields.append(self._fields(pool.conv_state[slot], pool.ssm_state[slot]))
            page_indices.append(page)
        for copies in bounded_att_buffer_copies(
            page_fields=page_fields,
            page_buffers=cpu_page_buffers,
            layout=layout,
            page_indices=page_indices,
            write=True,
        ):
            copy_bounded_att_tensors(copies=copies)

    def read_state_pages_from_cpu(
        self,
        cpu_page_buffers: torch.Tensor,
        cpu_page_indices: List[int],
        big_page_slot_ids: List[int],
        layout: StateTransferLayout,
    ) -> None:
        """读入指定大页槽位；请求运行态另行通过 restore_bounded_att_runtime_from_pages 恢复。"""
        assert layout.state_name == self.state_name, "transfer layout does not describe this resource"
        assert 0 <= layout.byte_start < layout.byte_end, "invalid State byte interval"
        assert cpu_page_buffers.device.type == "cpu", "CPU cache buffer must reside on CPU"
        pool = self.big_pages_pool
        assert isinstance(big_page_slot_ids, list), "slot indices must be a list"
        assert isinstance(cpu_page_indices, list), "CPU page indices must be a list"
        assert len(cpu_page_indices) == len(big_page_slot_ids), "CPU page and slot counts differ"
        # 本地目标：选定大页槽位的 Conv/SSM 快照，保留本 TP rank 的分片形状。
        pool_page_fields = [self._fields(pool.conv_state[slot], pool.ssm_state[slot]) for slot in big_page_slot_ids]
        # 外部来源：layout.regions 描述 CPU 全局字段和 TP 范围；write=False 将其读入本地页池。
        # 窗口保持快照顺序；请求 ring 的位置由后续 restore_bounded_att_runtime_from_pages 按长度恢复。
        for copies in bounded_att_buffer_copies(
            page_fields=pool_page_fields,
            page_buffers=cpu_page_buffers,
            layout=layout,
            page_indices=cpu_page_indices,
            write=False,
        ):
            copy_bounded_att_tensors(copies=copies)

    # ========================================================================
    # 四、DP 迁移：大页快照与基准运行态
    # ========================================================================

    def copy_big_pages_from_dp(
        self,
        src_page_ids: List[int],
        dst_page_ids: List[int],
        remote_dp_bounded_state: ReplayKDAState,
    ) -> None:
        """复制同一 TP 分片的大页 Conv/SSM 快照，保留源页和双方请求运行态。"""
        assert isinstance(src_page_ids, list) and isinstance(dst_page_ids, list), "page indices must be lists"
        assert len(src_page_ids) == len(dst_page_ids), "source and destination page counts differ"
        assert (
            type(remote_dp_bounded_state) is type(self) and remote_dp_bounded_state.config == self.config
        ), "DP copy requires the same algorithm and equivalent TP state configurations"
        src_pool = remote_dp_bounded_state.big_pages_pool
        dst_pool = self.big_pages_pool
        assert src_pool is not None and dst_pool is not None, "DP big page copy requires both page pools"
        # 大页保存的是基准快照，与请求槽位及 MTP 候选区无关；页的引用和使用期限由调用方管理。
        # 页池均须在本进程注册；GPU 在当前流直接访问两端 CPU 存储，完成后才能归还页面。
        conv_copies, ssm_copies = [], []
        for src_page, dst_page in zip(src_page_ids, dst_page_ids):
            conv_copies.append((src_pool.conv_state[src_page], dst_pool.conv_state[dst_page]))
            ssm_copies.append((src_pool.ssm_state[src_page], dst_pool.ssm_state[dst_page]))
        copy_bounded_att_tensors(copies=conv_copies)
        copy_bounded_att_tensors(copies=ssm_copies)

    def copy_runtime_state_from_dp(
        self,
        src_req_indices: List[int],
        dst_req_indices: List[int],
        remote_dp_bounded_state: ReplayKDAState,
    ) -> None:
        """复制同一 TP 分片的 prefill 基准运行态并重置目标游标，不写入页池。"""
        assert isinstance(src_req_indices, list) and isinstance(dst_req_indices, list), "request indices must be lists"
        assert len(src_req_indices) == len(dst_req_indices), "source and destination request counts differ"
        assert (
            type(remote_dp_bounded_state) is type(self) and remote_dp_bounded_state.config == self.config
        ), "DP copy requires the same algorithm and equivalent TP state configurations"
        width = self.config.conv_width
        conv_copies = []
        for src_req, dst_req in zip(src_req_indices, dst_req_indices):
            self.runtime.clear_history(dst_req)
            conv_copies.append(
                (
                    remote_dp_bounded_state._accepted_conv(src_req),
                    self.conv_state[:, dst_req, :, :width],
                )
            )
        copy_bounded_att_tensors(copies=conv_copies)
        remote_dp_bounded_state.runtime.export_ssm(
            src_req_indices, [self.ssm_state[:, dst_req] for dst_req in dst_req_indices]
        )
        if self.mtp_state_index is not None:
            for dst_req in dst_req_indices:
                self.mtp_state_index[dst_req].fill_(0)

    # ========================================================================
    # 五、PD 分离传输
    # ========================================================================

    @deepcopy_result
    @lru_cache(maxsize=None)
    def get_pd_transfer_regions(self) -> tuple[TransferRegion, ...]:
        """返回 PD 传输字段与切片，交由管理器安排其在联合页中的位置。"""
        return self._get_transfer_regions()

    def write_to_pd_page(
        self,
        req_idx: int,
        pd_page_buffers: torch.Tensor,
        pd_page_index: int,
        state_token_len: int,
        layout: StateTransferLayout,
    ) -> None:
        """写出已接受 Conv 窗口和物化 SSM，不改写源 history。"""
        assert layout.state_name == self.state_name, "transfer layout does not describe this resource"
        assert 0 <= layout.byte_start < layout.byte_end, "invalid State byte interval"
        accepted = self.get_accepted_state(req_idx)
        fields = self._fields(accepted.conv, accepted.ssm)
        for copies in bounded_att_buffer_copies(
            page_fields=[fields],
            page_buffers=pd_page_buffers,
            layout=layout,
            page_indices=[pd_page_index],
            write=True,
        ):
            copy_bounded_att_tensors(copies=copies)

    def read_from_pd_page(
        self,
        pd_page_buffers: torch.Tensor,
        req_idx: int,
        pd_page_index: int,
        state_token_len: int,
        layout: StateTransferLayout,
    ) -> None:
        """恢复 state_token_len 处的基准状态并规范化控制信息；Conv/SSM 不依赖长度定位。"""
        assert layout.state_name == self.state_name, "transfer layout does not describe this resource"
        assert 0 <= layout.byte_start < layout.byte_end, "invalid State byte interval"
        self.runtime.clear_history(req_idx)
        fields = self._fields(
            self.conv_state[:, req_idx, :, : self.config.conv_width],
            self.ssm_state[:, req_idx],
        )
        for copies in bounded_att_buffer_copies(
            page_fields=[fields],
            page_buffers=pd_page_buffers,
            layout=layout,
            page_indices=[pd_page_index],
            write=False,
        ):
            copy_bounded_att_tensors(copies=copies)
        if self.mtp_state_index is not None:
            self.mtp_state_index[req_idx].fill_(0)

    # ========================================================================
    # 六、私有辅助实现
    # ========================================================================

    def _get_transfer_regions(self) -> tuple[TransferRegion, ...]:
        """依次描述全局 Conv 和 SSM 区域，并标明当前 TP rank 的读写范围。"""
        # 阶段一：确定全局 Conv 形状；只传有效历史窗口，不包含 MTP 候选区和游标。
        config = self.config
        storage_layer_count = len(config.logical_layer_ids)
        conv_channels = 2 * config.global_k_heads * config.key_dim + config.global_v_heads * config.value_dim
        conv_shape = (storage_layer_count, conv_channels, config.conv_width)
        conv_byte_end = prod(conv_shape) * config.conv_dtype.itemsize
        regions = []
        logger.debug(
            "%s building transfer regions: TP=%d/%d, Conv shape=%s",
            self.state_name,
            config.tp_rank,
            config.tp_world_size,
            conv_shape,
        )

        # 阶段二：Conv 按 Q/K/V 拼接；三个映射共用同一块全局 Conv 区域，不能重复累计字节数。
        local_channel_start = 0
        global_channel_start = 0
        for name, local_head_count, global_head_count, head_dim in (
            ("Q", config.num_k_heads, config.global_k_heads, config.key_dim),
            ("K", config.num_k_heads, config.global_k_heads, config.key_dim),
            ("V", config.num_v_heads, config.global_v_heads, config.value_dim),
        ):
            # 配置已保证完整分片或副本。每组连续 rank 保存相同 heads，仅组内首 rank 写出。
            replica_count = config.tp_world_size * local_head_count // global_head_count
            head_start = config.tp_rank // replica_count * local_head_count
            write_enabled = config.tp_rank % replica_count == 0
            local_channel_count = local_head_count * head_dim
            target_channel_start = global_channel_start + head_start * head_dim
            conv_region = TransferRegion(
                field="conv",
                byte_start=0,
                byte_end=conv_byte_end,
                dtype=config.conv_dtype,
                axes=("layer", "channel", "window"),
                shape=conv_shape,
                axis_mappings=(
                    AxisMapping(
                        axis="channel",
                        local_start=local_channel_start,
                        external_start=target_channel_start,
                        length=local_channel_count,
                    ),
                    AxisMapping(
                        axis="window",
                        local_start=0,
                        external_start=0,
                        length=config.conv_width,
                    ),
                ),
                write_enabled=write_enabled,
            )
            regions.append(conv_region)
            logger.debug(
                "%s Conv %s: local_channel_start=%d, global_channel_start=%d, "
                "channels=%d, replicas=%d, write_enabled=%s",
                self.state_name,
                name,
                local_channel_start,
                target_channel_start,
                local_channel_count,
                replica_count,
                write_enabled,
            )
            local_channel_start += local_channel_count
            global_channel_start += global_head_count * head_dim

        # 阶段三：SSM 沿 V head 切分；接在 Conv 之后，按 SSM dtype 对齐字节起点。
        replica_count = config.tp_world_size * config.num_v_heads // config.global_v_heads
        head_start = config.tp_rank // replica_count * config.num_v_heads
        write_enabled = config.tp_rank % replica_count == 0
        ssm_shape = (storage_layer_count, config.global_v_heads, config.key_dim, config.value_dim)
        element_bytes = config.ssm_dtype.itemsize
        ssm_byte_start = (conv_byte_end + element_bytes - 1) // element_bytes * element_bytes
        ssm_region = TransferRegion(
            field="ssm",
            byte_start=ssm_byte_start,
            byte_end=ssm_byte_start + prod(ssm_shape) * element_bytes,
            dtype=config.ssm_dtype,
            axes=("layer", "head", "key_dim", "value_dim"),
            shape=ssm_shape,
            axis_mappings=(
                AxisMapping(
                    axis="head",
                    local_start=0,
                    external_start=head_start,
                    length=config.num_v_heads,
                ),
                AxisMapping(
                    axis="key_dim",
                    local_start=0,
                    external_start=0,
                    length=config.key_dim,
                ),
                AxisMapping(
                    axis="value_dim",
                    local_start=0,
                    external_start=0,
                    length=config.value_dim,
                ),
            ),
            write_enabled=write_enabled,
        )
        regions.append(ssm_region)
        logger.debug(
            "%s transfer regions built: SSM head_start=%d, heads=%d, write_enabled=%s, SSM bytes=[%d, %d)",
            self.state_name,
            head_start,
            config.num_v_heads,
            write_enabled,
            ssm_region.byte_start,
            ssm_region.byte_end,
        )
        return tuple(regions)

    def _pool(self, page_kind: PageKind):
        """根据大小页类型取得已启用的页池，不访问 allocator。"""
        assert page_kind in ("big", "small"), f"unknown page kind {page_kind}"
        pool = self.big_pages_pool if page_kind == "big" else self.small_pages_pool
        assert pool is not None, f"{page_kind} pages are disabled for {self.state_name}"
        return pool

    def _runtime_fields(self, req: int):
        """返回请求基准位置的 Conv/SSM 视图，供恢复和传输使用。"""
        # TODO：当前 DP/PD 不会发生在 MTP 提交后，源状态游标为 0。
        # 若以后支持该场景，需要按已接受游标选取 Conv 窗口及 SSM 槽位。
        return self._fields(self.conv_state[:, req, :, : self.config.conv_width], self.ssm_state[:, req])

    @staticmethod
    def _fields(conv: torch.Tensor, ssm: torch.Tensor):
        return {
            "conv": TensorField(tensor=conv, axes=("layer", "channel", "window")),
            "ssm": TensorField(tensor=ssm, axes=("layer", "head", "key_dim", "value_dim")),
        }


from typing import NamedTuple


class _AcceptedTensors(NamedTuple):
    conv: torch.Tensor
    ssm: torch.Tensor


def _replay_settings(config: dict | None, mtp_step: int) -> tuple[int, str]:
    config = dict(config or {})
    capacity = int(config.get("cache_len", 16))
    compute_mode = config.get("compute_mode", "precompute" if mtp_step > 0 else "inline")
    return capacity, compute_mode


def _find_replay_state(att_state):
    manager = getattr(att_state.infer_state, "att_state_manager", None)
    if manager is None:
        return None
    role = att_state.infer_state.att_state_role
    matches = [state for state in manager.get_bounded_attention_states(role) if isinstance(state, ReplayKDAState)]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"expected one ReplayKDAState for {role}, found {len(matches)}")
    return matches[0]


def prepare_replay_kda_prefill(att_state) -> bool:
    """Replay prefill 先折叠 history，并用单槽 SSM 索引。"""

    state = _find_replay_state(att_state)
    if state is None:
        return False
    reqs = att_state.infer_state.b_req_idx
    state.merge_accepted_updates(reqs)
    att_state.b_conv_buffer_idx = reqs
    att_state.b_ssm_buffer_idx = reqs
    att_state._replay_state = state
    return True


def prepare_replay_kda_decode(att_state) -> bool:
    """准备 KDA replay decode 的 conv/MTP 索引，并折叠本轮 history。"""

    state = _find_replay_state(att_state)
    if state is None:
        return False
    draft_step = att_state.backend.mtp_step
    infer = att_state.infer_state
    if draft_step == 0:
        att_state.b_conv_buffer_idx = infer.b_req_idx
        att_state.b_ssm_buffer_idx = infer.b_req_idx
        att_state.b1_mtp_cu_q_seq_len = None
        att_state.b_num_accepted_tokens = None
    else:
        if att_state.backend.uses_dynamic_spec_verify_layout():
            from lightllm.common.basemodel.triton_kernel.linear_att.mtp_state_params import (
                build_dynamic_mtp_linear_att_state_params,
            )

            (
                att_state.b1_mtp_cu_q_seq_len,
                att_state.b_conv_buffer_idx,
                att_state.b_num_accepted_tokens,
            ) = build_dynamic_mtp_linear_att_state_params(
                b_req_idx=infer.b_req_idx,
                b_mtp_index=infer.b_mtp_index,
                req_to_mtp_state_index=state.mtp_state_index,
                hold_req_id=infer.req_manager.HOLD_REQUEST_ID,
            )
        else:
            width = draft_step + 1
            batch_size = infer.batch_size
            if batch_size % width:
                raise ValueError(
                    "KDA fixed-layout decode requires batch_size to be divisible by draft_step + 1, "
                    f"got batch_size={batch_size}, draft_step={draft_step}."
                )
            att_state.b1_mtp_cu_q_seq_len = torch.arange(
                0, batch_size + 1, width, dtype=torch.int32, device=infer.b_req_idx.device
            )
            att_state.b_conv_buffer_idx = infer.b_req_idx.view(-1, width)[:, 0].contiguous()
            att_state.b_num_accepted_tokens = state.mtp_state_index[att_state.b_conv_buffer_idx] + 1
        att_state.b_ssm_buffer_idx = att_state.b_conv_buffer_idx
    state.prepare_for_decode(att_state.b_conv_buffer_idx, att_state.b1_mtp_cu_q_seq_len)
    att_state._replay_state = state
    return True
