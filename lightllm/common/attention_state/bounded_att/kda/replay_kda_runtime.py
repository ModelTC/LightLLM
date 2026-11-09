"""KDA replay history。只保存 ring buffer、cursor，以及 accepted SSM 的物化和拷贝。"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import triton

from lightllm.common.ssm_types import SSMRunConfig
from lightllm.common.basemodel.triton_kernel.linear_att.replayssm import (
    _accept_updates,
    _prepare_fold,
    _fold_history,
    _materialize,
)
from lightllm.common.basemodel.triton_kernel.linear_att.ssm_acceptance import (
    commit_ssm_prefix,
)


def upload_indices(values: Sequence[int], device, dtype=torch.int64) -> torch.Tensor:
    """把 Python 索引上传到目标 device。"""

    return torch.tensor(list(values), dtype=dtype, pin_memory=True).to(device, non_blocking=True)


class ReplayKDARuntime:
    """请求槽上的 KDA replay history。

    ``state`` 是唯一的 canonical checkpoint；keys、deltas、gates、betas 和
    cursor 只属于请求当前的临时 history，不参与 CPU/PD checkpoint。decode
    先从 checkpoint 加载 history，再把新 token 写入 ring buffer；接受或
    checkpoint 导出时，materialize kernel 将已接受前缀折叠回 ``state``。
    """

    _run_config: SSMRunConfig | None = None

    @property
    def run_config(self) -> SSMRunConfig | None:
        """返回物化和 autotune 使用的 kernel 配置。"""

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
        """创建 KDA replay 的 canonical state、双半区 history 和 cursor。

        ``verify_width`` 决定一次 verify 的最大 token 数，``capacity`` 是
        history ring 的半区长度；``compute_mode`` 选择 projection 的执行方式。
        """

        assert state.dtype in (
            torch.float32,
            torch.bfloat16,
        ), "KDA replay requires FP32 or BF16 SSM state"
        assert verify_width > 0 and capacity >= max(4, verify_width) and capacity & (capacity - 1) == 0
        assert compute_mode in ("inline", "precompute")
        if not kda:
            raise ValueError("KDA replay runtime requires the KDA gate")
        self.state = state
        self.kda = True
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
        self.deltas = torch.empty(
            (layers, slots, hv, 2 * capacity, v),
            dtype=history_dtype,
            device=state.device,
        )
        self.gates = torch.empty(
            (layers, slots, hv, 2 * capacity, k),
            dtype=torch.float32,
            device=state.device,
        )
        self.raw_keys = torch.empty(key_shape, dtype=activation_dtype, device=state.device)
        self.raw_values = torch.empty(
            (layers, slots, hv, 2 * capacity, v),
            dtype=activation_dtype,
            device=state.device,
        )
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

    def export_ssm(
        self,
        req_indices: list[int],
        targets: list[torch.Tensor],
        *,
        flush: bool = False,
    ) -> None:
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
