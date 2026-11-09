"""GDN replay 的 bounded attention 状态。

Conv、canonical SSM 和 replay history 都定义在本类。页布局在 replay_gdn_checkpoint，
已接受快照在 replay_gdn_snapshot。一层 decode 重放在 replay_gdn_step。
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import NamedTuple

import torch
import triton

from lightllm.common.basemodel.triton_kernel.linear_att.replayssm import (
    _accept_updates,
    _fold_history,
    _materialize,
    _prepare_fold,
)
from lightllm.common.basemodel.triton_kernel.linear_att.ssm_acceptance import commit_ssm_prefix
from lightllm.common.ssm_types import SSMRunConfig

from ...utils import deepcopy_result, dtype_name
from ..base_state import BoundedAttentionState
from ..conv_ssm_types import ConvSSMStateConfig
from .replay_gdn_checkpoint import (
    build_transfer_regions,
    copy_big_pages,
    init_page_pools,
    read_pages_from_cpu,
    write_pages_to_cpu,
)
from .replay_gdn_snapshot import (
    accepted_conv,
    copy_runtime_from_dp,
    read_pd,
    restore_runtime,
    save_runtime,
    write_pd,
)


class _AcceptedTensors(NamedTuple):
    conv: torch.Tensor
    ssm: torch.Tensor


def _replay_settings(config: dict | None, mtp_step: int) -> tuple[int, str]:
    config = dict(config or {})
    capacity = int(config.get("cache_len", 16))
    compute_mode = config.get("compute_mode", "precompute" if mtp_step > 0 else "inline")
    return capacity, compute_mode


def _upload_indices(values: Sequence[int], device, dtype=torch.int64) -> torch.Tensor:
    """把 Python 索引上传到目标 device。"""

    return torch.tensor(list(values), dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def _bind_history(
    owner,
    ssm: torch.Tensor,
    capacity: int,
    verify_width: int,
    activation_dtype: torch.dtype,
    num_key_heads: int | None,
    gate_lower_bound: float,
    compute_mode: str,
    run_config: SSMRunConfig | dict[str, int] | None,
) -> None:
    """在 owner 上创建 GDN replay 的双半区 history 和 cursor。"""

    assert ssm.dtype in (torch.float32, torch.bfloat16), "GDN replay requires FP32 or BF16 SSM state"
    assert verify_width > 0 and capacity >= max(4, verify_width) and capacity & (capacity - 1) == 0
    assert compute_mode in ("inline", "precompute")
    owner.ssm_state = ssm
    owner.kda = False
    owner.gate_lower_bound = gate_lower_bound
    owner.fold_programs = 8 * torch.cuda.get_device_properties(ssm.device).multi_processor_count if ssm.is_cuda else 0
    owner.capacity = capacity
    owner.verify_width = verify_width
    owner.compute_mode = compute_mode
    owner.run_config = run_config
    owner._config_is_fixed = run_config is not None
    owner.batch_configs = {}
    layers, slots, hv, k, v = ssm.shape
    owner.num_key_heads = hv if num_key_heads is None else num_key_heads
    assert owner.num_key_heads > 0 and hv % owner.num_key_heads == 0
    owner.hold_req_id = slots - 1
    owner.cursors = torch.zeros(slots, dtype=torch.int32, device=ssm.device)
    owner.forward_cursors = torch.zeros_like(owner.cursors)
    # fold 时交替使用两个半区。precompute 且 BF16、短 history 时 history 用激活精度，其余用 FP32。
    history_dtype = (
        activation_dtype
        if activation_dtype == torch.bfloat16 and verify_width > 1 and compute_mode == "precompute" and capacity <= 8
        else torch.float32
    )
    key_shape = (layers, slots, owner.num_key_heads, 2 * capacity, k)
    owner.keys = torch.empty(key_shape, dtype=history_dtype, device=ssm.device)
    owner.deltas = torch.empty((layers, slots, hv, 2 * capacity, v), dtype=history_dtype, device=ssm.device)
    owner.gates = torch.empty((layers, slots, hv, 2 * capacity), dtype=torch.float32, device=ssm.device)
    owner.raw_keys = torch.empty(key_shape, dtype=activation_dtype, device=ssm.device)
    owner.raw_values = torch.empty((layers, slots, hv, 2 * capacity, v), dtype=activation_dtype, device=ssm.device)
    owner.betas = torch.empty((layers, slots, hv, 2 * capacity), dtype=torch.float32, device=ssm.device)


class ReplayGDNState(BoundedAttentionState):
    """GDN replay 运行态。Conv、单槽 SSM 和 history 都在本类，不沿用 native 的 MTP 槽。"""

    _run_config: SSMRunConfig | None = None

    @property
    def state(self) -> torch.Tensor:
        """Replay kernel 读取的 canonical SSM，与 ssm_state 是同一张量。"""

        return self.ssm_state

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
        config: ConvSSMStateConfig,
        request_capacity: int,
        mtp_step: int = 0,
        device: str | torch.device = "cuda",
        *,
        ssm_state_mode: str = "replay",
        ssm_replay_config: dict | None = None,
        gate_lower_bound: float = -5.0,
    ) -> None:
        """创建单槽 canonical SSM，并挂上只属于当前请求的 replay history。"""
        if ssm_state_mode != "replay":
            raise ValueError(f"ReplayGDNState only implements replay mode, got {ssm_state_mode}")
        assert isinstance(config, ConvSSMStateConfig), "config must be ConvSSMStateConfig"
        self.config = config
        self.state_name = None
        self.request_capacity = request_capacity
        self.mtp_step = mtp_step
        self.gate_lower_bound = gate_lower_bound
        self._algorithm = "gdn"
        self._logical_to_storage_layer = {layer_id: index for index, layer_id in enumerate(config.logical_layer_ids)}
        layer_count = len(config.logical_layer_ids)
        self.conv_state = torch.zeros(
            (
                layer_count,
                request_capacity,
                config.conv_channels,
                config.conv_width + mtp_step,
            ),
            dtype=config.conv_dtype,
            device=device,
        )
        self.ssm_state = torch.zeros(
            (
                layer_count,
                request_capacity,
                config.num_v_heads,
                config.key_dim,
                config.value_dim,
            ),
            dtype=config.ssm_dtype,
            device=device,
        )
        self.mtp_state_index = torch.zeros(request_capacity, dtype=torch.int32, device=device) if mtp_step else None
        self.big_pages_pool = None
        self.small_pages_pool = None
        capacity, compute_mode = _replay_settings(ssm_replay_config, mtp_step)
        _bind_history(
            self,
            self.ssm_state,
            capacity,
            mtp_step + 1,
            config.conv_dtype,
            config.num_k_heads,
            gate_lower_bound,
            compute_mode,
            None,
        )

    @classmethod
    def from_ssm(
        cls,
        ssm: torch.Tensor,
        capacity: int,
        verify_width: int,
        activation_dtype: torch.dtype = torch.bfloat16,
        *,
        num_key_heads: int | None = None,
        kda: bool = False,
        gate_lower_bound: float = -5.0,
        compute_mode: str = "inline",
        run_config: SSMRunConfig | dict[str, int] | None = None,
    ) -> "ReplayGDNState":
        """用已有 SSM 张量组装 history。kernel 测试和 autotune scratch 走这里。"""

        if kda:
            raise ValueError("GDN replay does not use the KDA gate")
        owner = cls.__new__(cls)
        owner.config = None
        owner.state_name = None
        owner.request_capacity = ssm.shape[1]
        owner.mtp_step = verify_width - 1
        owner._algorithm = "gdn"
        owner._logical_to_storage_layer = {}
        owner.conv_state = None
        owner.mtp_state_index = None
        owner.big_pages_pool = None
        owner.small_pages_pool = None
        _bind_history(
            owner,
            ssm,
            capacity,
            verify_width,
            activation_dtype,
            num_key_heads,
            gate_lower_bound,
            compute_mode,
            run_config,
        )
        return owner

    @property
    def ssm_slots_per_req(self) -> int:
        """每个请求只保留一份 canonical SSM。"""

        return 1

    @property
    def _ssm_executor(self):
        """History 张量在本对象上。测试沿用这个入口。"""

        return self

    def get_state_type(self) -> str:
        """返回算法名和 Conv/SSM 存储精度。"""
        return (
            f"{self._algorithm}_conv_{dtype_name(dtype=self.config.conv_dtype)}"
            f"_ssm_{dtype_name(dtype=self.config.ssm_dtype)}"
        )

    def initialize_page_pools(
        self,
        big_page_capacity: int,
        small_page_capacity: int = 0,
        device: str | torch.device | int = "cpu",
        pin_memory: bool = True,
    ) -> None:
        init_page_pools(self, big_page_capacity, small_page_capacity, device, pin_memory)

    def contains_logical_layer(self, logical_layer_id: int) -> bool:
        return logical_layer_id in self._logical_to_storage_layer

    def storage_layer(self, logical_layer_id: int) -> int:
        """返回逻辑层在 history 和 canonical state 里的存储下标。"""

        return self._logical_to_storage_layer[logical_layer_id]

    def get_layer_view(self, logical_layer_id: int):
        index = self.storage_layer(logical_layer_id)
        return self.conv_state[index], self.ssm_state[index]

    @deepcopy_result
    @lru_cache(maxsize=None)
    def get_cpu_transfer_regions(self):
        return build_transfer_regions(self)

    def write_state_pages_to_cpu(
        self,
        page_kind,
        page_slot_ids,
        cpu_page_buffers,
        cpu_page_indices,
        layout,
        cpu_page_ready_flags=None,
    ):
        write_pages_to_cpu(
            self,
            page_kind,
            page_slot_ids,
            cpu_page_buffers,
            cpu_page_indices,
            layout,
            cpu_page_ready_flags,
        )

    def read_state_pages_from_cpu(self, cpu_page_buffers, cpu_page_indices, big_page_slot_ids, layout):
        read_pages_from_cpu(self, cpu_page_buffers, cpu_page_indices, big_page_slot_ids, layout)

    def copy_big_pages_from_dp(self, src_page_ids, dst_page_ids, remote_dp_bounded_state):
        copy_big_pages(self, src_page_ids, dst_page_ids, remote_dp_bounded_state)

    @deepcopy_result
    @lru_cache(maxsize=None)
    def get_pd_transfer_regions(self):
        return build_transfer_regions(self)

    def reset_bounded_att_state(self, req_indices: list[int]) -> None:
        """清零请求的 Conv、单槽 SSM、history 和 MTP 游标。"""
        assert isinstance(req_indices, list), "request indices must be a list"
        for req_idx in req_indices:
            self.conv_state[:, req_idx].zero_()
            self.ssm_state[:, req_idx].zero_()
            self.clear_history(req_idx)
            if self.mtp_state_index is not None:
                self.mtp_state_index[req_idx].fill_(0)

    def save_bounded_att_runtime_to_pages(
        self,
        req_indices: list[int],
        state_token_lens: list[int],
        page_kind,
        page_slot_ids: list[int],
    ) -> None:
        save_runtime(self, req_indices, state_token_lens, page_kind, page_slot_ids)

    def restore_bounded_att_runtime_from_pages(
        self,
        req_indices: list[int],
        state_token_lens: list[int],
        page_kind,
        page_slot_ids: list[int],
    ) -> None:
        restore_runtime(self, req_indices, state_token_lens, page_kind, page_slot_ids)

    def commit_bounded_att_state(self, req_indices: torch.Tensor, accepted_steps: torch.Tensor) -> None:
        """提交已接受前缀，并推进 replay cursor。"""
        if self.mtp_state_index is None:
            return
        self.commit_accepted(self.mtp_state_index, req_indices, accepted_steps)

    def get_accepted_state(self, req_idx: int) -> _AcceptedTensors:
        """返回已接受 Conv 窗口和物化 SSM，不改写 canonical state。"""
        return _AcceptedTensors(accepted_conv(self, req_idx), self.accepted_ssm(req_idx))

    def copy_runtime_state_from_dp(
        self,
        src_req_indices: list[int],
        dst_req_indices: list[int],
        remote_dp_bounded_state: "ReplayGDNState",
    ) -> None:
        copy_runtime_from_dp(self, src_req_indices, dst_req_indices, remote_dp_bounded_state)

    def write_to_pd_page(
        self,
        req_idx: int,
        pd_page_buffers: torch.Tensor,
        pd_page_index: int,
        state_token_len: int,
        layout,
    ) -> None:
        write_pd(self, req_idx, pd_page_buffers, pd_page_index, state_token_len, layout)

    def read_from_pd_page(
        self,
        pd_page_buffers: torch.Tensor,
        req_idx: int,
        pd_page_index: int,
        state_token_len: int,
        layout,
    ) -> None:
        read_pd(self, pd_page_buffers, req_idx, pd_page_index, state_token_len, layout)

    def clear_history(self, req: int | torch.Tensor) -> None:
        """清空复用或恢复请求的临时 history，保留 canonical SSM。"""

        if type(req) is int:
            # 标量赋值会把 CPU tensor 复制到 CUDA slot。
            self.cursors[req].zero_()
        else:
            self.cursors[req] = 0

    def prepare_decode(self, reqs: torch.Tensor, cu_seqlens: torch.Tensor | None = None) -> torch.Tensor:
        """保存本轮 decode 的 history 起点，并折叠即将溢出的旧半区。"""

        positions = torch.empty_like(reqs)
        layers, _, hv, kd, vd = self.ssm_state.shape
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
            self.ssm_state,
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
            self.ssm_state.stride(0),
            self.raw_keys.stride(0),
            self.raw_values.stride(0),
            self.gates.stride(0),
            self.betas.stride(0),
            False,
            num_warps=2,
            num_stages=1,
        )
        return positions

    def accept_updates(self, reqs: torch.Tensor, accepted_state_indices: torch.Tensor | None = None) -> None:
        """更新 cursor，保留每个请求本轮已接受的前缀。"""

        _accept_updates[(triton.cdiv(reqs.numel(), 256),)](
            self.cursors,
            reqs,
            accepted_state_indices,
            reqs.numel(),
            self.hold_req_id,
            self.verify_width,
            256,
        )

    def accept_single_token(self, reqs: torch.Tensor) -> None:
        """非 MTP decode 把单 token 记入当前 history。"""

        self.accept_updates(reqs)

    def merge_accepted_updates(self, reqs: torch.Tensor) -> None:
        """Prefill 前把已接受 history 折回 canonical SSM，并清空 cursor。"""

        if reqs.numel() == 0:
            return
        self._materialize_accepted(reqs, self.ssm_state, snapshot=False)
        self.cursors.index_fill_(0, reqs.long(), 0)

    def _materialize_accepted(
        self,
        reqs: torch.Tensor,
        output: torch.Tensor,
        snapshot: bool,
        output_strides=None,
        flush_export: bool = False,
    ) -> None:
        """将请求的 accepted prefix 写入 canonical SSM 或调用方提供的输出页。"""

        assert not (snapshot and flush_export), "flush export must update the canonical state"
        layers, slots, hv, k, v = self.ssm_state.shape
        config = self.run_config or SSMRunConfig(32, 4)
        bv = config.BV
        _materialize[(triton.cdiv(v, bv), reqs.numel(), layers * hv)](
            self.ssm_state,
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
            False,
            FLUSH_EXPORT=flush_export,
            OUTPUT_STRIDES=output_strides,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )

    def accepted_ssm(self, req_idx: int) -> torch.Tensor:
        """返回不修改 canonical SSM 和 history 的单请求快照。"""

        layers, _, hv, k, v = self.ssm_state.shape
        output = torch.empty((layers, hv, k, v), dtype=self.ssm_state.dtype, device=self.ssm_state.device)
        reqs = torch.tensor([req_idx], dtype=torch.int32, device=self.ssm_state.device)
        self._materialize_accepted(reqs, output, snapshot=True)
        return output

    def export_accepted_state(
        self,
        req_indices: list[int],
        targets: list[torch.Tensor],
        accepted_indices: torch.Tensor | None = None,
        *,
        flush: bool = False,
    ) -> None:
        """将多个 accepted 快照直接写入调用方目标页。"""

        if not req_indices:
            return
        shape = (self.ssm_state.shape[0], *self.ssm_state.shape[2:])
        strides = targets[0].stride()
        assert len(req_indices) == len(targets)
        assert all(t.shape == shape and t.stride() == strides and t.dtype == self.ssm_state.dtype for t in targets)
        self._materialize_accepted(
            _upload_indices(req_indices, self.ssm_state.device, torch.int32),
            _upload_indices([t.data_ptr() for t in targets], self.ssm_state.device),
            snapshot=not flush,
            flush_export=flush,
            output_strides=strides,
        )

    def commit_accepted(self, indices: torch.Tensor, reqs: torch.Tensor, lengths: torch.Tensor) -> None:
        """把接受长度写入 MTP 槽位索引，并推进本请求的 replay cursor。"""

        if self.verify_width > 1:
            commit_ssm_prefix(indices, reqs, lengths, self.cursors)
        else:
            commit_ssm_prefix(indices, reqs, lengths)
            self.accept_updates(reqs, indices)

    def commit(self, acceptance, indices: torch.Tensor, state_name: str | None = None) -> None:
        """从 acceptance 取出请求和接受长度，供直接持有 history 的测试调用。"""

        self.commit_accepted(indices, acceptance.req_indices, acceptance.accepted_lengths)

    def export_ssm(self, req_indices: list[int], targets: list[torch.Tensor], *, flush: bool = False) -> None:
        """把已接受 SSM 写入调用方页面。CUDA 目标直接写指针，其他设备先物化再拷贝。"""

        if not req_indices:
            return
        target = targets[0]
        if target.is_cuda and target.device == self.ssm_state.device:
            self.export_accepted_state(req_indices, targets, flush=flush)
            return
        for req, dst in zip(req_indices, targets):
            snap = torch.empty(
                (self.ssm_state.shape[0], *self.ssm_state.shape[2:]),
                dtype=self.ssm_state.dtype,
                device=self.ssm_state.device,
            )
            self._materialize_accepted(
                torch.tensor([req], dtype=torch.int32, device=self.ssm_state.device),
                snap,
                snapshot=True,
            )
            dst.copy_(snap)
            if flush:
                self.ssm_state[:, req].copy_(snap)
                self.clear_history(req)
