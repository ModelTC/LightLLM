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
from .base import SSMExecutor, update_accepted_indices
from .export import upload_indices


class ReplaySSMExecutor(SSMExecutor):
    """以短 history 重放 decode 的 SSM 执行器。

    ``state`` 是唯一的 canonical checkpoint；keys、deltas、gates、betas 和
    cursor 只属于请求当前的临时 history，不参与 CPU/PD checkpoint。decode
    先从 checkpoint 加载 history，再把新 token 写入 ring buffer；接受或
    checkpoint 导出时，materialize kernel 将已接受前缀折叠回 ``state``。
    """

    def __init__(
        self,
        state,
        capacity,
        verify_width,
        activation_dtype=torch.bfloat16,
        *,
        num_key_heads=None,
        kda=False,
        gate_lower_bound=-5.0,
        compute_mode="inline",
        run_config=None,
    ):
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
        # Omission preserves the redundant layout for existing callers and A/B.
        self.num_key_heads = hv if num_key_heads is None else num_key_heads
        assert self.num_key_heads > 0 and hv % self.num_key_heads == 0
        self.hold_req_id = slots - 1
        self.cursors = torch.zeros(slots, dtype=torch.int32, device=state.device)
        self.forward_cursors = torch.zeros_like(self.cursors)
        # Alternate halves on fold so CTAs cannot overwrite history another
        # V tile is still reading. Cursor packs phase (2*L) and count (0..L).
        # BF16 can use a narrower short-history buffer for the precompute path;
        # FP32 remains the safe default for all other shapes.
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

    def clear_history(self, req):
        """Invalidate history for a reused/restored request; leave its SSM state intact."""
        if type(req) is int:
            # Scalar assignment copies a CPU tensor into the CUDA slot.
            self.cursors[req].zero_()
        else:
            self.cursors[req] = 0

    def prepare_decode(self, reqs, cu_seqlens=None):
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

    def accept_updates(self, reqs, accepted_state_indices=None):
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

    def merge_accepted_updates(self, reqs):
        """把接受的 history 物化回 canonical state，并清空请求 cursor。"""
        if reqs.numel() == 0:
            return
        self._materialize_accepted_state(reqs, self.state, snapshot=False)
        self.cursors.index_fill_(0, reqs.long(), 0)

    def _materialize_accepted_state(self, reqs, output, snapshot, output_strides=None, flush_export=False):
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

    def get_accepted_state(self, req_idx, accepted_index=None):
        """返回不修改 active state/history 的单请求 canonical 快照。"""
        layers, _, hv, k, v = self.state.shape
        output = torch.empty((layers, hv, k, v), dtype=self.state.dtype, device=self.state.device)
        reqs = torch.tensor([req_idx], dtype=torch.int32, device=self.state.device)
        self._materialize_accepted_state(reqs, output, snapshot=True)
        return output

    def export_accepted_state(self, req_indices, targets, accepted_indices=None, *, flush=False):
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

    def get_run_config(self, q, k, v, a, b, a_log, bias, cu_seqlens=None) -> SSMRunConfig:
        """按当前 kernel 形状选择或返回固定的 ReplaySSM 调优配置。"""

        return resolve_run_config(self, "replay", q, k, v, a, b, a_log, bias, cu_seqlens)

    def forward(
        self,
        linear_layer_idx,
        q,
        k,
        v,
        a,
        b,
        a_log,
        bias,
        reqs,
        cu_seqlens=None,
        run_config=None,
        *,
        state_indices=None,
        accepted_token_counts=None,
    ):
        """执行一层 ReplaySSM decode，并返回输出与本轮 run config。"""

        config = run_config or self.get_run_config(q, k, v, a, b, a_log, bias, cu_seqlens)
        hv, kd, vd = self.state.shape[-3:]
        assert self.num_key_heads in (q.shape[-2], hv)
        axis = 1 if cu_seqlens is not None else 0
        out = torch.empty_like(v)
        bv = config.BV
        precompute_projections = self.verify_width > 1 and bv >= 16 and self.compute_mode == "precompute"
        # Resolve tile dimensions in Python: bare max() in Triton JIT code
        # lowers to a tensor operation, which cannot define static shapes.
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

    def accept_single_token(self, reqs):
        """非 MTP decode 直接把单 token 写入当前 history。"""
        self.accept_updates(reqs)

    def commit(self, acceptance, indices, state_name):
        """应用接受结果；MTP 直接推进 cursor，单 token 还需写入 history。"""

        self.commit_validated(acceptance, indices, state_name)

    def commit_validated(self, acceptance, indices, state_name, *, commit_data=None):
        """应用已验证的接受结果，推进 replay cursor。"""

        if self.verify_width > 1:
            update_accepted_indices(acceptance, indices, replay_cursors=self.cursors)
        else:
            update_accepted_indices(acceptance, indices)
            self.accept_updates(acceptance.req_indices, indices)
