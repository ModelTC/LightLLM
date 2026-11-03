import torch
import triton

from lightllm.common.ssm_types import SSMRunConfig
from lightllm.common.basemodel.triton_kernel.linear_att.compact_ssm import _compact
from lightllm.common.basemodel.triton_kernel.linear_att.ssm_autotune import resolve_run_config
from .base import SSMState, update_accepted_indices


class CompactSSMState(SSMState):
    def __init__(
        self,
        state,
        verify_width,
        activation_dtype,
        *,
        num_key_heads=None,
        kda=False,
        gate_lower_bound=-5.0,
        run_config=None,
    ):
        self.state = state
        self.kda = kda
        self.gate_lower_bound = gate_lower_bound
        self.verify_width = verify_width
        # Verify and accepted-state reconstruction must share the same layout.
        self._config_is_fixed = run_config is not None
        self.batch_configs = {}
        self.run_config = dict(run_config or {"BV": 8, "num_warps": 1})
        assert self.run_config["BV"] in (8, 16, 32, 64, 128)
        assert self.run_config["num_warps"] in (1, 2, 4, 8)
        layers, slots, hv, k, v = state.shape
        self.num_key_heads = hv if num_key_heads is None else num_key_heads
        assert self.num_key_heads > 0 and hv % self.num_key_heads == 0
        self.hold_req_id = slots - 1
        shape = (layers, slots, hv, verify_width)
        self.keys = torch.empty(
            (layers, slots, self.num_key_heads, verify_width, k), device=state.device, dtype=activation_dtype
        )
        self.values = torch.empty((*shape, v), device=state.device, dtype=activation_dtype)
        self.decays = torch.empty((*shape, k) if kda else shape, device=state.device, dtype=torch.float32)
        self.betas = torch.empty(shape, device=state.device, dtype=torch.float32)

    def get_run_config(self, q, k, v, a, b, a_log, bias, cu_seqlens=None) -> SSMRunConfig:
        config = resolve_run_config(self, "compact", q, k, v, a, b, a_log, bias, cu_seqlens)
        return SSMRunConfig(**config)

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
        assert cu_seqlens is not None, "compact replay is only used for speculative verify"
        config = run_config or self.get_run_config(q, k, v, a, b, a_log, bias, cu_seqlens)
        if not isinstance(config, SSMRunConfig):
            config = SSMRunConfig(**config)
        _, slots, hv, kd, vd = self.state.shape
        assert self.num_key_heads in (q.shape[-2], hv)
        out = torch.empty_like(v)
        # Match native reduction layouts: changing BV can alter BF16 rounding.
        bv = config.BV
        _compact[(triton.cdiv(vd, bv), reqs.numel(), hv)](
            q,
            k,
            v,
            a,
            b,
            a_log,
            bias,
            self.state[linear_layer_idx],
            self.keys[linear_layer_idx],
            self.values[linear_layer_idx],
            self.decays[linear_layer_idx],
            self.betas[linear_layer_idx],
            reqs,
            cu_seqlens,
            None,
            out,
            q.stride(1),
            k.stride(1),
            v.stride(1),
            a.stride(0),
            b.stride(0),
            slots,
            q.shape[-2],
            hv,
            self.num_key_heads,
            kd,
            vd,
            self.verify_width,
            self.hold_req_id,
            triton.next_power_of_2(kd),
            bv,
            False,
            self.kda,
            self.gate_lower_bound,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )
        return out, config

    def accept_updates(self, reqs, accepted_state_indices, run_config=None):
        """Merge this round's prefix; accepted_state_indices contains per-request last accepted indexes."""
        assert run_config is not None or not self.batch_configs, "compact acceptance requires the verify configuration"
        layers, slots, hv, kd, vd = self.state.shape
        config = run_config or self.run_config
        if not isinstance(config, SSMRunConfig):
            config = SSMRunConfig(**config)
        bv = config.BV
        _compact[(triton.cdiv(vd, bv), reqs.numel(), layers * hv)](
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            self.state,
            self.keys,
            self.values,
            self.decays,
            self.betas,
            reqs,
            None,
            accepted_state_indices,
            None,
            0,
            0,
            0,
            0,
            0,
            slots,
            hv,
            hv,
            self.num_key_heads,
            kd,
            vd,
            self.verify_width,
            self.hold_req_id,
            triton.next_power_of_2(kd),
            bv,
            True,
            self.kda,
            self.gate_lower_bound,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )

    def create_commit(self, run_config):
        if not isinstance(run_config, SSMRunConfig):
            raise ValueError("compact commit requires the configuration of its forward")
        return run_config

    def validate_commit(self, context, state_name):
        if context is None or not context.batches:
            raise ValueError("compact acceptance requires forward commit batches")
        for _, config in context.partitions(state_name):
            if not isinstance(config, SSMRunConfig):
                raise ValueError(f"missing compact forward configuration for {state_name}")

    def commit(self, reqs, accepted_steps, indices, context, state_name):
        self.validate_commit(context, state_name)
        update_accepted_indices(reqs, accepted_steps, indices, context)
        for partition, config in context.partitions(state_name):
            self.accept_updates(reqs[partition], indices, run_config=config)
