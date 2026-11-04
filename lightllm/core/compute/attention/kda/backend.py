"""Kimi Delta Attention bound to Conv/SSM views and per-layer gate parameters."""

from dataclasses import dataclass
from importlib.util import find_spec

import torch

from lightllm.core.compute.attention.gdn.backend import GDNBatchMetadata
from lightllm.core.contracts.provider import AttentionOffer, ProbeResult
from lightllm.core.compute.attention.kda.spec import KimiDeltaAttentionSpec


@dataclass(frozen=True)
class KDAStaticParams:
    spec: KimiDeltaAttentionSpec
    conv_weight: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor


@dataclass(frozen=True)
class KDAInputs:
    qkv: torch.Tensor
    gate: torch.Tensor
    beta: torch.Tensor


class BoundKimiDeltaAttention:
    transfer_requires_compute_sync = False

    def __init__(self, offer, view, params):
        self.name, self.phase, self.spec = offer.backend, offer.phase, params.spec
        self.conv, self.ssm = view.fields[:2]
        self.candidate_slots = view.format.candidate_slots
        self.accepted_index = view.fields[2] if self.candidate_slots > 1 else None
        self.metadata_key = (
            (self.name, self.accepted_index.data_ptr()) if self.accepted_index is not None else self.name
        )
        self.params, self.limits = params, offer.execution_limits

    def create_metadata(self, batch):
        if batch.batch_size > self.limits.max_batch_size or batch.max_q_len > self.limits.max_seq_len:
            raise ValueError("batch exceeds planned KDA capacity")
        if not 0 <= batch.microbatch_index < self.limits.microbatch_count:
            raise ValueError("batch exceeds planned KDA lanes")
        if self.phase == "decode" and batch.max_q_len != 1:
            raise ValueError("KDA verification requires its own execution plan")
        initial = batch.ready_lengths > 0 if self.phase == "prefill" else None
        requests, cu_q = batch.req_indices, batch.cu_q
        indices, accepted = requests, None
        if self.phase == "verify":
            from lightllm.common.basemodel.triton_kernel.linear_att.mtp_state_params import (
                build_dynamic_mtp_linear_att_state_params,
            )

            if batch.max_q_len != 1 or batch.mtp_indices is None or batch.hold_request_id is None:
                raise ValueError("KDA verification requires candidate rows and a padding request identity")
            cu_q, requests, accepted = build_dynamic_mtp_linear_att_state_params(
                requests, batch.mtp_indices, self.accepted_index, batch.hold_request_id
            )
            indices = requests[:, None] * self.candidate_slots + torch.arange(
                self.candidate_slots, dtype=requests.dtype, device=requests.device
            )
        elif self.accepted_index is not None:
            accepted = self.accepted_index[requests] + 1
            indices = requests * self.candidate_slots + accepted - 1
        return GDNBatchMetadata(requests, cu_q, initial, indices, accepted)

    def run(self, inputs, metadata):
        # The convolution may consume its temporary QKV projection in place.
        from lightllm.common.basemodel.triton_kernel.linear_att.causal_conv1d import (
            causal_conv1d_fn,
            causal_conv1d_update,
        )

        spec, params = self.spec, self.params
        count, width = inputs.qkv.shape[0], spec.heads * spec.head_dim
        for tensor, shape in (
            (inputs.qkv, (count, 3 * width)),
            (inputs.gate, (count, width)),
            (inputs.beta, (count, spec.heads)),
        ):
            if tensor.shape != shape or tensor.dtype != spec.dtype or tensor.device != self.conv.device:
                raise ValueError("KDA inputs disagree with the bound projection layout")
        if self.phase == "prefill":
            from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops.kda import chunk_kda_with_fused_gate

            conv = self.conv[..., : spec.conv_size - 1]
            if metadata.accepted_tokens is not None:
                offsets = (
                    metadata.accepted_tokens[:, None, None]
                    - 1
                    + torch.arange(spec.conv_size - 1, device=self.conv.device)
                )
                history = self.conv[metadata.request_indices].gather(-1, offsets.expand(-1, 3 * width, -1))
                conv[metadata.request_indices] = history
            mixed = causal_conv1d_fn(
                inputs.qkv.transpose(0, 1),
                params.conv_weight,
                bias=None,
                query_start_loc=metadata.cu_q,
                cache_indices=metadata.request_indices,
                has_initial_state=metadata.has_initial_state,
                conv_states=conv,
                activation="silu",
            ).transpose(0, 1)
            q, k, v = (t.reshape(1, count, spec.heads, spec.head_dim) for t in mixed.split(width, -1))
            initial = self.ssm[metadata.ssm_indices]
            initial = torch.where(metadata.has_initial_state[:, None, None, None], initial, 0)
            output, final = chunk_kda_with_fused_gate(
                q=q,
                k=k,
                v=v,
                raw_g=inputs.gate.reshape(1, count, spec.heads, spec.head_dim),
                beta=inputs.beta.float().sigmoid().unsqueeze(0),
                A_log=params.a_log,
                g_bias=params.dt_bias,
                initial_state=initial,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=metadata.cu_q,
                safe_gate=True,
                lower_bound=spec.gate_lower_bound,
            )
            self.ssm[metadata.request_indices * self.candidate_slots] = final.to(self.ssm.dtype)
            if self.accepted_index is not None:
                self.accepted_index[metadata.request_indices] = 0
        else:
            from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops.kda_decode import fused_recurrent_kda

            extra = {}
            conv_update = causal_conv1d_update
            if self.phase == "verify":
                from lightllm.common.basemodel.triton_kernel.linear_att.causal_conv1d_mtp import (
                    causal_conv1d_update as causal_conv1d_update_mtp,
                )

                conv_update = causal_conv1d_update_mtp
                extra = dict(
                    mtp_step=self.candidate_slots - 1,
                    num_accepted_tokens=metadata.accepted_tokens,
                    query_start_loc=metadata.cu_q,
                )
            mixed = conv_update(
                inputs.qkv,
                self.conv,
                params.conv_weight,
                bias=None,
                activation="silu",
                conv_state_indices=metadata.request_indices,
                **extra,
            )
            shape = (1, count) if self.phase == "verify" else (count, 1)
            q, k, v = (t.reshape(*shape, spec.heads, spec.head_dim) for t in mixed.split(width, -1))
            output, _ = fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                raw_gate=inputs.gate.reshape(*shape, width),
                raw_beta=inputs.beta.reshape(*shape, spec.heads),
                a_log=params.a_log,
                gate_bias=params.dt_bias,
                initial_state=self.ssm,
                lower_bound=spec.gate_lower_bound,
                inplace_final_state=True,
                ssm_state_indices=metadata.ssm_indices,
                cu_seqlens=metadata.cu_q if self.phase == "verify" else None,
                num_accepted_tokens=metadata.accepted_tokens,
            )
        return output


class KimiDeltaAttentionBackend:
    supports_auxiliary_states = False
    implementation = "triton"

    def __init__(self, phase):
        if phase not in ("prefill", "decode", "verify"):
            raise ValueError("unsupported KDA execution phase")
        self.phase = phase
        self.name = f"triton.kda.{phase}"

    def probe(self, spec, fmt, device_caps, limits, *, auxiliary_formats=None):
        if not isinstance(spec, KimiDeltaAttentionSpec):
            return ProbeResult(reason="requires Kimi Delta Attention semantics")
        if not device_caps.is_nvidia or spec.dtype not in (torch.float16, torch.bfloat16):
            return ProbeResult(reason="KDA requires CUDA and FP16/BF16 inputs")
        if find_spec("sgl_kernel") is None:
            return ProbeResult(reason="KDA convolution requires sgl_kernel")
        if spec.head_dim not in (32, 64, 128) or spec.conv_size not in (2, 3, 4):
            return ProbeResult(reason="unsupported KDA head or convolution dimensions")
        if self.phase != "prefill" and spec.head_dim != 128:
            return ProbeResult(reason="KDA recurrent decode requires 128-dimensional heads")
        if (
            fmt.encoding != "conv_ssm"
            or fmt.layout != "request_conv_ssm"
            or fmt.version != 1
            or fmt.fields != (("conv", "ssm", "accepted_index") if fmt.candidate_slots > 1 else ("conv", "ssm"))
            or fmt.dtype != spec.dtype
            or fmt.field_dtypes[:2] not in ((spec.dtype, torch.bfloat16), (spec.dtype, torch.float32))
            or fmt.field_dtypes[2:] != ((torch.int32,) if fmt.candidate_slots > 1 else ())
        ):
            return ProbeResult(reason="KDA requires explicit Conv and BF16/FP32 SSM fields")
        if fmt.candidate_slots != limits.max_verify_tokens:
            return ProbeResult(reason="KDA state candidate count differs from the execution capacity")
        if (self.phase == "verify" and fmt.candidate_slots == 1) or (
            self.phase == "decode" and fmt.candidate_slots != 1
        ):
            return ProbeResult(reason="KDA verification requires distinct state candidate slots")
        return ProbeResult(offers=(AttentionOffer(self.name, self.phase, fmt, execution_limits=limits),))

    def bind(self, offer, view, params, workspace, *, auxiliary_views=None):
        spec, conv, ssm = params.spec, *view.fields[:2]
        slots = view.format.candidate_slots
        if offer.backend != self.name or offer.phase != self.phase or offer.format != view.format:
            raise ValueError("offer does not match KDA binding")
        if (
            conv.ndim != 3
            or conv.shape[1:] != (3 * spec.heads * spec.head_dim, spec.conv_size - 1 + slots - 1)
            or ssm.shape != (conv.shape[0] * slots, spec.heads, spec.head_dim, spec.head_dim)
            or (conv.dtype, ssm.dtype) != view.format.field_dtypes[:2]
            or conv.device != ssm.device
            or not conv.is_contiguous()
            or not ssm.is_contiguous()
        ):
            raise ValueError("KDA state views disagree with the declared format")
        if slots > 1:
            cursor = view.fields[2]
            if (
                cursor.shape != (conv.shape[0],)
                or cursor.dtype != torch.int32
                or cursor.device != conv.device
                or not cursor.is_contiguous()
            ):
                raise ValueError("KDA accepted-state cursor disagrees with the candidate storage")
        for tensor, shape, dtype in (
            (params.conv_weight, (3 * spec.heads * spec.head_dim, spec.conv_size), spec.dtype),
            (params.a_log, (spec.heads,), torch.float32),
            (params.dt_bias, (spec.heads * spec.head_dim,), torch.float32),
        ):
            if tensor.shape != shape or tensor.dtype != dtype or tensor.device != conv.device:
                raise ValueError("KDA static parameters disagree with the bound semantics")
        return BoundKimiDeltaAttention(offer, view, params)
