"""Gated DeltaNet kernels bound to request-local Conv/SSM views and static weights."""

from dataclasses import dataclass
from importlib.util import find_spec

import torch

from .specs import AttentionOffer, GatedDeltaNetSpec, ProbeResult


@dataclass(frozen=True)
class GDNStaticParams:
    spec: GatedDeltaNetSpec
    conv_weight: torch.Tensor
    conv_bias: torch.Tensor | None
    a_log: torch.Tensor
    dt_bias: torch.Tensor

    @classmethod
    def from_weights(cls, spec, weight):
        return cls(
            spec,
            weight.linear_conv1d.mm_param.weight,
            weight.linear_conv1d.bias,
            weight.linear_A_log.weight,
            weight.linear_dt_bias.weight,
        )


@dataclass
class GDNBatchMetadata:
    request_indices: torch.Tensor
    cu_q: torch.Tensor
    has_initial_state: torch.Tensor | None
    ssm_indices: torch.Tensor | None = None
    accepted_tokens: torch.Tensor | None = None

    def copy_for_graph(self, new):
        for name in ("request_indices", "cu_q", "has_initial_state", "ssm_indices", "accepted_tokens"):
            old, value = getattr(self, name), getattr(new, name)
            if old is None and value is None:
                continue
            if old is None or value is None or old.shape != value.shape or old.dtype != value.dtype:
                raise ValueError(f"GDN Graph metadata changed shape/dtype: {name}")
            if old.data_ptr() != value.data_ptr():
                old.copy_(value, non_blocking=True)


class BoundGatedDeltaNet:
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
        if self.phase == "prefill":
            if self.name.startswith("flashqla."):
                from flash_qla import chunk_gated_delta_rule
            else:
                from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops import chunk_gated_delta_rule
            self.prefill_kernel = chunk_gated_delta_rule

    def prepare_batch(self, batch):
        return batch.execution_metadata[self.metadata_key]

    def create_metadata(self, batch):
        if batch.batch_size > self.limits.max_batch_size or batch.max_q_len > self.limits.max_seq_len:
            raise ValueError("batch exceeds planned GDN capacity")
        if not 0 <= batch.microbatch_index < self.limits.microbatch_count:
            raise ValueError("batch exceeds planned execution lanes")
        if self.phase == "decode" and batch.max_q_len != 1:
            raise ValueError("GDN decode requires one token per request; verification needs its own plan")
        initial = batch.ready_lengths > 0 if self.phase == "prefill" else None
        requests, cu_q = batch.req_indices, batch.cu_q
        indices, accepted = requests, None
        if self.phase == "verify":
            from lightllm.common.basemodel.triton_kernel.linear_att.mtp_state_params import (
                build_dynamic_mtp_linear_att_state_params,
            )

            if batch.mtp_indices is None or batch.hold_request_id is None:
                raise ValueError("GDN verification requires candidate indices and a padding request identity")
            cu_q, requests, accepted = build_dynamic_mtp_linear_att_state_params(
                requests, batch.mtp_indices, self.accepted_index, batch.hold_request_id
            )
            indices = requests[:, None] * self.candidate_slots + torch.arange(
                self.candidate_slots, dtype=requests.dtype, device=requests.device
            )
        elif self.accepted_index is not None:
            # Snapshot the shared cursor before any layer normalizes its prefill state.
            accepted = self.accepted_index[requests] + 1
            indices = requests * self.candidate_slots + accepted - 1
        return GDNBatchMetadata(requests, cu_q, initial, indices, accepted)

    def run(self, inputs, metadata):
        spec, params = self.spec, self.params
        if inputs.ndim != 2 or inputs.shape[1] != spec.input_width or inputs.dtype != spec.dtype:
            raise ValueError("GDN input disagrees with the bound projection layout")
        qk_dim, value_dim = spec.k_heads * spec.key_dim, spec.v_heads * spec.value_dim
        mixed, z, b, a = inputs.split((spec.conv_channels, value_dim, spec.v_heads, spec.v_heads), dim=-1)
        z = z.view(-1, spec.v_heads, spec.value_dim)
        if self.phase == "verify":
            from lightllm.common.basemodel.triton_kernel.linear_att.causal_conv1d_mtp import causal_conv1d_update
            from lightllm.common.basemodel.triton_kernel.linear_att.mtp_fused_recurrent import (
                mtp_fused_recurrent_gated_delta_rule,
            )

            convolved = causal_conv1d_update(
                mixed,
                self.conv,
                params.conv_weight,
                mtp_step=self.candidate_slots - 1,
                bias=params.conv_bias,
                activation=spec.activation,
                conv_state_indices=metadata.request_indices,
                num_accepted_tokens=metadata.accepted_tokens,
                query_start_loc=metadata.cu_q,
            )
            q, k, v = convolved.split((qk_dim, qk_dim, value_dim), dim=-1)
            output, _ = mtp_fused_recurrent_gated_delta_rule(
                q=q.reshape(1, -1, spec.k_heads, spec.key_dim),
                k=k.reshape(1, -1, spec.k_heads, spec.key_dim),
                v=v.reshape(1, -1, spec.v_heads, spec.value_dim),
                initial_state=self.ssm,
                cu_seqlens=metadata.cu_q.long(),
                ssm_state_indices=metadata.ssm_indices,
                ssm_state_write_indices=metadata.ssm_indices,
                num_accepted_tokens=metadata.accepted_tokens,
                A_log=params.a_log,
                dt_bias=params.dt_bias,
                a_raw=a,
                b_raw=b,
                # 与 prefill 独立归一化、门值算子的中间精度一致。
                round_intermediates=True,
            )
        elif self.phase == "decode":
            from lightllm.common.basemodel.triton_kernel.linear_att.gdn_decode_pack import conv_pack_gdn_decode_inputs
            from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops import fused_recurrent_gated_delta_rule

            q, k, v, z, a, b = conv_pack_gdn_decode_inputs(
                mixed,
                z,
                a,
                b,
                self.conv,
                params.conv_weight,
                params.conv_bias,
                metadata.request_indices,
                spec.activation,
                spec.conv_size,
                spec.k_heads,
                spec.key_dim,
                spec.v_heads,
                spec.value_dim,
            )
            output, _ = fused_recurrent_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                initial_state=self.ssm,
                inplace_final_state=True,
                ssm_state_indices=metadata.request_indices,
                use_qk_l2norm_in_kernel=True,
                A_log=params.a_log,
                dt_bias=params.dt_bias,
                a_raw=a,
                b_raw=b,
            )
        else:
            from lightllm.common.basemodel.triton_kernel.linear_att.causal_conv1d import causal_conv1d_fn
            from lightllm.common.basemodel.triton_kernel.linear_att.fused_gdn_gating import fused_gdn_gating

            g, beta = fused_gdn_gating(params.a_log, a, b, params.dt_bias)
            conv = self.conv[..., : spec.conv_size - 1]
            if metadata.accepted_tokens is not None:
                offsets = (
                    metadata.accepted_tokens[:, None, None] - 1 + torch.arange(spec.conv_size - 1, device=inputs.device)
                )
                history = self.conv[metadata.request_indices].gather(-1, offsets.expand(-1, spec.conv_channels, -1))
                conv[metadata.request_indices] = history
            convolved = causal_conv1d_fn(
                mixed.transpose(0, 1),
                params.conv_weight,
                bias=params.conv_bias,
                query_start_loc=metadata.cu_q,
                cache_indices=metadata.request_indices,
                has_initial_state=metadata.has_initial_state,
                conv_states=conv,
                activation=spec.activation,
            ).transpose(0, 1)
            q, k, v = convolved.split((qk_dim, qk_dim, value_dim), dim=-1)
            initial = self.ssm[metadata.ssm_indices]
            initial = torch.where(metadata.has_initial_state[:, None, None, None], initial, 0)
            output, final = self.prefill_kernel(
                q=q.reshape(1, -1, spec.k_heads, spec.key_dim),
                k=k.reshape(1, -1, spec.k_heads, spec.key_dim),
                v=v.reshape(1, -1, spec.v_heads, spec.value_dim),
                g=g.unsqueeze(0),
                beta=beta.unsqueeze(0),
                initial_state=initial,
                output_final_state=True,
                cu_seqlens=metadata.cu_q,
                use_qk_l2norm_in_kernel=True,
            )
            self.ssm[metadata.request_indices * self.candidate_slots] = final.to(self.ssm.dtype)
            if self.accepted_index is not None:
                self.accepted_index[metadata.request_indices] = 0
        return output.reshape(-1, spec.v_heads, spec.value_dim), z


class GatedDeltaNetBackend:
    def __init__(self, implementation, phase):
        if implementation not in ("triton", "flashqla") or phase not in ("prefill", "decode", "verify"):
            raise ValueError("unsupported GDN backend or phase")
        self.implementation, self.phase = implementation, phase
        self.name = f"{implementation}.gdn.{phase}"

    def probe(self, attention_spec, format_spec, device_caps, limits):
        if not isinstance(attention_spec, GatedDeltaNetSpec):
            return ProbeResult(reason="requires Gated DeltaNet semantics")
        spec, fmt = attention_spec, format_spec
        if device_caps.device_type != "cuda":
            return ProbeResult(reason="GDN requires CUDA")
        if (
            spec.dtype not in (torch.float16, torch.bfloat16)
            or spec.key_dim not in (32, 64, 128, 256)
            or spec.value_dim not in (32, 64, 128, 256)
            or spec.conv_size not in (2, 3, 4)
            or spec.activation not in ("silu", "swish")
        ):
            return ProbeResult(reason="unsupported GDN dtype, head dimensions, convolution or activation")
        if (
            fmt.encoding != "conv_ssm"
            or fmt.layout != "request_conv_ssm"
            or fmt.version != 1
            or fmt.fields != (("conv", "ssm", "accepted_index") if fmt.candidate_slots > 1 else ("conv", "ssm"))
            or fmt.dtype != spec.dtype
            or fmt.field_dtypes[:2] not in ((spec.dtype, torch.bfloat16), (spec.dtype, torch.float32))
            or fmt.field_dtypes[2:] != ((torch.int32,) if fmt.candidate_slots > 1 else ())
        ):
            return ProbeResult(reason="GDN requires explicit Conv and BF16/FP32 SSM fields")
        if fmt.candidate_slots != limits.max_verify_tokens:
            return ProbeResult(reason="GDN state candidate count differs from the execution capacity")
        if (self.phase == "verify" and fmt.candidate_slots == 1) or (
            self.phase == "decode" and fmt.candidate_slots != 1
        ):
            return ProbeResult(reason="GDN verification requires distinct state candidate slots")
        if self.implementation == "flashqla":
            if self.phase != "prefill":
                return ProbeResult(reason="FlashQLA only provides GDN prefill")
            if find_spec("flash_qla") is None or device_caps.capability is None or device_caps.capability[0] < 9:
                return ProbeResult(reason="FlashQLA requires its library and Hopper or newer")
            if spec.key_dim != 128 or spec.value_dim != 128:
                return ProbeResult(reason="FlashQLA requires 128-dimensional key and value heads")
        return ProbeResult(offers=(AttentionOffer(self.name, self.phase, fmt, execution_limits=limits),))

    def bind(self, offer, state_view, static_params, workspace):
        if offer.backend != self.name or offer.phase != self.phase or offer.format != state_view.format:
            raise ValueError("offer does not match GDN binding")
        spec = static_params.spec
        conv, ssm = state_view.fields[:2]
        slots = state_view.format.candidate_slots
        if (
            conv.ndim != 3
            or conv.shape[1:] != (spec.conv_channels, spec.conv_size - 1 + slots - 1)
            or ssm.shape != (conv.shape[0] * slots, spec.v_heads, spec.key_dim, spec.value_dim)
            or (conv.dtype, ssm.dtype) != state_view.format.field_dtypes[:2]
            or conv.device != ssm.device
            or not conv.is_contiguous()
            or not ssm.is_contiguous()
        ):
            raise ValueError("GDN state views disagree with their declared format and semantics")
        if slots > 1:
            cursor = state_view.fields[2]
            if (
                cursor.shape != (conv.shape[0],)
                or cursor.dtype != torch.int32
                or cursor.device != conv.device
                or not cursor.is_contiguous()
            ):
                raise ValueError("GDN accepted-state cursor disagrees with the candidate storage")
        tensors = (
            (static_params.conv_weight, (spec.conv_channels, spec.conv_size), spec.dtype),
            (static_params.a_log, (spec.v_heads,), torch.float32),
            (static_params.dt_bias, (spec.v_heads,), torch.float32),
        )
        if static_params.conv_bias is not None:
            tensors += ((static_params.conv_bias, (spec.conv_channels,), spec.dtype),)
        for tensor, shape, dtype in tensors:
            if tensor.shape != shape or tensor.dtype != dtype or tensor.device != conv.device:
                raise ValueError("GDN static parameters disagree with the bound semantics")
        return BoundGatedDeltaNet(offer, state_view, static_params)
