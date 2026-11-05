"""Linear compute binds native weights independently of checkpoint loading."""

from dataclasses import dataclass
from functools import partial
from typing import Callable
from importlib import import_module
from importlib.util import find_spec

import torch

from lightllm.common.execution.contracts.provider import LinearOffer, ProbeResult


@dataclass(frozen=True)
class LinearConfig:
    encoding: str
    execute: Callable
    activation_group_size: int | None = None
    output_block_size: int = 1
    allocates_output: bool = False
    per_output_scale: bool = False
    min_capability: tuple[int, int] = (8, 9)
    alignment: int = 1
    dependency: str | None = None


@dataclass(frozen=True)
class LinearConfig:
    encoding: str
    execute: Callable
    activation_group_size: int | None = None
    output_block_size: int = 1
    allocates_output: bool = False
    per_output_scale: bool = False
    min_capability: tuple[int, int] = (8, 9)
    alignment: int = 1
    dependency: str | None = None


class BoundLinear:
    def __init__(self, offer, weight, scale, bias, allocate, config):
        self.name, self.spec, self.format = offer.backend, offer.spec, offer.format
        self.weight, self.scale, self.bias = weight, scale, bias
        self.allocate = allocate
        self.matrix = weight if self.spec.batches is not None else weight.t()
        self.config = config
        self.allocates_output = config.allocates_output
        self._execute = partial(config.execute, self)

    def run(self, input_tensor, out=None, use_custom_tensor_mananger=True):
        spec, x = self.spec, input_tensor
        if (spec.batches is None and x.ndim != 2) or (
            spec.batches is not None and (x.ndim != 3 or x.shape[0] != spec.batches)
        ):
            raise ValueError("linear input disagrees with its bound batch layout")
        if x.dtype != spec.dtype or x.device != self.weight.device or x.shape[-1] != spec.in_features:
            raise ValueError("linear input disagrees with its bound dimensions, dtype or device")
        shape = (*x.shape[:-1], spec.out_features)
        allocate = self.allocate if use_custom_tensor_mananger else torch.empty
        if out is None and (not self.allocates_output or x.numel() == 0):
            out = allocate(shape, dtype=x.dtype, device=x.device)
        elif out is not None and (out.shape != shape or out.device != x.device):
            raise ValueError("linear output disagrees with its bound shape or device")
        if (
            out is not None
            and out.dtype != x.dtype
            and (self.config.encoding != "plain" or self.bias is not None or spec.batches is not None)
        ):
            raise ValueError("this linear implementation requires output and input dtypes to match")
        if x.numel() == 0:
            return out
        return self._execute(x, out, allocate)

    def _plain(self, x, out, allocate):
        if self.spec.batches is not None:
            return torch.bmm(x, self.matrix, out=out)
        if self.bias is not None:
            return torch.addmm(self.bias, x, self.matrix, out=out)
        if out.dtype != x.dtype:
            return torch.mm(x, self.matrix, out=out, out_dtype=out.dtype)
        return torch.mm(x, self.matrix, out=out)

    def _quantize(self, x, allocate, group_size):
        from lightllm.common.basemodel.triton_kernel.quantization.fp8act_quant_kernel import (
            lightllm_per_token_group_quant_fp8,
        )

        q = allocate(x.shape, dtype=self.weight.dtype, device=x.device)
        scale = allocate((x.shape[0], x.shape[1] // group_size), dtype=torch.float32, device=x.device)
        lightllm_per_token_group_quant_fp8(x, group_size, q, scale)
        return q, scale

    def _fp8_tensor(self, x, out, allocate):
        from lightllm.common.basemodel.triton_kernel.quantization.scaled_mm_per_token_kernel import (
            fp8_scaled_mm_per_token,
        )

        q, scale = self._quantize(x, allocate, self.spec.in_features)
        return fp8_scaled_mm_per_token(q, self.matrix, scale, self.scale, x.dtype, out)

    def _fp8_sgl(self, x, out, allocate):
        from sgl_kernel import fp8_scaled_mm

        q, scale = self._quantize(x, allocate, self.spec.in_features)
        result = fp8_scaled_mm(q, self.matrix, scale, self.scale, x.dtype)
        return out.copy_(result) if out is not None else result

    def _fp8_group(self, x, out, allocate):
        from lightllm.common.basemodel.triton_kernel.quantization.scaled_mm_per_token_group_quant_kernel import (
            scaled_mm_act_per_group_w_perchannel,
        )

        group = self.config.activation_group_size
        q, scale = self._quantize(x, allocate, group)
        return scaled_mm_act_per_group_w_perchannel(q, self.matrix, scale, self.scale, group, out)

    def _fp8_block(self, x, out, allocate):
        from lightllm.common.basemodel.triton_kernel.quantization.fp8act_quant_kernel import per_token_group_quant_fp8
        from lightllm.common.basemodel.triton_kernel.quantization.fp8w8a8_block_gemm_kernel import (
            w8a8_block_fp8_matmul,
        )

        q, scale = per_token_group_quant_fp8(x, 128, dtype=self.weight.dtype, alloc_func=allocate)
        w8a8_block_fp8_matmul(q, self.matrix, scale, self.scale.t(), out, (128, 128), dtype=x.dtype)
        return out


class LinearBackend:
    implementations = {
        "torch.linear": LinearConfig("plain", BoundLinear._plain),
        "triton.linear.fp8_tensor": LinearConfig("fp8_per_tensor", BoundLinear._fp8_tensor),
        "sgl.linear.fp8_tensor": LinearConfig(
            "fp8_per_tensor",
            BoundLinear._fp8_sgl,
            allocates_output=True,
            per_output_scale=True,
            min_capability=(9, 0),
            alignment=16,
            dependency="sgl_kernel",
        ),
        "triton.linear.fp8_channel_g64": LinearConfig(
            "fp8_per_channel",
            BoundLinear._fp8_group,
            activation_group_size=64,
            per_output_scale=True,
        ),
        "triton.linear.fp8_channel_g128": LinearConfig(
            "fp8_per_channel",
            BoundLinear._fp8_group,
            activation_group_size=128,
            per_output_scale=True,
        ),
        "triton.linear.fp8_block128": LinearConfig(
            "fp8_block128",
            BoundLinear._fp8_block,
            activation_group_size=128,
            output_block_size=128,
        ),
    }

    def __init__(self, name):
        self.name = name
        self.config = self.implementations[name]

    def probe(self, spec, fmt, caps):
        quantized = self.config.encoding != "plain"
        layout = "batch_input_output" if spec.batches is not None else "output_input"
        fields = ("weight", "scale") if quantized else ("weight",)
        if (
            fmt.encoding != self.config.encoding
            or fmt.layout != layout
            or fmt.fields != fields
            or fmt.version != 1
            or fmt.dtype != (torch.float8_e4m3fn if quantized else spec.dtype)
            or fmt.group_size != (128 if fmt.encoding == "fp8_block128" else None)
        ):
            return ProbeResult(reason="linear format is incompatible with the selected implementation")
        if spec.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return ProbeResult(reason="linear requires FP16, BF16 or FP32 inputs")
        if quantized:
            if spec.batches is not None or not caps.is_nvidia or spec.dtype == torch.float32:
                return ProbeResult(reason="FP8 linear requires CUDA FP16/BF16 matrices")
            if caps.capability is None or caps.capability < self.config.min_capability:
                return ProbeResult(reason="FP8 linear requires SM89 or newer")
            if fmt.scale_dtype != torch.float32:
                return ProbeResult(reason="FP8 linear requires FP32 scales")
            group = self.config.activation_group_size
            if group is not None and spec.in_features % group:
                return ProbeResult(reason="linear input dimension must contain complete quantization groups")
            if spec.out_features % self.config.output_block_size:
                return ProbeResult(reason="block128 weight storage requires complete output blocks")
            if self.config.dependency is not None:
                if spec.in_features % self.config.alignment or spec.out_features % self.config.alignment:
                    return ProbeResult(reason="SGL FP8 linear requires Hopper and dimensions aligned to 16")
                if find_spec("sgl_kernel") is None:
                    return ProbeResult(reason="SGL linear requires sgl_kernel")
                if not callable(getattr(import_module("sgl_kernel"), "fp8_scaled_mm", None)):
                    return ProbeResult(reason="installed SGL kernel does not provide scaled FP8 linear")
        return ProbeResult(offers=(LinearOffer(self.name, spec, fmt),))

    def bind(self, offer, weight, scale=None, bias=None, *, allocate=torch.empty):
        spec, fmt = offer.spec, offer.format
        shape = (spec.out_features, spec.in_features)
        if spec.batches is not None:
            shape = (spec.batches, spec.in_features, spec.out_features)
        if offer.backend != self.name or weight.shape != shape or weight.dtype != fmt.dtype:
            raise ValueError("linear weight view does not match its selected offer")
        if bias is not None and (
            self.config.encoding != "plain"
            or spec.batches is not None
            or bias.shape != (spec.out_features,)
            or bias.dtype != spec.dtype
            or bias.device != weight.device
        ):
            raise ValueError("linear bias is incompatible with its selected implementation")
        if self.config.encoding == "plain":
            if scale is not None:
                raise ValueError("plain linear cannot consume quantization scales")
        else:
            if scale is None or scale.dtype != torch.float32 or scale.device != weight.device:
                raise ValueError("FP8 linear requires native FP32 scale views on the weight device")
            expected = (spec.out_features // 128, spec.in_features // 128)
            if fmt.encoding == "fp8_block128":
                valid = scale.shape == expected
            elif self.config.per_output_scale:
                valid = scale.shape == (spec.out_features,)
            else:
                valid = scale.numel() == 1
            if not valid or not weight.is_contiguous() or not scale.is_contiguous():
                raise ValueError("FP8 weight or scale layout does not match its declared format")
        return BoundLinear(offer, weight, scale, bias, allocate, self.config)
