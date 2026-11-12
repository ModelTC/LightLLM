from dataclasses import dataclass
from typing import Mapping, Optional

import torch
import torch.nn.functional as F
import triton

from lightllm.common.gated_residual.kernels import grouped_norm
from .ple_kernels import _gate, _conv, _conv_history


@dataclass(frozen=True)
class PLELayer:
    key_weight: torch.Tensor
    value_weight: torch.Tensor
    norm_key: torch.Tensor
    norm_query: torch.Tensor
    norm_conv: torch.Tensor
    conv_weight: torch.Tensor
    branches: int
    dilation: int
    eps: float = 1e-6

    def __post_init__(self):
        width = self.norm_key.numel()
        if self.branches < 1 or not width or width % self.branches or self.dilation < 1 or self.eps <= 0:
            raise ValueError("PLE requires positive dimensions, dilation and epsilon")
        if self.key_weight.ndim != 2 or self.key_weight.shape[1] == 0:
            raise ValueError("PLE key projection must be a nonempty matrix")
        if self.key_weight.shape[0] != width or self.value_weight.shape != (self.hidden_size, self.key_weight.shape[1]):
            raise ValueError("PLE projection dimensions do not match residual/embedding widths")
        if any(weight.shape != (width,) for weight in (self.norm_key, self.norm_query, self.norm_conv)):
            raise ValueError("PLE normalization vectors must match the residual width")
        if self.conv_weight.ndim != 2 or self.conv_weight.shape[0] != width or not self.conv_weight.shape[1]:
            raise ValueError("PLE convolution weight must have shape (residual width, kernel size)")
        for weight in (
            self.key_weight,
            self.value_weight,
            self.norm_key,
            self.norm_query,
            self.norm_conv,
            self.conv_weight,
        ):
            if not weight.is_cuda or not weight.is_contiguous():
                raise ValueError("PLE weights must be contiguous CUDA tensors")
            if weight.device != self.key_weight.device or weight.dtype != self.key_weight.dtype:
                raise ValueError("PLE weights must share a device and dtype")
        if self.key_weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("PLE weights require a floating point dtype")

    @property
    def hidden_size(self) -> int:
        return self.norm_key.numel() // self.branches

    @property
    def context_len(self) -> int:
        return (self.conv_weight.shape[1] - 1) * self.dilation

    @classmethod
    def from_checkpoint(
        cls, weights: Mapping[str, torch.Tensor], prefix: str, branches: int, dilation: int, eps: float = 1e-6
    ) -> "PLELayer":
        conv = weights[f"{prefix}.conv1d.weight"]
        if conv.ndim != 3 or conv.shape[1] != 1:
            raise ValueError("Checkpoint PLE convolution must be depthwise with shape (width, 1, kernel)")
        return cls(
            weights[f"{prefix}.key_proj.weight"],
            weights[f"{prefix}.value_proj.weight"],
            weights[f"{prefix}.norm_key.weight"],
            weights[f"{prefix}.norm_query.weight"],
            weights[f"{prefix}.norm_conv.weight"],
            conv.squeeze(1),
            branches,
            dilation,
            eps,
        )

    def empty_history(self, requests: int) -> torch.Tensor:
        return torch.zeros(
            (requests, self.context_len, self.norm_key.numel()),
            dtype=self.key_weight.dtype,
            device=self.key_weight.device,
        )

    def _check_tensor(self, tensor: torch.Tensor):
        if (
            tensor.device != self.key_weight.device
            or tensor.dtype != self.key_weight.dtype
            or not tensor.is_contiguous()
        ):
            raise ValueError("PLE inputs must be contiguous and share the weights' device and dtype")

    def _validate_state(self, values: torch.Tensor, starts: torch.Tensor, history: torch.Tensor):
        width = self.norm_key.numel()
        self._check_tensor(values)
        self._check_tensor(history)
        if values.ndim != 2 or values.shape[1] != width:
            raise ValueError("PLE values must have shape (packed tokens, residual width)")
        if starts.ndim != 1 or starts.numel() == 0 or starts.dtype != torch.int32:
            raise ValueError("PLE starts must be an int32 cumulative-length vector")
        if starts.device != values.device or not starts.is_contiguous():
            raise ValueError("PLE starts must be contiguous and on the input device")
        requests = starts.numel() - 1
        if history.shape != (requests, self.context_len, width):
            raise ValueError("PLE history must have shape (requests, context length, residual width)")
        if requests == 0 and values.shape[0]:
            raise ValueError("Nonempty PLE input requires at least one request")

    @torch.no_grad()
    def forward(
        self,
        residual: torch.Tensor,
        embeddings: torch.Tensor,
        starts: torch.Tensor,
        history: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        self._validate_state(residual, starts, history)
        self._check_tensor(embeddings)
        tokens = residual.shape[0]
        if embeddings.shape != (tokens, self.key_weight.shape[1]):
            raise ValueError("PLE embeddings must have shape (packed tokens, embedding width)")
        if mask is not None:
            if (
                mask.shape != (tokens,)
                or mask.dtype != torch.bool
                or mask.device != residual.device
                or not mask.is_contiguous()
            ):
                raise ValueError("PLE mask must be a contiguous boolean vector on the input device")
        key = grouped_norm(F.linear(embeddings, self.key_weight), self.norm_key, self.branches, self.eps)
        query = grouped_norm(residual, self.norm_query, self.branches, self.eps)
        value = F.linear(embeddings, self.value_weight)
        gated = torch.empty_like(residual)
        if tokens:
            _gate[(tokens, self.branches)](
                key,
                query,
                value,
                mask,
                gated,
                self.hidden_size,
                self.branches,
                mask is not None,
                triton.next_power_of_2(self.hidden_size),
                enable_fp_fusion=False,
            )

        conv_inputs = grouped_norm(gated, self.norm_conv, self.branches, self.eps)
        output = torch.empty_like(residual)
        if tokens:
            requests = starts.numel() - 1
            _conv[(tokens, triton.cdiv(residual.shape[1], 256))](
                conv_inputs,
                gated,
                self.conv_weight,
                starts,
                history,
                output,
                residual.shape[1],
                self.conv_weight.shape[1],
                self.dilation,
                requests,
                requests.bit_length(),
                256,
                enable_fp_fusion=False,
            )
        return output, conv_inputs

    @torch.no_grad()
    def history_after(
        self,
        conv_inputs: torch.Tensor,
        starts: torch.Tensor,
        history: torch.Tensor,
        accepted_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._validate_state(conv_inputs, starts, history)
        requests = starts.numel() - 1
        if accepted_lengths is not None:
            if accepted_lengths.shape != (requests,) or accepted_lengths.dtype != torch.int32:
                raise ValueError("PLE accepted lengths must be an int32 vector with one count per request")
            if accepted_lengths.device != conv_inputs.device or not accepted_lengths.is_contiguous():
                raise ValueError("PLE accepted lengths must be contiguous and on the input device")
        result = torch.empty_like(history)
        if requests and self.context_len:
            _conv_history[(requests, triton.cdiv(self.context_len * self.norm_key.numel(), 256))](
                conv_inputs,
                starts,
                history,
                accepted_lengths,
                result,
                self.norm_key.numel(),
                self.context_len,
                accepted_lengths is not None,
                256,
            )
        return result
