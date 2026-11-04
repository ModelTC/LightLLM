"""Content identities for static KV scales, independent of paths and TP shards."""

import hashlib
import json

import torch


def kv_scale_identity(scales: torch.Tensor, scale_mode: str) -> str:
    scales = scales.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not scales.numel() or not bool((torch.isfinite(scales) & (scales > 0)).all()):
        raise ValueError("KV calibration scales must be finite and positive")
    # Implicit unit scales and an explicit all-ones file describe the same bytes.
    if bool((scales == 1).all()):
        payload = b"unit"
    else:
        payload = json.dumps(list(scales.shape)).encode() + scales.numpy().astype("<f4").tobytes()
    digest = hashlib.sha256(scale_mode.encode() + b"\0" + payload).hexdigest()
    return f"kv-scale-v1:{digest}"


def calibration_table_identity(calibration, scale_mode: str) -> str:
    """Identify the startup KV table; Q scales are execution parameters, not KV format."""
    if calibration is None:
        if scale_mode != "per_tensor":
            raise ValueError("per-head FP8 requires a KV calibration file")
        scales = torch.ones((1, 2), dtype=torch.float32)
    else:
        if calibration.get("quant_type", scale_mode) != scale_mode:
            raise ValueError("calibration quantization mode does not match the State")
        scales = torch.tensor(calibration["scales"], dtype=torch.float32)
        heads = calibration["num_head"] if scale_mode == "per_head" else 1
        if tuple(calibration["scales_shape"]) != scales.shape or scales.shape != (calibration["num_layers"], 2 * heads):
            raise ValueError("invalid calibration scale shape")
    return kv_scale_identity(scales, scale_mode)
