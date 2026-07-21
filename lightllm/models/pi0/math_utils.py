import math

import torch


def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float = 0.004,
    max_period: float = 4.0,
) -> torch.Tensor:
    """OpenPI's float64 sinusoidal timestep embedding."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("time must have shape [batch]")
    compute_dtype = torch.float32 if time.device.type == "mps" else torch.float64
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=compute_dtype, device=time.device)
    period = min_period * (max_period / min_period) ** fraction
    radians = (2.0 * math.pi / period)[None, :] * time.to(compute_dtype)[:, None]
    return torch.cat([torch.sin(radians), torch.cos(radians)], dim=-1)


def denoise_schedule(num_steps: int, *, device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
    timesteps = 1.0 + torch.arange(num_steps, dtype=torch.float32, device=device) * dt
    return timesteps, dt
