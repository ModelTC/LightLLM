# SPDX-License-Identifier: Apache-2.0

"""Expansion and mean contraction of flattened residual streams."""

from __future__ import annotations

import torch


def hc_expand(x: torch.Tensor, streams: int) -> torch.Tensor:
    """Copy each token to S residual streams.

    x: [T, H]. Return [T, S * H] with the same dtype and device.
    Each stream has the same embedding.
    """

    assert x.ndim == 2
    return x.unsqueeze(1).expand(-1, streams, -1).reshape(x.shape[0], streams * x.shape[1])


def hc_contract(x: torch.Tensor, streams: int) -> torch.Tensor:
    """Take the mean of S residual streams.

    x: [T, S * H]. Return [T, H] with the same dtype and device.
    Models with learned gates use hc_head instead.
    """

    assert x.ndim == 2 and x.shape[-1] % streams == 0
    return x.view(x.shape[0], streams, x.shape[1] // streams).mean(dim=1)
