# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from lightllm.models.glm5_next_mtp.triton_kernel.zero_position_embedding import (
    zero_position_embedding_,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_zero_position_embedding_only_changes_position_zero_rows():
    embeddings = torch.arange(15, device="cuda", dtype=torch.bfloat16).view(3, 5)
    original = embeddings.clone()
    position_ids = torch.tensor([0, 1, 0], device="cuda", dtype=torch.int64)

    zero_position_embedding_(embeddings, position_ids)

    torch.testing.assert_close(embeddings[0], torch.zeros_like(embeddings[0]), rtol=0, atol=0)
    torch.testing.assert_close(embeddings[1], original[1], rtol=0, atol=0)
    torch.testing.assert_close(embeddings[2], torch.zeros_like(embeddings[2]), rtol=0, atol=0)
