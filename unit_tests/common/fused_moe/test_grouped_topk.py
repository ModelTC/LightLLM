import torch
import time
import pytest
import numpy as np
from lightllm.common.fused_moe.grouped_topk import triton_grouped_topk
from lightllm.common.fused_moe.topk_select import biased_grouped_topk as grouped_topk
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

seed = 42
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@pytest.mark.parametrize(
    "expert_num, topk_group, group_num, topk_num, scoring_func, token_num",
    [
        (*a, b, c)
        for a in [(256, 4, 8, 8), (160, 3, 8, 6)]
        for b in [
            "sigmoid",
        ]
        for c in [1, 8, 256, 1024, 2048, 4096, 8192]
    ],
)
def test_grouped_topk(expert_num, topk_group, group_num, topk_num, scoring_func, token_num):
    print("test", expert_num, topk_group, group_num, topk_num, scoring_func, token_num)
    dtype = torch.float32
    hidden_state = torch.randn((token_num, 1), dtype=dtype, device="cuda")
    gating_output = torch.randn((token_num, expert_num), dtype=dtype, device="cuda") * 10
    correction_bias = torch.randn((expert_num,), dtype=dtype, device="cuda")
    correction_bias[correction_bias <= 0.0] = 0.0

    old_topk_weights, old_topk_ids = grouped_topk(
        hidden_state,
        gating_output=gating_output,
        correction_bias=correction_bias,
        topk=topk_num,
        renormalize=True,
        num_expert_group=group_num,
        topk_group=topk_group,
        scoring_func=scoring_func,
    )

    new_topk_weights, new_topk_ids = triton_grouped_topk(
        None,
        gating_output=gating_output,
        correction_bias=correction_bias,
        topk=topk_num,
        renormalize=True,
        num_expert_group=group_num,
        topk_group=topk_group,
        scoring_func=scoring_func,
    )

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(60):
        old_topk_weights, old_topk_ids = grouped_topk(
            hidden_state,
            gating_output=gating_output,
            correction_bias=correction_bias,
            topk=topk_num,
            renormalize=True,
            num_expert_group=group_num,
            topk_group=topk_group,
            scoring_func=scoring_func,
        )
    torch.cuda.synchronize()
    print(f"old cost time {time.time() - start} s")

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(60):
        new_topk_weights, new_topk_ids = triton_grouped_topk(
            None,
            gating_output=gating_output,
            correction_bias=correction_bias,
            topk=topk_num,
            renormalize=True,
            num_expert_group=group_num,
            topk_group=topk_group,
            scoring_func=scoring_func,
        )
    torch.cuda.synchronize()
    print(f"new cost time {time.time() - start} s")

    assert torch.equal(torch.sort(old_topk_ids, dim=1)[0], torch.sort(new_topk_ids, dim=1)[0])
    assert torch.allclose(
        torch.sort(old_topk_weights, dim=1)[0], torch.sort(new_topk_weights, dim=1)[0], atol=1e-3, rtol=1e-1
    ), f"max delta {torch.max(torch.sort(old_topk_weights, dim=1)[0] - torch.sort(new_topk_weights, dim=1)[0])}"
    return


if __name__ == "__main__":
    pytest.main()
