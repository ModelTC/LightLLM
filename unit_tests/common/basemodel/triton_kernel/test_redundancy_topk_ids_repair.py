import torch
import pytest
from lightllm.common.basemodel.triton_kernel.redundancy_topk_ids_repair import redundancy_topk_ids_repair
from lightllm.common.basemodel.triton_kernel.redundancy_topk_ids_repair import expert_id_counter
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def test_redundancy_topk_ids_repair():
    ep_expert_num = 4
    global_rank = 0
    redundancy_expert_num = 1
    topk_ids = torch.tensor(
        [
            [0, 1, 2, 3],
            [7, 9, 10, 11],
            [1, 3, 5, 7],
        ],
        dtype=torch.int64,
        device="cuda",
    )

    redundancy_expert_ids = torch.tensor(
        [
            0,
        ],
        dtype=torch.int64,
        device="cuda",
    )

    expert_id_counter = torch.zeros(12, dtype=torch.int64, device="cuda")

    redundancy_topk_ids_repair(
        topk_ids=topk_ids,
        redundancy_expert_ids=redundancy_expert_ids,
        ep_expert_num=ep_expert_num,
        global_rank=global_rank,
        expert_counter=expert_id_counter,
        enable_counter=True,
    )

    ans_topk_ids = torch.tensor(
        [
            [0, 1, 2, 3],
            [7, 9, 10, 11],
            [1, 3, 5, 7],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    ans_topk_ids = (ans_topk_ids // ep_expert_num) * redundancy_expert_num + ans_topk_ids
    new_redundancy_expert_ids = (redundancy_expert_ids // ep_expert_num) * redundancy_expert_num + redundancy_expert_ids
    ans_topk_ids[ans_topk_ids == new_redundancy_expert_ids[0]] = (
        (ep_expert_num + redundancy_expert_num) * global_rank + ep_expert_num + 0
    )

    assert torch.equal(topk_ids, ans_topk_ids)
    assert torch.equal(
        expert_id_counter, torch.tensor([1, 2, 1, 2, 0, 1, 0, 2, 0, 1, 1, 1], dtype=torch.int64, device="cuda")
    )

    ep_expert_num = 4
    global_rank = 1
    redundancy_expert_num = 1
    topk_ids = torch.tensor(
        [
            [0, 1, 2, 3],
            [7, 9, 10, 11],
            [1, 3, 5, 7],
        ],
        dtype=torch.int64,
        device="cuda",
    )

    redundancy_expert_ids = torch.tensor(
        [
            5,
        ],
        dtype=torch.int64,
        device="cuda",
    )
    redundancy_topk_ids_repair(
        topk_ids=topk_ids,
        redundancy_expert_ids=redundancy_expert_ids,
        ep_expert_num=ep_expert_num,
        global_rank=global_rank,
    )

    ans_topk_ids = torch.tensor(
        [
            [0, 1, 2, 3],
            [7, 9, 10, 11],
            [1, 3, 5, 7],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    ans_topk_ids = (ans_topk_ids // ep_expert_num) * redundancy_expert_num + ans_topk_ids
    new_redundancy_expert_ids = (redundancy_expert_ids // ep_expert_num) * redundancy_expert_num + redundancy_expert_ids
    ans_topk_ids[ans_topk_ids == new_redundancy_expert_ids[0]] = (
        (ep_expert_num + redundancy_expert_num) * global_rank + ep_expert_num + 0
    )

    assert torch.equal(topk_ids, ans_topk_ids)


def test_expert_id_counter():
    token_num = 256
    tok_ids = torch.randint(
        low=0,
        high=12,
        size=(token_num, 8),
        dtype=torch.int64,
        device="cuda",
    )
    expert_counter = torch.zeros(12, dtype=torch.int64, device="cuda")
    expert_id_counter(topk_ids=tok_ids, expert_counter=expert_counter)

    ans_expert_counter = torch.zeros(12, dtype=torch.int64, device="cuda")
    ids, counts = torch.unique(tok_ids.view(-1), return_counts=True)
    ans_expert_counter[ids] = counts

    assert torch.equal(expert_counter, ans_expert_counter)

    # test speed
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(100):
            tok_ids = torch.randint(
                low=0,
                high=12,
                size=(token_num, 8),
                dtype=torch.int64,
                device="cuda",
            )
            expert_counter = torch.zeros(12, dtype=torch.int64, device="cuda")
            expert_id_counter(topk_ids=tok_ids, expert_counter=expert_counter)
    graph.replay()

    start_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    graph.replay()
    end_event = torch.cuda.Event(enable_timing=True)
    end_event.record()
    torch.cuda.synchronize()
    logger.info(f"expert_id_counter time cost: {start_event.elapsed_time(end_event)} ms")


if __name__ == "__main__":
    pytest.main()
