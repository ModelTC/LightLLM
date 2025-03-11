import pytest
import torch
import random
from lightllm.common.kv_trans_kernel.kv_trans_v2 import kv_trans_v2


@pytest.mark.parametrize(
    "token_num",
    [token_num for token_num in range(5, 10)],
)
def test_kv_trans_v2(token_num):
    dp_size_in_node = 8
    head_num = 2
    head_dim = 512
    kv_buffer_token_num = 512
    mems = []
    for _ in range(dp_size_in_node):
        mems.append(torch.randn((kv_buffer_token_num, head_num, head_dim), dtype=torch.float16, device="cuda"))
    input_mems = torch.tensor([e.data_ptr() for e in mems], dtype=torch.uint64, device="cuda")
    input_idx = [random.randint(0, kv_buffer_token_num - 1) for _ in range(token_num)]
    input_idx = torch.tensor(input_idx, dtype=torch.int32, device="cuda")
    input_dp_idx = [random.randint(0, dp_size_in_node - 1) for _ in range(token_num)]
    input_dp_idx = torch.tensor(input_dp_idx, dtype=torch.int32, device="cuda")

    true_output = torch.zeros((token_num, head_num, head_dim), dtype=torch.float16, device="cuda")
    test_output = torch.zeros((token_num, head_num, head_dim), dtype=torch.float16, device="cuda")
    output_idx = torch.arange(0, token_num, 1, dtype=torch.int32, device="cuda")

    kv_trans_v2(input_mems, input_idx, input_dp_idx, test_output, output_idx, dp_size_in_node)

    for dest_token_index, token_index, dp_index in zip(
        list(range(token_num)), input_idx.cpu().numpy(), input_dp_idx.cpu().numpy()
    ):
        true_output[dest_token_index, :, :] = mems[dp_index][token_index]

    assert torch.equal(true_output, test_output)
    return


if __name__ == "__main__":
    pytest.main()
