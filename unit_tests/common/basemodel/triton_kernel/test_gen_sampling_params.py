import torch
import time
import pytest
from lightllm.common.basemodel.triton_kernel.gen_sampling_params import token_id_counter, gen_sampling_params
from lightllm.common.basemodel.triton_kernel.gen_sampling_params import update_req_to_token_id_counter
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def test_token_id_counter():
    test_prompt_ids = torch.randint(0, 100, size=(7853,), device="cuda")
    test_token_id_counter = torch.zeros((100,), dtype=torch.int32, device="cuda")
    token_id_counter(prompt_ids=test_prompt_ids, out_token_id_counter=test_token_id_counter)

    true_out = torch.bincount(test_prompt_ids, minlength=test_token_id_counter.shape[0])
    assert torch.equal(test_token_id_counter, true_out)

    test_prompt_ids = torch.randint(0, 50 * 10000, size=(128 * 1024,), device="cuda")
    test_token_id_counter = torch.zeros((50 * 10000,), device="cuda", dtype=torch.int32)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(100):
        token_id_counter(prompt_ids=test_prompt_ids, out_token_id_counter=test_token_id_counter)
    end_event.record()
    end_event.synchronize()
    logger.info(f"test_token_id_count cost time: {start_event.elapsed_time(end_event)} ms")


def test_gen_sampling_params():

    # Mocking ReqSamplingParamsManager for testing
    class MockReqSamplingParamsManager:
        def __init__(self, batch_size, vocab_size):
            self.req_to_presence_penalty = torch.rand((batch_size,), dtype=torch.float32, device="cuda")
            self.req_to_frequency_penalty = torch.rand((batch_size,), dtype=torch.float32, device="cuda")
            self.req_to_repetition_penalty = torch.rand((batch_size,), dtype=torch.float32, device="cuda")
            self.req_to_temperature = torch.rand((batch_size,), dtype=torch.float32, device="cuda")
            self.req_to_exponential_decay_length_penalty = torch.rand((batch_size,), dtype=torch.float32, device="cuda")

    batch_size = 1083
    vocab_size = 100
    req_sampling_params_manager = MockReqSamplingParamsManager(batch_size, vocab_size)
    b_req_idx = torch.arange(batch_size, device="cuda").flip(dims=[0])

    (
        b_presence_penalty,
        b_frequency_penalty,
        b_repetition_penalty,
        b_temperature,
        b_exponential_decay_length_penalty,
    ) = gen_sampling_params(b_req_idx, req_sampling_params_manager)

    assert torch.equal(b_presence_penalty, req_sampling_params_manager.req_to_presence_penalty[b_req_idx])
    assert torch.equal(b_frequency_penalty, req_sampling_params_manager.req_to_frequency_penalty[b_req_idx])
    assert torch.equal(b_repetition_penalty, req_sampling_params_manager.req_to_repetition_penalty[b_req_idx])
    assert torch.equal(b_temperature, req_sampling_params_manager.req_to_temperature[b_req_idx])
    assert torch.equal(
        b_exponential_decay_length_penalty,
        req_sampling_params_manager.req_to_exponential_decay_length_penalty[b_req_idx],
    )


def test_update_req_to_token_id_counter():
    b_req_idx = torch.tensor([0, 1, 3, 2], dtype=torch.int32, device="cuda")
    next_token_ids = torch.tensor([0, 1, 1, 0], dtype=torch.int32, device="cuda")
    req_to_out_token_id_counter = torch.zeros((4, 4), dtype=torch.int32, device="cuda")
    update_req_to_token_id_counter(b_req_idx, next_token_ids, req_to_out_token_id_counter)
    expected_output = torch.tensor(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ],
        dtype=torch.int32,
        device="cuda",
    )

    assert torch.equal(req_to_out_token_id_counter, expected_output)


if __name__ == "__main__":
    pytest.main()
