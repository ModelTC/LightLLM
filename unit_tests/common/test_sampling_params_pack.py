from types import SimpleNamespace
import pytest
import torch
from lightllm.server.router.model_infer.mode_backend import generic_post_process as sampler


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def requests():
    reqs = []
    for idx, out_len, temperature, top_p, top_k, invalid in [
        (9, 0, 0.25, 1.0, 1, [3, 5]),
        (2, 7, 0.7, 0.7, 40, []),
        (15, 10, 1.2, 0.9, 129280, [7]),
    ]:
        req = SimpleNamespace(
            req_idx=idx,
            vocab_size=129280,
            generator=None,
            shm_req=SimpleNamespace(input_len=100),
            sampling_param=SimpleNamespace(
                invalid_token_ids=invalid,
                shm_param=SimpleNamespace(
                    exponential_decay_length_penalty=SimpleNamespace(to_tuple=lambda: (2, 1.0)),
                    min_new_tokens=6,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                ),
            ),
        )
        req.get_cur_total_len = lambda out_len=out_len: 100 + out_len
        reqs.append(req)
    return reqs


def test_packed_fields_preserve_dtype_order_and_invalid_prefix():
    reqs = requests()
    reqs[1].generator = object()
    result = sampler._get_post_sample_tensors(reqs)
    values = [[9, 2, 15], [0.25, 0.7, 1.2], [1.0, 0.7, 0.9], [1, 40, 129280], [0, 5, 8], [True, False, False]]
    dtypes = [torch.int32, torch.float32, torch.float32, torch.int32, torch.int32, torch.bool]
    for tensor, data, dtype in zip(result[:6], values, dtypes):
        assert tensor.dtype == dtype and tensor.is_contiguous()
        torch.testing.assert_close(tensor.cpu(), torch.tensor(data, dtype=dtype), rtol=0, atol=0)
    assert len({x.untyped_storage().data_ptr() for x in result[:6]}) == 1
    torch.testing.assert_close(result[6].cpu(), torch.tensor([3, 5, 7], dtype=torch.int32))
    torch.testing.assert_close(result[7].cpu(), torch.tensor([0, 2, 2, 3], dtype=torch.int32))
    assert result[8:] == (False, True, False, False, True)


def test_dynamic_changes_and_duplicate_rows_keep_previous_result_alive():
    reqs = requests()
    first = sampler._get_post_sample_tensors([reqs[2], reqs[0], reqs[2]])
    param = reqs[2].sampling_param.shm_param
    param.temperature, param.top_p, param.top_k = 2.0, 1.0, 1
    second = sampler._get_post_sample_tensors([reqs[2], reqs[0], reqs[2]])
    torch.cuda.synchronize()
    torch.testing.assert_close(first[1].cpu(), torch.tensor([1.2, 0.25, 1.2]), rtol=0, atol=0)
    torch.testing.assert_close(second[1].cpu(), torch.tensor([2.0, 0.25, 2.0]), rtol=0, atol=0)
    torch.testing.assert_close(second[2].cpu(), torch.ones(3), rtol=0, atol=0)
    torch.testing.assert_close(second[3].cpu(), torch.ones(3, dtype=torch.int32), rtol=0, atol=0)
    assert second[8:] == (True, True, False, True, False)
