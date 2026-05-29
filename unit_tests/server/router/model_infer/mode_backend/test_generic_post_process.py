import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("requires CUDA", allow_module_level=True)

from lightllm.server.router.model_infer.mode_backend.generic_post_process import _trim_post_sample_tensors


def test_trim_post_sample_tensors():
    selected = torch.tensor([1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32, device="cuda")
    selected_rows = torch.where(selected.cpu() == 1)[0]
    dynamic_batch_size = int(selected.sum().item())

    b_req_idx = torch.arange(12, dtype=torch.int32, device="cuda") + 10
    b_temperatures = torch.arange(12, dtype=torch.float32, device="cuda") + 0.5
    b_top_ps = torch.arange(12, dtype=torch.float32, device="cuda") / 100 + 0.8
    b_top_ks = torch.arange(12, dtype=torch.int32, device="cuda") + 100
    b_length_penalty_param = torch.arange(12, dtype=torch.int32, device="cuda") + 200
    b_mask_eos_reqs = torch.tensor(
        [True, False, True, False, True, False, True, False, True, False, True, False],
        dtype=torch.bool,
        device="cuda",
    )

    (
        out_b_req_idx,
        out_b_temperatures,
        out_b_top_ps,
        out_b_top_ks,
        out_b_length_penalty_param,
        out_b_mask_eos_reqs,
    ) = _trim_post_sample_tensors(
        dynamic_batch_size=dynamic_batch_size,
        selected_run_reqs=selected,
        b_req_idx=b_req_idx,
        b_temperatures=b_temperatures,
        b_top_ps=b_top_ps,
        b_top_ks=b_top_ks,
        b_length_penalty_param=b_length_penalty_param,
        b_mask_eos_reqs=b_mask_eos_reqs,
    )
    torch.cuda.synchronize()

    assert torch.equal(out_b_req_idx.cpu(), b_req_idx.cpu()[selected_rows])
    assert torch.equal(out_b_temperatures.cpu(), b_temperatures.cpu()[selected_rows])
    assert torch.equal(out_b_top_ps.cpu(), b_top_ps.cpu()[selected_rows])
    assert torch.equal(out_b_top_ks.cpu(), b_top_ks.cpu()[selected_rows])
    assert torch.equal(out_b_length_penalty_param.cpu(), b_length_penalty_param.cpu()[selected_rows])
    assert torch.equal(out_b_mask_eos_reqs.cpu(), b_mask_eos_reqs.cpu()[selected_rows])
