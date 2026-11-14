import json

import pytest

from lightllm.server.api_start import _launch_subprocesses
from lightllm.server.core.objs.start_args_type import StartArgs


@pytest.mark.parametrize("backend", ["auto", "fa3", "triton", "flashinfer"])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_decode_node_batch_max_tokens_covers_mtp_rows(monkeypatch, tmp_path, backend, phase):
    for name in (
        "_set_envs_and_config",
        "auto_set_max_req_total_len",
        "auto_set_fused_shared_experts",
        "set_unique_server_name",
        "auto_set_page_size",
    ):
        monkeypatch.setattr(f"lightllm.server.api_start.{name}", lambda args: None)

    def stop(args):
        raise RuntimeError("startup validation passed")

    monkeypatch.setattr("lightllm.server.api_start.auto_set_response_parsers", stop)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    args = StartArgs(
        run_mode="decode",
        model_dir=str(tmp_path),
        enable_ep_moe=True,
        running_max_req_size=256,
        mtp_mode="vanilla_no_att",
        mtp_step=2,
        max_req_total_len=4096,
        batch_max_tokens=4096,
        chunked_prefill_size=2048,
        eos_id=[0],
        data_type="float16",
        disable_vision=True,
        disable_audio=True,
        disable_shm_warning=True,
        router_token_ratio=0.0,
    )

    setattr(args, f"llm_{phase}_att_backend", backend)
    args.llm_prefill_linear_att_backend = "flashqla"
    args.llm_decode_linear_att_backend = "triton"
    if backend == "flashinfer":
        with pytest.raises(AssertionError, match=f"--llm_{phase}_att_backend"):
            _launch_subprocesses(args)
        return

    with pytest.raises(RuntimeError, match="startup validation passed"):
        _launch_subprocesses(args)

    assert args.batch_max_tokens == 768
