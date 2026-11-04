"""Opt-in local checkpoint regression; does not download weights or start a service."""

from dataclasses import asdict
import json
import os
from types import SimpleNamespace

import pytest
import torch

from unit_tests.common.execution.test_model_cuda import prepare_model_batch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not os.environ.get("LIGHTLLM_TEST_QWEN35_CHECKPOINT"), reason="local Qwen3.5 checkpoint required")
@pytest.mark.parametrize("reference_kind", ["legacy", "transformers_bf16", "transformers_fp16"])
@torch.no_grad()
def test_qwen35_checkpoint_matches_reference_and_restores_recurrent_state(monkeypatch, reference_kind):
    from transformers import AutoTokenizer
    from lightllm.models.qwen3_5.model import Qwen3_5TpPartModel
    from lightllm.common.basemodel.mtp_manager import MtpManager
    from lightllm.distributed.communication_op import dist_group_manager
    from lightllm.server.core.objs.start_args_type import StartArgs
    from lightllm.utils.envs_utils import get_env_start_args, get_llm_data_type

    directory = os.environ["LIGHTLLM_TEST_QWEN35_CHECKPOINT"]
    for name in (
        "GLOBAL_RANK",
        "GLOBAL_DP_RANK",
        "DP_RANK_IN_NODE",
        "CURRENT_RANK_IN_DP",
        "CURRENT_RANK_IN_NODE",
        "CURRENT_DEVICE_ID",
    ):
        monkeypatch.setenv("LIGHTLLM_" + name, "0")
    for name in ("GLOBAL_WORLD_SIZE", "DP_SIZE", "DP_WORLD_SIZE", "NODE_WORLD_SIZE"):
        monkeypatch.setenv("LIGHTLLM_" + name, "1")
    args = StartArgs(
        model_dir=directory,
        data_type="float16" if reference_kind == "transformers_fp16" else "bfloat16",
        page_size=1,
        llm_prefill_att_backend=["triton", "triton"],
        llm_decode_att_backend=["triton", "triton"],
        linear_att_ssm_data_type="float32",
        running_max_req_size=2,
    )
    monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(asdict(args)))
    get_env_start_args.cache_clear()
    get_llm_data_type.cache_clear()
    monkeypatch.setattr(MtpManager, "_instance", None)
    monkeypatch.setattr(dist_group_manager, "groups", [SimpleNamespace()])

    def create(native):
        with monkeypatch.context() as patch:
            if not native:
                patch.setattr(Qwen3_5TpPartModel, "model_adapter_class", None)
            model = Qwen3_5TpPartModel(
                dict(
                    run_mode="normal",
                    weight_dir=directory,
                    max_total_token_num=128,
                    batch_max_tokens=64,
                    max_req_num=2,
                    max_seq_length=64,
                    graph_max_batch_size=2,
                    graph_max_len_in_batch=64,
                    disable_cudagraph=True,
                )
            )
        model.prepare_attention_states()
        model.att_state_manager.finish_state_registration()
        for state in model.att_state_manager.get_full_attention_states():
            state.device = torch.device("cuda")
            state.allocate_storage(128)
        model.req_manager = SimpleNamespace(
            req_to_token_indexs=torch.arange(128, dtype=torch.int32, device="cuda").view(2, 64)
        )
        if not native:
            model._init_att_backend()
            model._init_att_backend1()
        else:
            model.prefill_att_backend = model.decode_att_backend = None
            model.prefill_att_backend1 = model.decode_att_backend1 = None
        model._init_execution_plan()
        model._init_hidden_collector()
        assert (model.execution_plan is not None) == native
        if native:
            monkeypatch.setattr(
                model.att_state_manager, "get_state", lambda *a, **k: pytest.fail("runtime manager lookup")
            )
        return model

    try:
        legacy = create(False) if reference_kind == "legacy" else None
        reference = None
        if reference_kind.startswith("transformers"):
            module = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
            # Keep the reference independent of optimized attention and convolution kernels.
            monkeypatch.setattr(module, "causal_conv1d_fn", None)
            monkeypatch.setattr(module, "causal_conv1d_update", None)
            monkeypatch.setattr(module, "chunk_gated_delta_rule", None)
            monkeypatch.setattr(module, "fused_recurrent_gated_delta_rule", None)
            reference = (
                module.Qwen3_5ForConditionalGeneration.from_pretrained(
                    directory, local_files_only=True, dtype=torch.float32, attn_implementation="eager"
                )
                .cuda()
                .eval()
            )
        native = create(True)
        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        sequences = [tokenizer.encode(prompt) for prompt in ("请简短解释为什么天空是蓝色的。", "用 Python 写一个计算斐波那契数列的函数。")]
        assert max(map(len, sequences)) < 48
        cached = [0, 0]
        rms_errors, peak_errors = [], []

        def forward(prefill):
            outputs = []
            if reference is not None:
                outputs.append(
                    torch.cat(
                        [
                            reference(torch.tensor([seq], device="cuda"), use_cache=False).logits[:, -1].float()
                            for seq in sequences
                        ]
                    )
                )
            for model in (native,) if legacy is None else (legacy, native):
                batch = prepare_model_batch(model, sequences, cached, prefill)
                outputs.append(
                    (model._context_forward(batch) if prefill else model._token_forward(batch)).logits.clone()
                )
            if reference is None:
                torch.testing.assert_close(outputs[1], outputs[0], rtol=3e-3, atol=5e-3)
            else:
                # Compare low-precision execution with an independent FP32 reference.
                difference = outputs[1].float() - outputs[0]
                rms_errors.extend((difference.square().mean(-1) / outputs[0].square().mean(-1)).sqrt().tolist())
                peak_errors.extend((difference.abs().amax(-1) / outputs[0].abs().amax(-1)).tolist())
            assert torch.equal(outputs[1].argmax(-1), outputs[0].argmax(-1))
            if legacy is not None:
                old_states = legacy.att_state_manager.get_bounded_attention_states()
                new_states = native.att_state_manager.get_bounded_attention_states()
                for old, new in zip(old_states, new_states):
                    assert old.config == new.config
                    torch.testing.assert_close(new.conv_state, old.conv_state, rtol=3e-3, atol=5e-3)
                    torch.testing.assert_close(new.ssm_state, old.ssm_state, rtol=3e-3, atol=5e-3)
            return outputs[0].argmax(-1).tolist()

        tokens = forward(True)
        for _ in range(4):
            cached = list(map(len, sequences))
            sequences = [seq + [token] for seq, token in zip(sequences, tokens)]
            tokens = forward(False)
        cached = list(map(len, sequences))
        suffix = tokenizer.encode("请再简短一点。")
        sequences = [seq + suffix for seq in sequences]
        tokens = forward(True)
        for state in native.att_state_manager.get_bounded_attention_states():
            state.initialize_page_pools(2, device="cuda", pin_memory=False)
            state.save_bounded_att_runtime_to_pages([0, 1], list(map(len, sequences)), "big", [0, 1])
            state.conv_state.zero_()
            state.ssm_state.zero_()
            state.restore_bounded_att_runtime_from_pages([0, 1], list(map(len, sequences)), "big", [0, 1])
        cached = list(map(len, sequences))
        sequences = [seq + [token] for seq, token in zip(sequences, tokens)]
        forward(False)
        if reference is not None:
            assert max(rms_errors) <= 0.01, rms_errors
            assert max(peak_errors) <= 1 / 32, peak_errors
    finally:
        get_env_start_args.cache_clear()
        get_llm_data_type.cache_clear()
