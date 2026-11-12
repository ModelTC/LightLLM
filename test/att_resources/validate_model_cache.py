"""Validate resource restoration against an uninterrupted, real model decode.

Run from the repository root with local model weights, for example:
    PYTHONPATH=. python test/att_resources/validate_model_cache.py \
        --model_dir /models/Qwen3.5-0.8B --report /tmp/hybrid-cache.json

This checks GPU pages, CPU pages, DP copies and PD buffers on one GPU. Network
transport is tested separately. Each CPU page captures a real forward boundary.
Decode uses CUDA graphs by default; pass --disable_cudagraph to check eager mode.
Native MTP can be checked with --mtp_mode vanilla_with_att --mtp_step 1 and
--mtp_draft_model_dir pointing to the checkpoint containing its draft weights.
MTP uses identical candidate packets with varying accepted widths, not sampling.
"""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.distributed as dist
from transformers import PretrainedConfig

from lightllm.distributed import dist_group_manager
from lightllm.models import get_model
from lightllm.server.api_cli import make_argument_parser
from lightllm.utils.dist_utils import init_distributed_env
from lightllm.utils.envs_utils import get_unique_server_name, set_env_start_args, set_unique_server_name
from lightllm.utils.service_shm_cleanup import ServiceShmCleanup

if TYPE_CHECKING:
    from argparse import Namespace
    from collections.abc import Sequence

    from lightllm.common.basemodel.basemodel import TpPartBaseModel


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark" / "static_inference"))
from static_benchmark import (
    StaticBenchmarkExecutor,
    TokenSource,
    add_static_benchmark_args,
    build_cases,
    build_model_kvargs,
    init_mtp_draft_models,
    normalize_args,
)


@torch.inference_mode()
def validate(model: TpPartBaseModel, args: Namespace, draft_models: Sequence[TpPartBaseModel] = ()):
    executor = StaticBenchmarkExecutor(
        args=args,
        model=model,
        draft_models=draft_models,
        token_source=TokenSource(args),
    )
    resources = model.att_state_manager
    full_states = resources.get_full_attention_states("main")
    assert full_states
    for state in full_states:
        assert state.capacity == resources.full_att_allocator.size
    for draft in draft_models:
        assert draft.att_state_manager is resources
        assert not hasattr(draft, "full_att_state")
        draft_states = resources.get_full_attention_states(draft.att_state_role)
        assert draft_states
        for state in draft_states:
            assert all(state is not main_state for main_state in full_states)
            assert state.capacity == resources.full_att_allocator.size
        assert draft.req_manager is model.req_manager
    prefix_len, page_tokens = args.input_len, args.cpu_cache_token_page_size
    if prefix_len % page_tokens:
        raise ValueError("input_len must contain complete CPU pages")
    pages = list(range(prefix_len // page_tokens))
    lengths = [(page + 1) * page_tokens for page in pages]
    hybrid = model.is_hybrid_att_model
    big = resources.bounded_att_big_page_allocator
    source_slots = big.alloc(len(pages)) if hybrid else None
    allocated_slots = list(source_slots) if hybrid else []
    source_req = executor._alloc_req_indexes(1)
    resources.reset_bounded_att_state(source_req.tolist())
    tokens = np.random.default_rng(42).integers(
        low=100,
        high=10000,
        size=(1, prefix_len),
        dtype=np.int64,
    )
    for page, length in zip(pages, lengths):
        batch = executor._make_prefill_input(tokens[:, page * page_tokens : length], source_req, page * page_tokens)
        output = model.forward(batch)
        if draft_models:
            executor._fill_mtp_prefill_kv(
                case=None,
                main_prefill_input=batch,
                main_output=output,
                first_next_ids=executor._argmax_ids(output.logits),
            )
        if hybrid:
            resources.save_bounded_att_runtime_to_pages(
                req_indices=source_req.tolist(),
                state_token_lens=[length],
                page_kind="big",
                page_slot_ids=[source_slots[page]],
            )
    torch.cuda.synchronize()
    next_token = output.logits.argmax(dim=-1).cpu()
    source_indices = model.req_manager.req_to_token_indexs[int(source_req[0]), :prefix_len].clone()

    cpu = torch.zeros(
        (len(pages), resources.get_cpu_cache_operator().get_page_bytes()), dtype=torch.uint8, pin_memory=True
    )
    resources.get_cpu_cache_operator().write_req(
        token_indices=source_indices,
        cpu_page_buffers=cpu,
        cpu_page_indices=pages,
        layouts=resources.get_cpu_cache_operator().get_page_layouts(),
        big_page_slot_ids=source_slots if hybrid else None,
    )
    # Host-side checkpoint repacking consumes the CPU page on the next call.
    torch.cuda.synchronize()
    full_att_layouts = resources.get_pd_operator().get_pd_full_att_page_layouts()
    runtime_layouts = resources.get_pd_operator().get_pd_bounded_att_page_layouts()
    pd = {"kv": torch.empty((1, resources.get_pd_operator().get_page_bytes()), dtype=torch.uint8, device="cuda")}
    if runtime_layouts:
        pd["att_state"] = torch.empty(
            (1, resources.get_pd_operator().get_page_bytes()), dtype=torch.uint8, device="cuda"
        )
    resources.get_pd_operator().write_page(
        pd_page_buffers=pd["kv"],
        pd_page_index=0,
        kind="kv",
        layouts=full_att_layouts,
        token_indices=source_indices,
    )
    if "att_state" in pd:
        resources.get_pd_operator().write_page(
            pd_page_buffers=pd["att_state"],
            pd_page_index=0,
            kind="att_state",
            layouts=runtime_layouts,
            req_idx=int(source_req[0]),
            state_token_len=prefix_len,
        )

    targets = {}
    for kind in ("gpu_pages", "cpu_pages", "dp_copy", "pd_buffer"):
        request = executor._alloc_req_indexes(1)
        resources.reset_bounded_att_state(request.tolist())
        executor._ensure_req_kv_capacity(request, torch.tensor([prefix_len], dtype=torch.int32))
        indices = model.req_manager.req_to_token_indexs[int(request[0]), :prefix_len].clone()
        assert not set(indices.tolist()) & set(source_indices.tolist())
        if kind == "gpu_pages":
            resources.copy_full_att_tokens(source_indices, indices)
            if hybrid:
                resources.restore_bounded_att_runtime_from_pages(
                    req_indices=request.tolist(),
                    state_token_lens=[prefix_len],
                    page_kind="big",
                    page_slot_ids=[source_slots[-1]],
                )
        elif kind == "cpu_pages":
            slots = big.alloc(len(pages)) if hybrid else None
            if hybrid:
                allocated_slots.extend(slots)
            resources.get_cpu_cache_operator().read_req(
                cpu_page_buffers=cpu,
                cpu_page_indices=pages,
                token_indices=indices,
                layouts=resources.get_cpu_cache_operator().get_page_layouts(),
                big_page_slot_ids=slots,
                req_idx=int(request[0]) if hybrid else None,
                state_token_len=prefix_len if hybrid else None,
            )
        elif kind == "dp_copy":
            resources.get_dp_shared_operator().copy_from_dp(
                remote_att_manager=resources,
                src_token_indices=source_indices,
                dst_token_indices=indices,
                src_big_page_ids=[],
                dst_big_page_ids=[],
                src_req_idx=int(source_req[0]),
                dst_req_idx=int(request[0]),
            )
        else:
            resources.get_pd_operator().read_page(
                pd_page_buffers=pd["kv"],
                pd_page_index=0,
                kind="kv",
                layouts=full_att_layouts,
                token_indices=indices,
            )
            if "att_state" in pd:
                resources.get_pd_operator().read_page(
                    pd_page_buffers=pd["att_state"],
                    pd_page_index=0,
                    kind="att_state",
                    layouts=runtime_layouts,
                    req_idx=int(request[0]),
                    state_token_len=prefix_len,
                )
        targets[kind] = request

    errors = {kind: [] for kind in targets}
    draft_errors = {kind: [] for kind in targets} if draft_models else None
    generated = []
    accepted_widths = []
    sequence_length = prefix_len
    step_width = args.mtp_step + 1 if draft_models else 1
    for step in range(args.output_len):
        generated.append(int(next_token[0]))
        accepted_width = step % step_width + 1
        accepted_widths.append(accepted_width)

        def decode(request: torch.Tensor):
            request_rows, mtp_index = executor._build_mtp_decode_index_tensors(request, step_width)
            lengths = executor._build_mtp_seq_len(torch.tensor([sequence_length], dtype=torch.int32), step_width)
            batch = executor._make_decode_input(
                batch_size=step_width,
                req_idx=request_rows,
                mtp_index=mtp_index,
                seq_len=lengths,
                input_ids=next_token.repeat(step_width),
                max_kv_seq_len=int(lengths[-1]),
            )
            output = model.forward(batch)
            logits = [output.logits.clone()]
            for draft in draft_models:
                batch.input_ids = output.logits.argmax(dim=-1)
                batch.mtp_draft_input_hiddens = output.mtp_collector.spec_hidden
                output = draft.forward(batch)
                logits.append(output.logits.clone())
            if draft_models and hybrid:
                cuda_req_indices = request.cuda(non_blocking=True)
                accepted_steps = torch.full_like(cuda_req_indices, accepted_width, dtype=torch.int32)
                for item in (model, *draft_models):
                    resources.commit_bounded_att_state(
                        req_indices=cuda_req_indices,
                        accepted_steps=accepted_steps,
                        role=item.att_state_role,
                    )
            return logits

        reference = decode(source_req)
        assert all(torch.isfinite(logits).all() for logits in reference)
        for kind, request in targets.items():
            actual = decode(request)
            for restored_logits, reference_logits in zip(actual, reference):
                torch.testing.assert_close(
                    actual=restored_logits,
                    expected=reference_logits,
                    rtol=1e-4,
                    atol=1e-4,
                )
                assert torch.equal(restored_logits.argmax(dim=-1), reference_logits.argmax(dim=-1))
            errors[kind].append(float((actual[0] - reference[0]).abs().max()))
            if draft_models:
                draft_errors[kind].append([float((a - b).abs().max()) for a, b in zip(actual[1:], reference[1:])])
        next_token = reference[0][accepted_width - 1 : accepted_width].argmax(dim=-1).cpu()
        sequence_length += accepted_width

    torch.cuda.synchronize()
    if hybrid:
        big.free(allocated_slots)
    executor._reset_model_cache()
    assert resources.full_att_allocator.can_use_mem_size == resources.full_att_allocator.size
    return {
        "model": args.model_dir,
        "model_class": type(model).__name__,
        "transport": "local_buffers",
        "prefix_tokens": prefix_len,
        "cpu_pages": len(pages),
        "decode_steps": args.output_len,
        "generated_token_ids": generated,
        "max_logit_errors": errors,
        "draft_models": [type(draft).__name__ for draft in draft_models],
        "max_draft_logit_errors": draft_errors,
        "accepted_widths": accepted_widths,
        "allocator_reclaimed": True,
        "cuda_graphs": not args.disable_cudagraph,
    }


def main():
    parser = make_argument_parser()
    add_static_benchmark_args(parser)
    parser.add_argument("--report", required=True)
    parser.set_defaults(
        tp=1,
        dp=1,
        data_type="bfloat16",
        batch_size=1,
        max_batch_size=1,
        input_len=128,
        output_len=4,
        batch_max_tokens=256,
        max_total_token_num=2048,
        running_max_req_size=8,
        graph_max_batch_size=2,
        max_req_total_len=512,
        page_size=16,
        cpu_cache_token_page_size=64,
        enable_cpu_cache=True,
        linear_att_hash_page_size=64,
        linear_att_page_block_num=1,
        linear_att_cache_size=16,
        llm_prefill_att_backend="fa3",
        llm_prefill_linear_att_backend="triton",
        llm_decode_att_backend="fa3",
        llm_decode_linear_att_backend="triton",
        disable_cudagraph=False,
        pd_kv_page_size=2048,
    )
    parsed = parser.parse_args()
    if parsed.tp != 1 or parsed.dp != 1 or parsed.mtp_mode not in (None, "vanilla_with_att"):
        parser.error("this validation uses TP=1, DP=1 and optionally native vanilla_with_att MTP")
    with socket.socket() as port:
        port.bind(("127.0.0.1", 0))
        parsed.nccl_port = port.getsockname()[1]
    args = normalize_args(parsed, build_cases(parsed))
    set_unique_server_name(args)
    set_env_start_args(args)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    try:
        kvargs = build_model_kvargs(args, 0)
        init_distributed_env(kvargs)
        dist_group_manager.create_groups(group_size=1)
        config, _ = PretrainedConfig.get_config_dict(args.model_dir)
        model, _ = get_model(config, kvargs)
        draft_models = init_mtp_draft_models(args, kvargs, model)
        from lightllm.common.attention_state import AttentionStateManager
        from lightllm.common.basemodel.attention import reserve_attention_resources
        from lightllm.common.req_manager import ReqManager
        from lightllm.utils.envs_utils import set_model_init_status
        from lightllm.utils.torch_memory_saver_utils import MemoryTag

        models = (model, *draft_models)
        assert all(getattr(item, "att_state_manager", None) is None for item in models), "resources already initialized"
        assert all(item.max_req_num == model.max_req_num for item in models), "models must share the request domain"
        # main/draft 共用请求映射；按所有模型的最大需求分配，再逐模型绑定。
        max_seq_len = max(item.get_req_manager_max_seq_len() for item in models)
        with model.torch_memory_saver.region(tag=MemoryTag.KV_CACHE):
            req_manager = ReqManager(max_request_num=model.max_req_num, max_sequence_length=max_seq_len)
        for item in models:
            item.bind_req_manager(req_manager=req_manager)
        for item in models:
            item.prepare_attention_states()
        manager: AttentionStateManager = model.att_state_manager
        assert all(item.att_state_manager is manager for item in models), "models must share one resource manager"
        with model.torch_memory_saver.region(tag=MemoryTag.KV_CACHE):
            manager.finish_state_registration()
            for item in models:
                item.prepare_attention_backends()
                reserve_attention_resources(item)
            manager.load_full_att_calibration(args=model.args)
            manager.initialize_storage(
                model.args, token_capacity=model.max_total_token_num, mem_fraction=model.mem_fraction
            )
        for item in models:
            item.init_runtime()
        torch.cuda.empty_cache()
        set_model_init_status(True)
        report = validate(model, args, draft_models)
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        ServiceShmCleanup(get_unique_server_name()).cleanup_service_resources()


if __name__ == "__main__":
    main()
