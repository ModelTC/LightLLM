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

from typing import TYPE_CHECKING
import json
from pathlib import Path
import socket
import sys

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
    executor = StaticBenchmarkExecutor(args, model, draft_models, TokenSource(args))
    resources = model.att_resource_manager
    (full_state,) = resources.get_full_attention_states("main")
    for draft in draft_models:
        assert draft.att_resource_manager is resources
        assert not hasattr(draft, "full_att_state")
        (draft_state,) = resources.get_full_attention_states("draft0")
        assert draft_state is not full_state
        assert draft_state.capacity == full_state.capacity == resources.full_att_allocator.size
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
    resources.reset_attention_tail_state(source_req.tolist())
    tokens = np.random.default_rng(42).integers(100, 10000, (1, prefix_len), dtype=np.int64)
    for page, length in zip(pages, lengths):
        batch = executor._make_prefill_input(tokens[:, page * page_tokens : length], source_req, page * page_tokens)
        output = model.forward(batch)
        if draft_models:
            executor._fill_mtp_prefill_kv(None, batch, output, executor._argmax_ids(output.logits))
        if hybrid:
            resources.save_bounded_att_pages(source_req.tolist(), page_kind="big", slot_ids=[source_slots[page]])
    torch.cuda.synchronize()
    next_token = output.logits.argmax(dim=-1).cpu()
    source_indices = model.req_manager.req_to_token_indexs[int(source_req[0]), :prefix_len].clone()

    cpu = torch.zeros((len(pages), resources.get_cpu_page_bytes()), dtype=torch.uint8, pin_memory=True)
    resources.write_req_to_cpu(
        source_indices,
        cpu,
        pages,
        layouts=resources.get_cpu_transfer_layouts(),
        page_kind="big" if hybrid else None,
        slot_ids=source_slots,
    )
    # Host-side checkpoint repacking consumes the CPU page on the next call.
    torch.cuda.synchronize()
    full_att_layouts = resources.get_pd_full_att_transfer_layouts()
    runtime_layouts = resources.get_pd_bounded_att_or_tail_transfer_layouts()
    pd = {"kv": torch.empty((1, resources.get_pd_page_bytes()), dtype=torch.uint8, device="cuda")}
    if runtime_layouts:
        pd["att_state"] = torch.empty((1, resources.get_pd_page_bytes()), dtype=torch.uint8, device="cuda")
    resources.write_to_pd_buffer(
        pd["kv"],
        pd_page_index=0,
        kind="kv",
        layouts=full_att_layouts,
        token_indices=source_indices,
    )
    if "att_state" in pd:
        resources.write_to_pd_buffer(
            pd["att_state"],
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
        resources.reset_attention_tail_state(request.tolist())
        executor._ensure_req_kv_capacity(request, torch.tensor([prefix_len], dtype=torch.int32))
        indices = model.req_manager.req_to_token_indexs[int(request[0]), :prefix_len].clone()
        assert not set(indices.tolist()) & set(source_indices.tolist())
        if kind == "gpu_pages":
            resources.copy_full_att_tokens(source_indices, indices)
            if hybrid:
                resources.restore_bounded_att_pages(request.tolist(), page_kind="big", slot_ids=[source_slots[-1]])
            resources.reset_attention_tail_state(request.tolist())
        elif kind == "cpu_pages":
            slots = big.alloc(len(pages)) if hybrid else None
            if hybrid:
                allocated_slots.extend(slots)
            resources.read_req_from_cpu(
                cpu,
                pages,
                indices,
                layouts=resources.get_cpu_transfer_layouts(),
                big_page_slot_ids=slots,
                req_idx=int(request[0]) if getattr(resources, "cpu_tail_restore", False) else None,
            )
            if hybrid:
                resources.restore_bounded_att_pages(request.tolist(), page_kind="big", slot_ids=[slots[-1]])
        elif kind == "dp_copy":
            resources.copy_from_dp(
                resources,
                source_indices,
                indices,
                mappings=resources.dp_copy_mappings(resources),
                src_req_indices=source_req.tolist() if hybrid else None,
                dst_req_indices=request.tolist() if hybrid else None,
            )
        else:
            resources.read_from_pd_buffer(
                pd["kv"],
                pd_page_index=0,
                kind="kv",
                layouts=full_att_layouts,
                token_indices=indices,
            )
            if "att_state" in pd:
                resources.read_from_pd_buffer(
                    pd["att_state"],
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
                step_width,
                request_rows,
                mtp_index,
                lengths,
                next_token.repeat(step_width),
                int(lengths[-1]),
            )
            output = model.forward(batch)
            logits = [output.logits.clone()]
            for draft in draft_models:
                batch.input_ids = output.logits.argmax(dim=-1)
                batch.mtp_draft_input_hiddens = output.mtp_collector.spec_hidden
                output = draft.forward(batch)
                logits.append(output.logits.clone())
            if draft_models and hybrid:
                (
                    resources.main_kda_conv_bf16_ssm_fp32_state if hasattr(resources, "main_kda_conv_bf16_ssm_fp32_state") else resources.main_gdn_conv_bf16_ssm_fp32_state
                ).commit_bounded_att_state(request, [accepted_width])
            return logits

        reference = decode(source_req)
        assert all(torch.isfinite(logits).all() for logits in reference)
        for kind, request in targets.items():
            actual = decode(request)
            for restored_logits, reference_logits in zip(actual, reference):
                torch.testing.assert_close(restored_logits, reference_logits, rtol=1e-4, atol=1e-4)
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
        llm_prefill_att_backend=["fa3", "triton"],
        llm_decode_att_backend=["fa3", "triton"],
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
        from lightllm.utils.envs_utils import set_model_init_status
        from lightllm.utils.torch_memory_saver_utils import MemoryTag
        from lightllm.common.att_resources import AttResourceManager
        from lightllm.common.req_manager import ReqManager

        models = (model, *draft_models)
        assert all(getattr(item, "att_resource_manager", None) is None for item in models), "resources already initialized"
        assert all(item.max_req_num == model.max_req_num for item in models), "models must share the request domain"
        # main/draft 共用请求映射；按所有模型的最大需求分配，再逐模型绑定。
        max_seq_len = max(item.get_req_manager_max_seq_len() for item in models)
        with model.torch_memory_saver.region(tag=MemoryTag.KV_CACHE):
            req_manager = ReqManager(max_request_num=model.max_req_num, max_sequence_length=max_seq_len)
        for item in models:
            item.bind_req_manager(req_manager=req_manager)
        for item in models:
            item.prepare_att_resources()
        manager: AttResourceManager = model.att_resource_manager
        assert all(item.att_resource_manager is manager for item in models), "models must share one resource manager"
        with model.torch_memory_saver.region(tag=MemoryTag.KV_CACHE):
            manager.finish_state_creation()
            manager.load_full_att_calibration(args=model.args)
            manager.initialize_resources(
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
