import gc
from collections import deque
from dataclasses import asdict, dataclass

import torch

from lightllm.common.basemodel.batch_objs import ModelInput
from lightllm.common.req_manager import ReqManager
from lightllm.utils.log_utils import init_logger
from lightllm.utils.torch_memory_saver_utils import MemoryTag

logger = init_logger(__name__)


@dataclass(frozen=True)
class RuntimeMemory:
    bootstrap_tokens: int
    profile_peak_reserved_bytes: int
    fixed_reserved_bytes: int
    runtime_reserved_bytes: int
    capacity_reserved_bytes: int
    final_tokens: int
    final_peak_reserved_bytes: int
    attempted_capacities: tuple[int, ...]


def _align(value, page_size):
    return (value + page_size - 1) // page_size * page_size


def _drain(models):
    for model in models:
        model.execution.synchronize()
    models[0].device_runtime.synchronize()


def _initialize_profile_storage(manager):
    for state in manager.get_full_attention_states():
        for tensor in state.storage_tensors():
            tensor.zero_()
    return sum(
        tensor.numel() * tensor.element_size()
        for state in manager.get_full_attention_states()
        for tensor in state.storage_tensors()
    )


def _profile_layout(model, queries, histories):
    manager = model.att_state_manager
    req_manager = model.req_manager
    allocator = manager.full_att_allocator
    page_size = allocator.page_size
    requests = [req_manager.alloc() for _ in queries]
    if any(index is None for index in requests):
        raise ValueError("runtime profile exceeds the configured request domain")
    manager.reset_bounded_att_state(requests)
    hold = torch.as_tensor(allocator.hold_token_indices, device=model.device_runtime.device, dtype=torch.int32)
    for req, query, history in zip(requests, queries, histories):
        row = req_manager.req_to_token_indexs[req]
        row.view(-1, page_size).copy_(hold)
        count = _align(history % page_size + query, page_size)
        first = history // page_size * page_size
        row[first : first + count] = allocator.alloc(count).to(row.device, non_blocking=True)
    return requests


def _profile_input(model, requests, queries, histories, *, prefill):
    device = model.device_runtime.device
    lengths = [history + query for history, query in zip(histories, queries)]
    if prefill:
        rows, mtp = requests, [0] * len(requests)
        total_queries = sum(queries)
        starts, offset = [], 0
        for query in queries:
            starts.append(offset)
            offset += query
        extra = dict(
            b_ready_cache_len=torch.tensor(histories, dtype=torch.int32, device=device),
            b_prefill_start_loc=torch.tensor(starts, dtype=torch.int32, device=device),
            b_is_decode_req=torch.zeros(len(rows), dtype=torch.bool, device=device),
            b_prefill_has_output_cpu=[True] * len(rows),
        )
    else:
        width = model.mtp_manager.get_decode_tokens_per_request(model.is_mtp_draft_model)
        rows = [req for req in requests for _ in range(width)]
        mtp = list(range(width)) * len(requests)
        lengths = [history + step + 1 for history in histories for step in range(width)]
        total_queries = len(rows)
        extra = dict(
            b_position_delta=torch.zeros(len(rows), dtype=torch.int32, device=device),
            b_shared_seq_len=torch.zeros(len(rows), dtype=torch.int32, device=device),
            b_shared_radix_node_id=torch.full((len(rows),), -1, dtype=torch.int32, device=device),
        )
    return ModelInput(
        batch_size=len(rows),
        total_token_num=sum(lengths),
        max_q_seq_len=max(queries, default=1) if prefill else 1,
        max_kv_seq_len=max(lengths, default=1),
        max_cache_len=max(histories, default=0) if prefill else None,
        input_ids=torch.arange(total_queries, device=device, dtype=torch.int64) % model.vocab_size,
        b_req_idx=torch.tensor(rows, device=device, dtype=torch.int32),
        b_mtp_index=torch.tensor(mtp, device=device, dtype=torch.int32),
        b_seq_len=torch.tensor(lengths, device=device, dtype=torch.int32),
        multimodal_params=[{"images": [], "audios": []} for _ in rows],
        is_prefill=prefill,
        **extra,
        **model._gen_special_model_input(total_queries),
    )


@torch.no_grad()
def _profile_case(models, queries, histories, *, prefill, overlap):
    main = models[0]
    manager, req_manager = main.att_state_manager, main.req_manager
    requests = _profile_layout(main, queries, histories)
    retained = deque(maxlen=max(model.execution.max_pending for model in models))
    for _ in range(2):
        outputs = []
        for model in models:
            slices = (slice(None),) if not overlap else (slice(0, len(requests) // 2), slice(len(requests) // 2, None))
            inputs = [
                _profile_input(model, requests[part], queries[part], histories[part], prefill=prefill)
                for part in slices
            ]
            if overlap:
                forward = model.microbatch_overlap_prefill if prefill else model.microbatch_overlap_decode
                result = forward(*inputs)
            else:
                result = (model.forward(inputs[0]),)
            outputs.extend(result)
            for output in result:
                if output.attention_commit is not None:
                    output.attention_commit.consume(req_manager)
                torch.softmax(output.logits, dim=-1)
            if not prefill:
                width = model.mtp_manager.get_decode_tokens_per_request(model.is_mtp_draft_model)
                indexes = torch.tensor(requests, dtype=torch.int32, device=main.device_runtime.device)
                widths = torch.full_like(indexes, width)
                manager.commit_bounded_att_state(indexes, widths, widths, model.att_state_role)
        retained.append(outputs)
    _drain(models)
    manager.reset_bounded_att_state(requests)
    req_manager.free(requests)
    manager.full_att_allocator.free_all()


def _profile_runtime(models):
    main = models[0]
    token_limit = max(model.batch_max_tokens for model in models)
    seq_limit = min(model.max_seq_length for model in models)
    request_limit = main.max_req_num
    cold_requests = max(1, (token_limit + seq_limit - 1) // seq_limit)
    if cold_requests > request_limit:
        raise ValueError("batch_max_tokens exceeds the configured request/sequence envelope")
    cold = [min(seq_limit, token_limit - index * seq_limit) for index in range(cold_requests)]
    query = min(token_limit, max(1, seq_limit // 2))
    requests = min(request_limit, token_limit)
    many = [token_limit // requests + (index < token_limit % requests) for index in range(requests)]
    width = max(model.mtp_manager.get_decode_tokens_per_request(model.is_mtp_draft_model) for model in models)
    cases = (
        (cold, [0] * len(cold), True),
        ([query], [seq_limit - query], True),
        (many, [seq_limit - query for query in many], True),
        ([width] * request_limit, [seq_limit - width] * request_limit, False),
    )
    graphs = [(model.graph, model.prefill_graph) for model in models]
    try:
        for model in models:
            model.graph = model.prefill_graph = None
        for queries, histories, prefill in cases:
            overlap = (
                main.args.enable_prefill_microbatch_overlap if prefill else main.args.enable_decode_microbatch_overlap
            )
            _profile_case(models, queries, histories, prefill=prefill, overlap=overlap)
    finally:
        for model, (decode_graph, prefill_graph) in zip(models, graphs):
            model.graph, model.prefill_graph = decode_graph, prefill_graph


def _release_runtime_storage(models):
    _drain(models)
    manager = models[0].att_state_manager
    assert models[0].req_manager.req_list.is_all_free(), "startup requests are still allocated"
    manager.close_graphs()
    for model in models:
        model.graph = model.prefill_graph = None
        model.attention.unbind(model)
    _drain(models)
    manager.release_storage()
    gc.collect()
    models[0].device_runtime.empty_cache()


def initialize_model_runtime(models):
    models = tuple(models)
    main = models[0]
    args, runtime = main.args, main.device_runtime
    requested_capacity = main.max_total_token_num
    assert all(getattr(model, "att_state_manager", None) is None for model in models), "resources already initialized"
    assert all(model.max_req_num == main.max_req_num for model in models), "models must share the request domain"
    with main.torch_memory_saver.region(tag=MemoryTag.KV_CACHE):
        req_manager = ReqManager(
            max_request_num=main.max_req_num,
            max_sequence_length=max(model.get_req_manager_max_seq_len() for model in models),
        )
    for model in models:
        model.bind_req_manager(req_manager)
        model.prepare_attention_states()
    manager = main.att_state_manager
    assert all(model.att_state_manager is manager for model in models), "models must share one State manager"
    manager.finish_state_registration()
    for model in models:
        model.prepare_attention_backends()
        model.attention.reserve(model)
    with main.torch_memory_saver.region(tag=MemoryTag.KV_CACHE):
        manager.load_full_att_calibration(args)

    width = max(model.mtp_manager.get_decode_tokens_per_request(model.is_mtp_draft_model) for model in models)
    bootstrap_tokens = _align(
        max(
            max(model.max_seq_length for model in models),
            max(model.batch_max_tokens for model in models) + main.max_req_num * (args.page_size + width),
        ),
        args.page_size,
    )
    with main.torch_memory_saver.region(tag=MemoryTag.KV_CACHE):
        manager.initialize_storage(args, bootstrap_tokens, main.mem_fraction, device=runtime.device)
    bootstrap_bytes = _initialize_profile_storage(manager)
    runtime.reset_peak_memory()
    for model in models:
        model.init_runtime()
    _profile_runtime(models)
    peak = runtime.memory_reserved(peak=True)
    _release_runtime_storage(models)
    fixed = runtime.memory_reserved()
    runtime_reserve = max(0, peak - bootstrap_bytes - fixed)

    def capacity_reservation(physical_tokens):
        return sum(model.attention.capacity_memory_bytes(model, physical_tokens) for model in models)

    attempted = []
    for attempt in range(4):
        with main.torch_memory_saver.region(tag=MemoryTag.KV_CACHE):
            manager.initialize_storage(
                args,
                requested_capacity,
                main.mem_fraction,
                device=runtime.device,
                runtime_reserved_bytes=runtime_reserve,
                capacity_reservation=capacity_reservation,
            )
        final_capacity = manager.full_att_allocator.size
        attempted.append(final_capacity)
        _initialize_profile_storage(manager)
        runtime.reset_peak_memory()
        for model in models:
            model.init_runtime(autotune=False)
        _profile_runtime(models)
        _drain(models)
        final_peak = runtime.memory_reserved(peak=True)
        runtime.empty_cache()
        future_peak = max(0, final_peak - runtime.memory_reserved())
        free, total = runtime.memory_info()
        transfer = (
            manager.get_pd_operator().get_page_bytes() * args.pd_kv_page_num
            if args.run_mode in ("prefill", "decode")
            else 0
        )
        needed = (
            int(total * (1 - main.mem_fraction))
            + transfer
            + future_peak
            + capacity_reservation(final_capacity + args.page_size)
        )
        available = torch.tensor(free - needed, dtype=torch.int64, device=runtime.device)
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            torch.distributed.all_reduce(available, op=torch.distributed.ReduceOp.MIN)
        shortfall = max(0, -available.item())
        if not shortfall:
            break
        if requested_capacity is not None or attempt == 3:
            raise MemoryError(
                f"final runtime profile exceeds the admitted memory budget by {shortfall} bytes; "
                f"capacities={attempted}, peak_reserved={final_peak}; reduce KV capacity or workload"
            )
        page_bytes = args.page_size * manager.get_full_att_token_bytes()
        runtime_reserve += _align(shortfall, page_bytes) + page_bytes
        logger.info(
            "Final profile needs %d additional bytes; retiring capacity %d and readmitting with reserve %d",
            shortfall,
            final_capacity,
            runtime_reserve,
        )
        _release_runtime_storage(models)
        fixed = runtime.memory_reserved()
    result = RuntimeMemory(
        bootstrap_tokens,
        peak,
        fixed,
        runtime_reserve,
        capacity_reservation(final_capacity + args.page_size),
        final_capacity,
        final_peak,
        tuple(attempted),
    )
    for model in models:
        model.runtime_memory = result
    logger.info("Runtime memory admission (measured tensor peak): %s", asdict(result))
    return result
