"""DP overlap EAGLE proposer 共享辅助函数。"""

from __future__ import annotations

import copy
from typing import Callable

import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.common.basemodel.triton_kernel.gen_mtp_prefill_params import (
    gen_mtp_new_input_ids,
)
from lightllm.common.basemodel.triton_kernel.mtp_utils import gen_b_req_mtp_start_loc
from lightllm.common.basemodel.triton_kernel.select_mtp_rows import (
    select_accepted_tail_rows,
)
from lightllm.server.router.model_infer.mtp_speculative import utils as mtp_utils
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.base import (
    BaseDpOverlapProposer,
)
from lightllm.server.router.model_infer.mtp_speculative.proposers.base import (
    MtpMemIndexesToFree,
)
from lightllm.server.router.model_infer.mtp_speculative.proposers.proposal_type import (
    EagleSpecProposal,
)
from lightllm.server.router.model_infer.pin_mem_manager import g_pin_mem_manager


def _prepare_eagle_prefill_inputs(
    model_input: ModelInput,
    b_next_token_ids: torch.Tensor,
    mtp_draft_input_hiddens: torch.Tensor,
) -> None:
    """构造 DP-overlap EAGLE draft model 的 prefill 输入。"""

    model_input.b_is_decode_req = g_pin_mem_manager.get_const_gpu_tensor(
        key="dp_overlap_eagle_prefill_b_is_decode_req",
        shape=model_input.b_req_idx.shape,
        fill_value=False,
        dtype=torch.bool,
    )
    model_input.input_ids = gen_mtp_new_input_ids(
        input_ids=model_input.input_ids,
        b_next_token_ids=b_next_token_ids,
        b_seq_len=model_input.b_seq_len,
        b_ready_cache_len=model_input.b_ready_cache_len,
    )
    model_input.mtp_draft_input_hiddens = mtp_draft_input_hiddens


def generate_eagle_tokens(
    proposer: BaseDpOverlapProposer,
    model_output: ModelOutput,
    map_draft_token_ids: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if proposer.enable_dynmaic_mtp:
        draft_token_ids, draft_token_probs = proposer.backend._gen_argmax_token_ids_and_prob(model_output)
        return map_draft_token_ids(draft_token_ids), draft_token_probs.float()
    return (
        map_draft_token_ids(proposer.backend._gen_argmax_token_ids(model_output)),
        None,
    )


def prepare_eagle_verify_decode_input(
    model_input: ModelInput,
    input_ids: torch.Tensor,
    target_hidden: torch.Tensor,
) -> None:
    """复用 target verify 布局构造 DP-overlap drafter 输入。"""

    assert not model_input.is_prefill
    model_input.input_ids = input_ids
    model_input.mtp_draft_input_hiddens = target_hidden


def fill_dp_eagle_draft_model_kv_state_overlap(
    proposer: BaseDpOverlapProposer,
    target_model_input0: ModelInput,
    target_model_output0: ModelOutput,
    target_next_token_ids0: torch.Tensor,
    target_model_input1: ModelInput,
    target_model_output1: ModelOutput,
    target_next_token_ids1: torch.Tensor,
) -> None:
    """使用两个 target prefill microbatch 初始化 EAGLE draft state。"""

    _prepare_eagle_prefill_inputs(
        model_input=target_model_input0,
        b_next_token_ids=target_next_token_ids0,
        mtp_draft_input_hiddens=target_model_output0.mtp_collector.spec_hidden,
    )
    _prepare_eagle_prefill_inputs(
        model_input=target_model_input1,
        b_next_token_ids=target_next_token_ids1,
        mtp_draft_input_hiddens=target_model_output1.mtp_collector.spec_hidden,
    )
    proposer.backend.draft_models[0]._microbatch_overlap_prefill_cuda(target_model_input0, target_model_input1)


def get_dp_overlap_req_start_rows(b_mtp_index: torch.Tensor, request_num: int) -> torch.Tensor:
    """根据动态 verify 布局恢复请求起始行。"""

    request_num = int(request_num)
    if request_num == 0:
        assert b_mtp_index.numel() == 0
        return torch.empty((0,), dtype=torch.int32, device=b_mtp_index.device)
    if b_mtp_index.is_cuda:
        return gen_b_req_mtp_start_loc(
            b_mtp_index=b_mtp_index,
            num_reqs=request_num,
        )
    req_start_rows = torch.nonzero(b_mtp_index == 0, as_tuple=False).flatten().to(dtype=torch.int32)
    assert req_start_rows.shape == (request_num,)
    return req_start_rows


def prepare_dp_eagle_overlap_decode_inputs(
    target_model_input0: ModelInput,
    target_model_input1: ModelInput,
    real_request_num0: int,
    real_request_num1: int,
    accept_len: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """恢复两个动态 verify microbatch 的请求起始行。"""

    assert accept_len.shape == (real_request_num0 + real_request_num1,)
    accept_len0 = accept_len[:real_request_num0]
    accept_len1 = accept_len[real_request_num0:]
    req_start_rows0 = get_dp_overlap_req_start_rows(
        b_mtp_index=target_model_input0.b_mtp_index,
        request_num=real_request_num0,
    )
    req_start_rows1 = get_dp_overlap_req_start_rows(
        b_mtp_index=target_model_input1.b_mtp_index,
        request_num=real_request_num1,
    )
    return (
        accept_len0,
        accept_len1,
        req_start_rows0,
        req_start_rows1,
    )


def propose_next_dp_eagle_autoregressive_overlap(
    proposer: BaseDpOverlapProposer,
    target_model_input0: ModelInput,
    target_model_output0: ModelOutput,
    target_next_token_ids0: torch.Tensor,
    target_model_input1: ModelInput,
    target_model_output1: ModelOutput,
    target_next_token_ids1: torch.Tensor,
    real_request_num0: int,
    real_request_num1: int,
    accept_len: torch.Tensor,
    draft_step: int,
    map_draft_token_ids: Callable[[torch.Tensor], torch.Tensor],
) -> EagleSpecProposal:
    """运行 DP EAGLE extend 后接单 token overlap decode 的 proposal 流程。"""

    assert target_next_token_ids0.shape == (target_model_input0.batch_size,)
    assert target_next_token_ids1.shape == (target_model_input1.batch_size,)
    (accept_len0, accept_len1, req_start_rows0, req_start_rows1,) = prepare_dp_eagle_overlap_decode_inputs(
        target_model_input0=target_model_input0,
        target_model_input1=target_model_input1,
        real_request_num0=real_request_num0,
        real_request_num1=real_request_num1,
        accept_len=accept_len,
    )

    model_inputs = [copy.copy(target_model_input0), copy.copy(target_model_input1)]
    model_outputs = (target_model_output0, target_model_output1)
    target_next_token_ids_by_batch = (target_next_token_ids0, target_next_token_ids1)
    accept_lens_by_batch = (accept_len0, accept_len1)
    req_start_rows_by_batch = (req_start_rows0, req_start_rows1)
    position_deltas_by_batch = tuple(model_input.b_position_delta for model_input in model_inputs)
    max_kv_seq_lens_by_batch = tuple(model_input.max_kv_seq_len for model_input in model_inputs)

    real_request_counts = (int(real_request_num0), int(real_request_num1))
    accepted_tail_rows_by_batch = []
    for model_input, model_output, token_ids, req_start_rows, accept_len in zip(
        model_inputs,
        model_outputs,
        target_next_token_ids_by_batch,
        req_start_rows_by_batch,
        accept_lens_by_batch,
    ):
        accepted_tail_rows = (req_start_rows + accept_len - 1).long()
        accepted_tail_rows_by_batch.append(accepted_tail_rows)
        prepare_eagle_verify_decode_input(
            model_input=model_input,
            input_ids=token_ids,
            target_hidden=model_output.mtp_collector.spec_hidden,
        )

    total_real_request_count = sum(real_request_counts)
    proposal_token_ids = target_next_token_ids0.new_empty((total_real_request_count, draft_step))
    proposal_schedule_scores = (
        torch.empty(
            (total_real_request_count, draft_step),
            dtype=torch.float32,
            device=target_next_token_ids0.device,
        )
        if proposer.enable_dynmaic_mtp
        else None
    )

    draft_model = proposer.backend.draft_models[0]
    extend_outputs = draft_model._microbatch_overlap_decode_cuda(*model_inputs)

    draft_token_ids_by_batch = []
    draft_hiddens_by_batch = []
    draft_seq_lens_by_batch = []
    draft_req_indices_by_batch = []
    draft_shared_seq_lens_by_batch = []
    draft_shared_radix_node_ids_by_batch = []
    proposal_row_offsets = (0, real_request_counts[0])
    for batch_index, (model_input, extend_output, accepted_tail_rows, real_request_count,) in enumerate(
        zip(
            model_inputs,
            extend_outputs,
            accepted_tail_rows_by_batch,
            real_request_counts,
        )
    ):
        accepted_tail_output = ModelOutput(logits=extend_output.logits.index_select(0, accepted_tail_rows))
        draft_token_ids, draft_token_probs = generate_eagle_tokens(
            proposer,
            accepted_tail_output,
            map_draft_token_ids,
        )
        draft_token_ids_by_batch.append(draft_token_ids)
        draft_hiddens_by_batch.append(extend_output.mtp_collector.spec_hidden.index_select(0, accepted_tail_rows))
        draft_seq_lens_by_batch.append(model_input.b_seq_len.index_select(0, accepted_tail_rows) + 1)
        draft_req_indices_by_batch.append(model_input.b_req_idx.index_select(0, accepted_tail_rows))
        draft_shared_seq_lens_by_batch.append(model_input.b_shared_seq_len.index_select(0, accepted_tail_rows))
        draft_shared_radix_node_ids_by_batch.append(
            model_input.b_shared_radix_node_id.index_select(0, accepted_tail_rows)
        )
        proposal_row_start = proposal_row_offsets[batch_index]
        proposal_token_ids[proposal_row_start : proposal_row_start + real_request_count, 0] = draft_token_ids[
            :real_request_count
        ]
        if proposal_schedule_scores is not None:
            proposal_schedule_scores[
                proposal_row_start : proposal_row_start + real_request_count, 0
            ] = draft_token_probs[:real_request_count]

    if draft_step == 1:
        return EagleSpecProposal(
            token_ids=proposal_token_ids,
            extra_mem_indexes_cpu=[],
            schedule_scores=proposal_schedule_scores,
        )

    for batch_index, model_input in enumerate(model_inputs):
        model_input.is_prefill = False
        model_input.batch_size = real_request_counts[batch_index]
        model_input.b_req_idx = draft_req_indices_by_batch[batch_index]
        model_input.b_mtp_index = torch.zeros_like(model_input.b_req_idx)
        model_input.b_seq_len = draft_seq_lens_by_batch[batch_index]
        assert position_deltas_by_batch[batch_index] is not None
        model_input.b_position_delta = position_deltas_by_batch[batch_index].index_select(
            0, accepted_tail_rows_by_batch[batch_index]
        )
        model_input.b_shared_seq_len = draft_shared_seq_lens_by_batch[batch_index]
        model_input.b_shared_radix_node_id = draft_shared_radix_node_ids_by_batch[batch_index]
        if len(model_input.multimodal_params) != model_input.batch_size:
            empty_multimodal_params = {"images": [], "audios": []}
            model_input.multimodal_params = [empty_multimodal_params] * model_input.batch_size

    extra_mem_indexes_cpu = mtp_utils.alloc_mem_indexes(total_real_request_count * (draft_step - 1))
    extra_mem_indexes = extra_mem_indexes_cpu.to(device=target_next_token_ids0.device, non_blocking=True)

    for step in range(1, draft_step):
        mem_start = (step - 1) * total_real_request_count
        step_mem_indexes = extra_mem_indexes[mem_start : mem_start + total_real_request_count]
        real_mem_start = 0
        for batch_index, model_input in enumerate(model_inputs):
            real_request_count = real_request_counts[batch_index]
            real_mem_indexes = step_mem_indexes[real_mem_start : real_mem_start + real_request_count]
            real_mem_start += real_request_count
            model_input.input_ids = draft_token_ids_by_batch[batch_index]
            model_input.mtp_draft_input_hiddens = draft_hiddens_by_batch[batch_index]
            model_input.mem_indexes = real_mem_indexes
            model_input.max_kv_seq_len = max_kv_seq_lens_by_batch[batch_index] + step
            model_input.total_token_num = model_input.batch_size * model_input.max_kv_seq_len

        draft_outputs = draft_model._microbatch_overlap_decode_cuda(*model_inputs)
        for batch_index, draft_output in enumerate(draft_outputs):
            draft_token_ids, draft_token_probs = generate_eagle_tokens(
                proposer,
                draft_output,
                map_draft_token_ids,
            )
            draft_token_ids_by_batch[batch_index] = draft_token_ids
            draft_hiddens_by_batch[batch_index] = draft_output.mtp_collector.spec_hidden
            draft_seq_lens_by_batch[batch_index].add_(1)
            real_request_count = real_request_counts[batch_index]
            proposal_row_start = proposal_row_offsets[batch_index]
            proposal_token_ids[proposal_row_start : proposal_row_start + real_request_count, step] = draft_token_ids[
                :real_request_count
            ]
            if proposal_schedule_scores is not None:
                proposal_schedule_scores[
                    proposal_row_start : proposal_row_start + real_request_count,
                    step,
                ] = draft_token_probs[:real_request_count]

    return EagleSpecProposal(
        token_ids=proposal_token_ids,
        extra_mem_indexes_cpu=[MtpMemIndexesToFree(mem_indexes_cpu=extra_mem_indexes_cpu)],
        schedule_scores=proposal_schedule_scores,
    )


def propose_next_dp_no_att_overlap(
    proposer: BaseDpOverlapProposer,
    target_model_input0: ModelInput,
    target_model_output0: ModelOutput,
    target_next_token_ids0: torch.Tensor,
    target_model_input1: ModelInput,
    target_model_output1: ModelOutput,
    target_next_token_ids1: torch.Tensor,
    real_request_num0: int,
    real_request_num1: int,
    accept_len: torch.Tensor,
    draft_step: int,
    get_draft_model: Callable[[int], object],
    map_draft_token_ids: Callable[[torch.Tensor], torch.Tensor],
) -> EagleSpecProposal:
    """将动态 verify 布局压缩为每请求一行，再执行双 microbatch No-Att draft。"""

    real_request_counts = (int(real_request_num0), int(real_request_num1))
    total_request_num = sum(real_request_counts)
    proposal_token_ids = target_next_token_ids0.new_empty((total_request_num, draft_step))
    proposal_schedule_scores = (
        torch.empty(
            (total_request_num, draft_step),
            dtype=torch.float32,
            device=target_next_token_ids0.device,
        )
        if proposer.enable_dynmaic_mtp
        else None
    )
    if draft_step == 0:
        return EagleSpecProposal(
            token_ids=proposal_token_ids,
            extra_mem_indexes_cpu=[],
            schedule_scores=proposal_schedule_scores,
        )

    assert target_next_token_ids0.shape == (target_model_input0.batch_size,)
    assert target_next_token_ids1.shape == (target_model_input1.batch_size,)
    (accept_len0, accept_len1, req_start_rows0, req_start_rows1,) = prepare_dp_eagle_overlap_decode_inputs(
        target_model_input0=target_model_input0,
        target_model_input1=target_model_input1,
        real_request_num0=real_request_num0,
        real_request_num1=real_request_num1,
        accept_len=accept_len,
    )

    model_inputs = [target_model_input0, target_model_input1]
    model_outputs = [target_model_output0, target_model_output1]
    target_token_ids_by_batch = [target_next_token_ids0, target_next_token_ids1]
    req_start_rows_by_batch = [req_start_rows0, req_start_rows1]
    accept_lens_by_batch = [accept_len0, accept_len1]
    draft_inputs = []
    draft_token_ids_by_batch = []
    draft_hiddens_by_batch = []
    for (model_input, model_output, token_ids, req_start_rows, batch_accept_len, request_num,) in zip(
        model_inputs,
        model_outputs,
        target_token_ids_by_batch,
        req_start_rows_by_batch,
        accept_lens_by_batch,
        real_request_counts,
    ):
        selected_rows = select_accepted_tail_rows(
            b_req_mtp_start_loc=req_start_rows,
            accept_len=batch_accept_len,
            input_ids=token_ids,
            hidden=model_output.mtp_collector.spec_hidden,
            b_req_idx=model_input.b_req_idx,
            b_mtp_index=model_input.b_mtp_index,
            b_seq_len=model_input.b_seq_len,
            mem_indexes=model_input.mem_indexes,
            b_shared_seq_len=model_input.b_shared_seq_len,
            b_shared_radix_node_id=model_input.b_shared_radix_node_id,
            b_position_delta=model_input.b_position_delta,
        )
        draft_input = copy.copy(model_input)
        draft_input.batch_size = request_num
        draft_input.input_ids = selected_rows.input_ids
        draft_input.b_req_idx = selected_rows.b_req_idx
        draft_input.b_mtp_index = selected_rows.b_mtp_index
        draft_input.b_seq_len = selected_rows.b_seq_len
        draft_input.mem_indexes = selected_rows.mem_indexes
        draft_input.b_shared_seq_len = selected_rows.b_shared_seq_len
        draft_input.b_shared_radix_node_id = selected_rows.b_shared_radix_node_id
        draft_input.b_position_delta = selected_rows.b_position_delta
        draft_input.mem_indexes_cpu = None
        draft_input.multimodal_params = [{"images": [], "audios": []} for _ in range(request_num)]
        draft_inputs.append(draft_input)
        draft_token_ids_by_batch.append(selected_rows.input_ids)
        draft_hiddens_by_batch.append(selected_rows.hidden)

    proposal_row_offsets = (0, real_request_counts[0])
    for step in range(draft_step):
        for batch_index, draft_input in enumerate(draft_inputs):
            draft_input.input_ids = draft_token_ids_by_batch[batch_index]
            draft_input.mtp_draft_input_hiddens = draft_hiddens_by_batch[batch_index]

        draft_outputs = get_draft_model(step)._microbatch_overlap_decode_cuda(*draft_inputs)
        for batch_index, draft_output in enumerate(draft_outputs):
            draft_token_ids, draft_token_probs = generate_eagle_tokens(
                proposer=proposer,
                model_output=draft_output,
                map_draft_token_ids=map_draft_token_ids,
            )
            draft_token_ids_by_batch[batch_index] = draft_token_ids
            draft_hiddens_by_batch[batch_index] = draft_output.mtp_collector.spec_hidden
            request_num = real_request_counts[batch_index]
            proposal_row_start = proposal_row_offsets[batch_index]
            proposal_token_ids[proposal_row_start : proposal_row_start + request_num, step] = draft_token_ids
            if proposal_schedule_scores is not None:
                proposal_schedule_scores[
                    proposal_row_start : proposal_row_start + request_num,
                    step,
                ] = draft_token_probs

    return EagleSpecProposal(
        token_ids=proposal_token_ids,
        extra_mem_indexes_cpu=[],
        schedule_scores=proposal_schedule_scores,
    )
