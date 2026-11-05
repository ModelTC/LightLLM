"""Batch layout, execution mode selection and output handling owned by Runner."""

import copy
import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput, PostLayerOutput
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.common.basemodel.triton_kernel.gather_token_id import gather_token, gather_token_prefill_decode_mixed
from lightllm.utils.custom_kernel_utis import pad2dim_tensor_to_new_batch
from lightllm.common.execution.device.cuda import CudaRuntime


def pad_decode(model_input: ModelInput, new_batch_size: int, hold_request_id: int):
    if model_input.batch_size == new_batch_size:
        return model_input

    assert model_input.batch_size <= new_batch_size

    padded_batch_size = new_batch_size - model_input.batch_size
    new_model_input = copy.copy(model_input)
    new_model_input.batch_size = new_batch_size
    new_model_input.total_token_num += padded_batch_size * 2
    new_model_input.max_kv_seq_len = max(2, model_input.max_kv_seq_len)
    new_model_input.input_ids = F.pad(new_model_input.input_ids, (0, padded_batch_size), mode="constant", value=1)
    new_model_input.b_req_idx = F.pad(
        new_model_input.b_req_idx, (0, padded_batch_size), mode="constant", value=hold_request_id
    )
    new_model_input.b_mtp_index = F.pad(new_model_input.b_mtp_index, (0, padded_batch_size), mode="constant", value=0)
    new_model_input.b_seq_len = F.pad(new_model_input.b_seq_len, (0, padded_batch_size), mode="constant", value=2)
    if new_model_input.b_position_delta is not None:
        new_model_input.b_position_delta = F.pad(
            new_model_input.b_position_delta, (0, padded_batch_size), mode="constant", value=0
        )
    new_model_input.multimodal_params = new_model_input.multimodal_params + [
        {"images": [], "audios": []} for _ in range(padded_batch_size)
    ]

    new_model_input.b_shared_seq_len = F.pad(
        new_model_input.b_shared_seq_len, (0, padded_batch_size), mode="constant", value=0
    )
    new_model_input.b_shared_radix_node_id = F.pad(
        new_model_input.b_shared_radix_node_id, (0, padded_batch_size), mode="constant", value=-1
    )

    # 特殊模型，特殊模式的特殊变量的特殊 padding
    if new_model_input.mtp_draft_input_hiddens is not None:
        new_model_input.mtp_draft_input_hiddens = pad2dim_tensor_to_new_batch(
            input=new_model_input.mtp_draft_input_hiddens,
            new_batch_size=new_batch_size,
        )

    new_model_input.check_input()
    return new_model_input


def pad_prefill(model_input: ModelInput, new_handle_token_num: int, hold_request_id: int):
    handle_token_num = model_input.input_ids.shape[0]
    if handle_token_num == new_handle_token_num:
        return model_input

    assert handle_token_num < new_handle_token_num

    padded_token_num = new_handle_token_num - handle_token_num
    assert padded_token_num > 0
    new_model_input = copy.copy(model_input)
    new_model_input.batch_size = model_input.batch_size + 1
    new_model_input.total_token_num += padded_token_num
    new_model_input.max_q_seq_len = max(padded_token_num, model_input.max_q_seq_len)
    new_model_input.max_kv_seq_len = max(padded_token_num, model_input.max_kv_seq_len)
    new_model_input.max_cache_len = max(0, model_input.max_cache_len)
    new_model_input.input_ids = F.pad(new_model_input.input_ids, (0, padded_token_num), mode="constant", value=1)
    new_model_input.b_req_idx = F.pad(new_model_input.b_req_idx, (0, 1), mode="constant", value=hold_request_id)
    new_model_input.b_mtp_index = F.pad(new_model_input.b_mtp_index, (0, 1), mode="constant", value=0)
    new_model_input.b_seq_len = F.pad(new_model_input.b_seq_len, (0, 1), mode="constant", value=padded_token_num)
    new_model_input.b_ready_cache_len = F.pad(new_model_input.b_ready_cache_len, (0, 1), mode="constant", value=0)
    new_model_input.b_is_decode_req = F.pad(new_model_input.b_is_decode_req, (0, 1), mode="constant", value=False)
    b_q_seq_len = new_model_input.b_seq_len - new_model_input.b_ready_cache_len
    new_model_input.b_prefill_start_loc = b_q_seq_len.cumsum(dim=0, dtype=torch.int32) - b_q_seq_len
    # 构建新的list, 使用 append 可能会让外面使用的数组引用发生变化，导致错误。
    new_model_input.b_prefill_has_output_cpu = [e for e in new_model_input.b_prefill_has_output_cpu] + [False]

    new_model_input.multimodal_params = [e for e in new_model_input.multimodal_params] + [{"images": [], "audios": []}]

    # 特殊模型，特殊模式的特殊变量的特殊 padding
    if new_model_input.mtp_draft_input_hiddens is not None:
        new_model_input.mtp_draft_input_hiddens = pad2dim_tensor_to_new_batch(
            input=new_model_input.mtp_draft_input_hiddens,
            new_batch_size=new_handle_token_num,
        )

    new_model_input.check_input()
    return new_model_input


def make_output(post_output: PostLayerOutput, infer_state: InferStateInfo) -> ModelOutput:
    output = ModelOutput(
        logits=post_output.logits.contiguous(),
        mtp_collector=infer_state.hidden_collector.finish_output(infer_state=infer_state),
        prompt_logics=infer_state.prompt_logics,
        logits_token_ids=post_output.logits_token_ids,
    )
    return output


def unpad_decode(model_output: ModelOutput, origin_batch_size: int):
    padded_batch_size = model_output.logits.shape[0]
    if padded_batch_size == origin_batch_size:
        return model_output
    new_model_output = copy.copy(model_output)
    new_model_output.logits = new_model_output.logits[0:origin_batch_size]
    if new_model_output.logits_token_ids is not None:
        new_model_output.logits_token_ids = new_model_output.logits_token_ids[:origin_batch_size]
    new_model_output.mtp_collector = model_output.mtp_collector.unpad_decode(
        padded_batch_size=padded_batch_size,
        origin_batch_size=origin_batch_size,
    )
    return new_model_output


def unpad_prefill(padded_model_output: ModelOutput, origin_handle_token_num: int, origin_batch_size: int):
    new_model_output = copy.copy(padded_model_output)
    # logits 始终只对应每个请求最后一个位置，移除 padding 的 req 对应的行。
    new_model_output.logits = new_model_output.logits[0:origin_batch_size]
    if new_model_output.logits_token_ids is not None:
        new_model_output.logits_token_ids = new_model_output.logits_token_ids[:origin_batch_size]
    new_model_output.mtp_collector = padded_model_output.mtp_collector.unpad_prefill(
        origin_handle_token_num=origin_handle_token_num
    )
    # prompt_logics 保存整个 prefill 阶段所有 token 位置的 logits，
    # 按实际处理的 token 数量裁剪掉 padding 部分（仅 return_all_prompt_logics 模式下非空）。
    if new_model_output.prompt_logics is not None:
        new_model_output.prompt_logics = new_model_output.prompt_logics[0:origin_handle_token_num]

    return new_model_output


@dataclass(frozen=True)
class BatchPolicy:
    decode_alignment: int = 1
    sequence_parallel_alignment: int = 1
    mixed_prefill: bool = False

    def __post_init__(self):
        if min(self.decode_alignment, self.sequence_parallel_alignment) <= 0:
            raise ValueError("batch alignment must be positive")

    def align_decode(self, rows):
        alignment = math.lcm(self.decode_alignment, self.sequence_parallel_alignment)
        return ((max(1, rows) + alignment - 1) // alignment) * alignment

    def align_prefill(self, tokens):
        alignment = self.sequence_parallel_alignment
        return ((max(1, tokens) + alignment - 1) // alignment) * alignment


@dataclass(frozen=True)
class BatchProgram:
    create_state: Callable
    initialize_state: Callable
    prefill: Callable
    decode: Callable
    overlap_prefill: Callable
    overlap_decode: Callable


class BatchRunner:
    def __init__(
        self, policy, program, requests, graphs, *, decode_padding=None, clear_transfer=None, runtime=CudaRuntime
    ):
        self.policy, self.program = policy, program
        self.requests, self.graphs = requests, graphs
        self.decode_padding = decode_padding
        self.clear_transfer = clear_transfer
        self.runtime = runtime

    def prepare_input(self, batch, *, move_to_device=False):
        if move_to_device:
            batch.to_device(self.requests.req_to_token_indexs.device)
        if batch.is_prefill:
            if self.policy.mixed_prefill and batch.input_ids.numel():
                gather_token_prefill_decode_mixed(
                    input_ids=batch.input_ids,
                    req_to_next_token_ids=self.requests.req_sampling_params_manager.req_to_next_token_ids,
                    b_req_idx=batch.b_req_idx,
                    b_mtp_index=batch.b_mtp_index,
                    b_is_decode_req=batch.b_is_decode_req,
                    b_prefill_start_loc=batch.b_prefill_start_loc,
                )
        elif batch.input_ids is None:
            if batch.batch_size:
                batch.input_ids = gather_token(
                    req_to_next_token_ids=self.requests.req_sampling_params_manager.req_to_next_token_ids,
                    b_req_idx=batch.b_req_idx,
                    b_mtp_index=batch.b_mtp_index,
                )
            else:
                batch.input_ids = torch.empty(0, dtype=torch.int64, device=batch.b_req_idx.device)

    def _initialize(self, state):
        self.program.initialize_state(state)
        state.init_att_state()

    def _ready_event(self):
        event = self.runtime.create_event()
        event.record()
        return event

    def run_prefill(self, *inputs, prepared=False):
        if len(inputs) not in (1, 2):
            raise ValueError("prefill requires one or two microbatches")
        overlap = len(inputs) == 2
        originals, states = [], []
        ready = None
        for lane, batch in enumerate(inputs):
            if not batch.is_prefill:
                raise ValueError("prefill received a decode batch")
            if not prepared:
                self.prepare_input(batch)
            tokens, rows = batch.input_ids.shape[0], batch.batch_size
            originals.append((tokens, rows))
            count = self.policy.align_prefill(tokens)
            graph = self.graphs.prefill
            if not overlap and graph is not None and graph.can_run(handle_token_num=count):
                count = graph.find_closest_graph_handle_token_num(handle_token_num=count)
            padded = pad_prefill(batch, count, self.requests.HOLD_REQUEST_ID)
            state = self.program.create_state(padded, lane)
            # Publish the token-index mapping before model compute, as the single-batch path requires.
            if not overlap:
                ready = self._ready_event()
            self._initialize(state)
            states.append(state)
        if overlap:
            ready = self._ready_event()
            outputs = self.program.overlap_prefill(*states)
        else:
            outputs = (self.program.prefill(states[0]),)
        outputs = tuple(unpad_prefill(out, tokens, rows) for out, (tokens, rows) in zip(outputs, originals))
        if overlap and self.clear_transfer is not None:
            self.clear_transfer()
        for output in outputs:
            output.prefill_mem_indexes_ready_event = ready
        return outputs if overlap else outputs[0]

    def run_decode(self, *inputs, prepared=False):
        if len(inputs) not in (1, 2):
            raise ValueError("decode requires one or two microbatches")
        for batch in inputs:
            if batch.is_prefill:
                raise ValueError("decode received a prefill batch")
            if not prepared:
                self.prepare_input(batch)
        rows = tuple(batch.batch_size for batch in inputs)
        size = self.policy.align_decode(max(rows))
        # Padding creates HOLD rows of length 2, which participate in Graph eligibility.
        step = self.graphs.plan_decode(size, max(2, *(batch.max_kv_seq_len for batch in inputs)))
        states = []
        for lane, batch in enumerate(inputs):
            padded = (
                pad_decode(batch, step.batch_size, self.requests.HOLD_REQUEST_ID)
                if self.decode_padding is None
                else self.decode_padding(batch, step.batch_size)
            )
            if padded.batch_size != step.batch_size:
                raise ValueError("model padding changed the selected execution capacity")
            state = self.program.create_state(padded, lane)
            step.prepare(state)
            self._initialize(state)
            states.append(state)
        overlap = len(inputs) == 2
        forward = self.program.overlap_decode if overlap else self.program.decode
        outputs = self.graphs.run_decode(step, forward, *states)
        if not overlap:
            return unpad_decode(outputs, rows[0])
        return tuple(unpad_decode(out, count) for out, count in zip(outputs, rows))
