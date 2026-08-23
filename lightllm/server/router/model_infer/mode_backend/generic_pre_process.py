import torch
import numpy as np
from typing import List, Tuple
from lightllm.server.router.model_infer.infer_batch import InferReq, g_infer_context
from lightllm.common.basemodel.batch_objs import ModelInput

INT64_MAX = torch.iinfo(torch.int64).max


def prepare_prefill_inputs(req_objs: List[InferReq], is_chuncked_mode: bool) -> Tuple[ModelInput, List[InferReq]]:
    run_reqs = []
    total_token_num = 0
    input_ids = []
    b_req_idx = []
    b_seq_len = []
    b_q_seq_len = []
    batch_multimodal_params = []
    b_ready_cache_len = []
    b_mtp_index = []
    b_prefill_has_output = []
    b_is_decode_req = []

    for req in req_objs:
        run_reqs.append(req)
        batch_multimodal_params.append(req.multimodal_params)
        b_req_idx.append(req.req_idx)

        if is_chuncked_mode:
            input_token_ids = req.get_chuncked_input_token_ids()
        else:
            input_token_ids = req.get_input_token_ids()

        b_prefill_has_output.append(False if len(input_token_ids) < req.get_cur_total_len() else True)

        seq_len = len(input_token_ids)
        input_token_len = seq_len - req.cur_kv_len

        input_id = input_token_ids[req.cur_kv_len :]

        b_seq_len.append(seq_len)
        b_q_seq_len.append(input_token_len)
        input_ids.append(input_id)
        total_token_num += seq_len
        b_ready_cache_len.append(req.cur_kv_len)
        b_mtp_index.append(0)
        if hasattr(req, "is_decode_req_mixed_in_prefill"):
            b_is_decode_req.append(True)
            del req.is_decode_req_mixed_in_prefill
        else:
            b_is_decode_req.append(False)

    # DP 模式下某个 rank 可能没有本地请求。这里保留真实的 0 shape，
    # 推理所需的 dummy request 统一由 BaseModel 在执行前补齐。
    max_kv_seq_len = max(b_seq_len, default=0)
    max_cache_len = max(b_ready_cache_len, default=0)
    max_q_seq_len = max(b_q_seq_len, default=0)

    input_ids = np.concatenate(input_ids, dtype=np.int64) if input_ids else np.empty((0,), dtype=np.int64)
    input_ids = torch.tensor(input_ids, dtype=torch.int64, device="cpu")
    b_req_idx = torch.tensor(b_req_idx, dtype=torch.int32, device="cpu")
    b_seq_len = torch.tensor(b_seq_len, dtype=torch.int32, device="cpu")
    b_is_decode_req = torch.tensor(b_is_decode_req, dtype=torch.bool, device="cpu")
    b_mtp_index = torch.tensor(b_mtp_index, dtype=torch.int32, device="cpu")
    b_ready_cache_len = torch.tensor(b_ready_cache_len, dtype=torch.int32, device="cpu")
    b_q_seq_len = torch.tensor(b_q_seq_len, dtype=torch.int32, device="cpu")
    b_prefill_start_loc = b_q_seq_len.cumsum(dim=0, dtype=torch.int32) - b_q_seq_len

    # dynamic prompt cache 准备 token
    if g_infer_context.radix_cache is not None:
        g_infer_context.radix_cache.free_radix_cache_to_get_enough_token(input_ids.shape[0])
    mem_indexes = g_infer_context.req_manager.mem_manager.alloc(input_ids.shape[0])

    model_input = ModelInput(
        batch_size=b_seq_len.shape[0],
        total_token_num=total_token_num,
        max_q_seq_len=max_q_seq_len,
        max_kv_seq_len=max_kv_seq_len,
        max_cache_len=max_cache_len,
        input_ids=input_ids,
        mem_indexes_cpu=mem_indexes,
        b_req_idx=b_req_idx,
        b_mtp_index=b_mtp_index,
        b_seq_len=b_seq_len,
        b_is_decode_req=b_is_decode_req,
        b_ready_cache_len=b_ready_cache_len,
        b_prefill_start_loc=b_prefill_start_loc,
        is_prefill=True,
        b_prefill_has_output_cpu=b_prefill_has_output,
        multimodal_params=batch_multimodal_params,
    )

    return model_input, run_reqs


def prepare_decode_inputs(req_objs: List[InferReq]) -> Tuple[ModelInput, List[InferReq]]:
    run_reqs: List[InferReq] = []
    total_token_num = 0
    b_req_idx = []
    b_mtp_index = []
    b_seq_len = []
    b_q_seq_len = []
    multimodal_params = []
    for req in req_objs:
        run_reqs.append(req)
        b_req_idx.append(req.req_idx)
        seq_len = req.get_cur_total_len()
        assert req.cur_kv_len == seq_len - 1, f"{req.cur_kv_len} {seq_len}"
        b_seq_len.append(seq_len)
        b_q_seq_len.append(1)
        total_token_num += seq_len
        b_mtp_index.append(0)
        multimodal_params.append(req.multimodal_params)
        # process the draft tokens.
        for step in range(req.mtp_step):
            run_reqs.append(req)
            b_req_idx.append(req.req_idx)
            seq_len += 1
            b_seq_len.append(seq_len)
            total_token_num += seq_len
            b_mtp_index.append(step + 1)
            multimodal_params.append(req.multimodal_params)
            b_q_seq_len.append(1)

    # 空 DP rank 同样构建完整的 decode ModelInput；BaseModel 会在 token
    # gather 和 attention 初始化之前补入内部 dummy request。
    max_kv_seq_len = max(b_seq_len, default=0)
    max_q_seq_len = max(b_q_seq_len, default=1)

    b_req_idx = torch.tensor(b_req_idx, dtype=torch.int32, device="cpu")
    b_seq_len = torch.tensor(b_seq_len, dtype=torch.int32, device="cpu")
    b_mtp_index = torch.tensor(b_mtp_index, dtype=torch.int32, device="cpu")
    b_position_delta = build_b_position_delta(multimodal_params)

    b_shared_seq_len = torch.tensor(
        [req.get_radix_cache_shared_len() for req in run_reqs], dtype=torch.int32, device="cpu"
    )
    b_shared_radix_node_id = torch.tensor(
        [-1 if req.shared_kv_node is None else req.shared_kv_node.time_id % INT64_MAX for req in run_reqs],
        dtype=torch.int64,
        device="cpu",
    )

    # dynamic prompt cache 准备 token
    if g_infer_context.radix_cache is not None:
        g_infer_context.radix_cache.free_radix_cache_to_get_enough_token(b_seq_len.shape[0])
    mem_indexes = g_infer_context.req_manager.mem_manager.alloc(b_seq_len.shape[0])

    model_input = ModelInput(
        batch_size=b_seq_len.shape[0],
        total_token_num=total_token_num,
        max_q_seq_len=max_q_seq_len,
        max_kv_seq_len=max_kv_seq_len,
        input_ids=None,
        mem_indexes_cpu=mem_indexes,
        b_req_idx=b_req_idx,
        b_mtp_index=b_mtp_index,
        b_seq_len=b_seq_len,
        b_position_delta=b_position_delta,
        b_shared_seq_len=b_shared_seq_len,
        b_shared_radix_node_id=b_shared_radix_node_id,
        is_prefill=False,
        multimodal_params=multimodal_params,
    )
    return model_input, run_reqs


def overlap_prepare_decode_inputs(req_objs: List[InferReq]):
    """按请求把 decode batch 拆成两个允许为空的 microbatch。"""

    split_req_bound = (len(req_objs) + 1) // 2
    decode_reqs0 = req_objs[:split_req_bound]
    decode_reqs1 = req_objs[split_req_bound:]
    model_input0, run_reqs0 = prepare_decode_inputs(
        req_objs=decode_reqs0,
    )
    model_input1, run_reqs1 = prepare_decode_inputs(
        req_objs=decode_reqs1,
    )
    return model_input0, run_reqs0, decode_reqs0, model_input1, run_reqs1, decode_reqs1


def overlap_prepare_prefill_inputs(req_objs: List[InferReq]):
    """按当前 prefill token 负载把完整请求分配到两个 microbatch。

    请求不能跨 microbatch 拆分，否则同一个 ``InferReq`` 会在后处理阶段被
    重复更新。所有请求统一按照两侧已分配 token 数进行贪心均衡。这里不
    创建 HOLD 请求，空侧保留完整的 0-shape ``ModelInput``，执行阶段需要
    的 padding 由 BaseModel 统一处理。
    """

    req_input_token_nums = [len(req.get_chuncked_input_token_ids()) - req.cur_kv_len for req in req_objs]
    assert all(token_num > 0 for token_num in req_input_token_nums)

    left_token_num = 0
    right_token_num = 0
    left_reqs = []
    right_reqs = []
    for req, token_num in zip(req_objs, req_input_token_nums):
        if left_token_num <= right_token_num:
            left_reqs.append(req)
            left_token_num += token_num
        else:
            right_reqs.append(req)
            right_token_num += token_num

    model_input0, run_reqs0 = prepare_prefill_inputs(
        req_objs=left_reqs,
        is_chuncked_mode=True,
    )
    model_input1, run_reqs1 = prepare_prefill_inputs(
        req_objs=right_reqs,
        is_chuncked_mode=True,
    )
    return model_input0, run_reqs0, model_input1, run_reqs1


def build_b_position_delta(multimodal_params: List[dict]) -> torch.Tensor:
    b_position_delta = []
    for params in multimodal_params:
        position_delta = 0
        for image in params.get("images", []):
            grid_thwd = image.get("grid_thwd")
            if grid_thwd is not None:
                position_delta += grid_thwd[3]
        b_position_delta.append(position_delta)
    return torch.tensor(b_position_delta, dtype=torch.int32, device="cpu")
