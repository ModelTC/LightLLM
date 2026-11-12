import torch
import numpy as np
from typing import List, Tuple
from lightllm.server.router.model_infer.infer_batch import InferReq, g_infer_context
from lightllm.common.basemodel.batch_objs import ModelInput

INT64_MAX = torch.iinfo(torch.int64).max


def prepare_prefill_inputs(req_objs: List[InferReq], is_chuncked_mode: bool) -> Tuple[ModelInput, List[InferReq]]:
    lookback_depth = g_infer_context.backend.model.token_lookback_depth
    token_lookback = []
    run_reqs = []
    total_token_num = 0
    input_ids = []
    b_req_idx = []
    b_seq_len = []
    b_q_seq_len = []
    batch_multimodal_params = []
    b_ready_cache_len = []
    b_prefill_commit_start = []
    replay_window = g_infer_context.backend.model.prefill_replay_window
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
        start = req.get_prefill_input_start(seq_len) if replay_window else req.cur_kv_len
        input_token_len = seq_len - start
        if lookback_depth:
            if start < req.cur_kv_len:
                history = (req.req_idx, start, input_token_ids[max(0, start - lookback_depth) : start].tolist())
                req.token_history_ready = True
                if start == 0:
                    history = None
            else:
                history = req.get_token_lookback(lookback_depth)
            if history is not None:
                token_lookback.append(history)

        input_id = input_token_ids[start:]

        b_seq_len.append(seq_len)
        b_q_seq_len.append(input_token_len)
        input_ids.append(input_id)
        total_token_num += seq_len
        b_ready_cache_len.append(start)
        b_prefill_commit_start.append(req.cur_kv_len)
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
    has_replay = b_ready_cache_len != b_prefill_commit_start
    prefill_tail_token_num = None
    if replay_window and max_q_seq_len > replay_window and len(b_q_seq_len) > 1:
        prefill_tail_token_num = sum(min(length, replay_window) for length in b_q_seq_len)

    input_ids = np.concatenate(input_ids, dtype=np.int64) if input_ids else np.empty((0,), dtype=np.int64)
    input_ids = torch.from_numpy(input_ids)
    b_req_idx = torch.tensor(b_req_idx, dtype=torch.int32, device="cpu")
    b_seq_len = torch.tensor(b_seq_len, dtype=torch.int32, device="cpu")
    b_is_decode_req = torch.tensor(b_is_decode_req, dtype=torch.bool, device="cpu")
    b_mtp_index = torch.tensor(b_mtp_index, dtype=torch.int32, device="cpu")
    b_ready_cache_len = torch.tensor(b_ready_cache_len, dtype=torch.int32, device="cpu")
    b_q_seq_len = torch.tensor(b_q_seq_len, dtype=torch.int32, device="cpu")
    b_prefill_start_loc = b_q_seq_len.cumsum(dim=0, dtype=torch.int32) - b_q_seq_len

    model_input = ModelInput(
        batch_size=b_seq_len.shape[0],
        total_token_num=total_token_num,
        max_q_seq_len=max_q_seq_len,
        max_kv_seq_len=max_kv_seq_len,
        max_cache_len=max_cache_len,
        input_ids=input_ids,
        b_req_idx=b_req_idx,
        b_mtp_index=b_mtp_index,
        b_seq_len=b_seq_len,
        b_is_decode_req=b_is_decode_req,
        b_ready_cache_len=b_ready_cache_len,
        b_prefill_start_loc=b_prefill_start_loc,
        is_prefill=True,
        b_prefill_has_output_cpu=b_prefill_has_output,
        multimodal_params=batch_multimodal_params,
        token_lookback=token_lookback or None,
        b_prefill_commit_start=(torch.tensor(b_prefill_commit_start, dtype=torch.int32) if has_replay else None),
        prefill_tail_token_num=prefill_tail_token_num,
    )

    return model_input, run_reqs


def prepare_decode_inputs(req_objs: List[InferReq]) -> Tuple[ModelInput, List[InferReq]]:
    lookback_depth = g_infer_context.backend.model.token_lookback_depth
    token_lookback = []
    run_reqs: List[InferReq] = []
    b_req_idx = []
    b_mtp_index = []
    b_seq_len = []
    multimodal_params = []
    b_position_delta = []
    b_shared_seq_len = []
    b_shared_radix_node_id = []
    for req in req_objs:
        if lookback_depth:
            history = req.get_token_lookback(lookback_depth)
            if history is not None:
                token_lookback.append(history)

        seq_len = req.get_cur_total_len()
        assert req.cur_kv_len == seq_len - 1, f"{req.cur_kv_len} != {seq_len}"
        params = req.multimodal_params
        position_delta = 0
        for image in params.get("images", []):
            grid_thwd = image.get("grid_thwd")
            if grid_thwd is not None:
                position_delta += grid_thwd[3]
        shared_seq_len = req.get_radix_cache_shared_len()
        node = req.shared_kv_node
        node_id = -1 if node is None else node.time_id % INT64_MAX

        run_reqs.append(req)
        b_req_idx.append(req.req_idx)
        b_seq_len.append(seq_len)
        b_mtp_index.append(0)
        multimodal_params.append(params)
        b_position_delta.append(position_delta)
        b_shared_seq_len.append(shared_seq_len)
        b_shared_radix_node_id.append(node_id)

        # Draft rows share the anchor's request metadata.
        for step in range(1, req.mtp_step + 1):
            run_reqs.append(req)
            b_req_idx.append(req.req_idx)
            b_seq_len.append(seq_len + step)
            b_mtp_index.append(step)
            multimodal_params.append(params)
            b_position_delta.append(position_delta)
            b_shared_seq_len.append(shared_seq_len)
            b_shared_radix_node_id.append(node_id)

    # 空 DP rank 同样构建完整的 decode ModelInput；BaseModel 会在 token
    # gather 和 attention 初始化之前补入内部 dummy request。
    total_token_num = sum(b_seq_len)
    max_kv_seq_len = max(b_seq_len, default=0)
    b_req_idx = torch.tensor(b_req_idx, dtype=torch.int32, device="cpu")
    b_seq_len = torch.tensor(b_seq_len, dtype=torch.int32, device="cpu")
    b_mtp_index = torch.tensor(b_mtp_index, dtype=torch.int32, device="cpu")
    b_position_delta = torch.tensor(b_position_delta, dtype=torch.int32, device="cpu")
    b_shared_seq_len = torch.tensor(b_shared_seq_len, dtype=torch.int32, device="cpu")
    b_shared_radix_node_id = torch.tensor(b_shared_radix_node_id, dtype=torch.int64, device="cpu")

    model_input = ModelInput(
        batch_size=b_seq_len.shape[0],
        total_token_num=total_token_num,
        max_q_seq_len=1,
        max_kv_seq_len=max_kv_seq_len,
        input_ids=None,
        b_req_idx=b_req_idx,
        b_mtp_index=b_mtp_index,
        b_seq_len=b_seq_len,
        b_position_delta=b_position_delta,
        b_shared_seq_len=b_shared_seq_len,
        b_shared_radix_node_id=b_shared_radix_node_id,
        is_prefill=False,
        multimodal_params=multimodal_params,
        token_lookback=token_lookback or None,
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
