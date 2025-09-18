import torch
import copy
from typing import List, Tuple
from lightllm.server.router.model_infer.infer_batch import InferReq
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from lightllm.server.router.model_infer.infer_batch import g_infer_state_lock
from lightllm.common.basemodel.batch_objs import ModelInput
from lightllm.common.basemodel.triton_kernel.gen_mtp_prefill_params import gen_mtp_new_input_ids


def prepare_mtp_prefill_inputs(
    model_input: ModelInput, b_next_token_ids: torch.Tensor, deepseekv3_mtp_draft_input_hiddens: torch.Tensor
):
    new_model_input = copy.copy(model_input)
    new_input_ids = gen_mtp_new_input_ids(
        input_ids=model_input.input_ids,
        b_next_token_ids=b_next_token_ids,
        b_seq_len=model_input.b_seq_len,
        b_ready_cache_len=model_input.b_ready_cache_len,
    )
    new_model_input.input_ids = new_input_ids
    new_model_input.deepseekv3_mtp_draft_input_hiddens = deepseekv3_mtp_draft_input_hiddens
    return new_model_input


def prepare_eagle_draft_decode_inputs(req_objs: List[InferReq]) -> Tuple[ModelInput, List[InferReq]]:
    run_reqs = []
    total_token_num = 0
    max_len_in_batch = 0
    b_req_idx = []
    b_mtp_index = []
    b_seq_len = []
    mtp_step = 0

    for req in req_objs:
        run_reqs.append(req)
        b_req_idx.append(req.req_idx)
        seq_len = req.get_cur_total_len()
        assert req.cur_kv_len == seq_len - 1
        b_seq_len.append(seq_len)
        total_token_num += seq_len
        max_len_in_batch = max(max_len_in_batch, seq_len)
        b_mtp_index.append(0)
        mtp_step = req.mtp_step  # 开启mtp 模式时，每个请求的mtp一定相同。

    b_req_idx = torch.tensor(b_req_idx, dtype=torch.int32, device="cpu")
    b_seq_len = torch.tensor(b_seq_len, dtype=torch.int32, device="cpu")
    b_mtp_index = torch.tensor(b_mtp_index, dtype=torch.int32, device="cpu")

    # draft 模型一次要decode多步，提前分配所需要的mem_index
    need_tokens = b_seq_len.shape[0] * (mtp_step + 1)

    # dynamic prompt cache 准备 token
    g_infer_state_lock.acquire()
    if g_infer_context.radix_cache is not None:
        g_infer_context.radix_cache.free_radix_cache_to_get_enough_token(need_tokens)
    mem_indexes = g_infer_context.req_manager.mem_manager.alloc(need_tokens)
    g_infer_state_lock.release()

    model_input = ModelInput(
        batch_size=b_seq_len.shape[0],
        total_token_num=total_token_num,
        max_len_in_batch=max_len_in_batch,
        input_ids=None,
        mem_indexes_cpu=mem_indexes,
        b_req_idx=b_req_idx,
        b_mtp_index=b_mtp_index,
        b_seq_len=b_seq_len,
        is_prefill=False,
    )
    return model_input, run_reqs
