import torch
import numpy as np
from typing import List, Tuple
from lightllm.server.router.model_infer.infer_batch import InferReq, g_infer_context
from lightllm.common.basemodel.infer_lock import g_infer_state_lock
from lightllm.common.basemodel.batch_objs import ModelInput
from lightllm.utils.envs_utils import (
    enable_diverse_mode_gqa_decode_fast_kernel,
    enable_triton_mtp_kernel,
    get_diverse_max_batch_shared_group_size,
    enable_dynamic_mtp_verify,
    get_env_start_args
)
from lightllm.utils.infer_utils import calculate_time


def prepare_prefill_inputs(req_objs: List[InferReq], is_chuncked_mode: bool) -> Tuple[ModelInput, List[InferReq]]:
    run_reqs = []
    total_token_num = 0
    prefix_total_token_num = 0
    input_ids = []
    b_req_idx = []
    b_seq_len = []
    b_q_seq_len = []
    batch_multimodal_params = []
    b_ready_cache_len = []
    b_mtp_index = []
    b_prefill_has_output = []

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
        prefix_total_token_num += req.cur_kv_len
        b_ready_cache_len.append(req.cur_kv_len)
        b_mtp_index.append(0)

    max_kv_seq_len = max(b_seq_len)
    max_cache_len = max(b_ready_cache_len)
    max_q_seq_len = max(b_q_seq_len)

    input_ids = np.concatenate(input_ids, dtype=np.int64)
    input_ids = torch.tensor(input_ids, dtype=torch.int64, device="cpu")
    b_req_idx = torch.tensor(b_req_idx, dtype=torch.int32, device="cpu")
    b_seq_len = torch.tensor(b_seq_len, dtype=torch.int32, device="cpu")
    b_mtp_index = torch.tensor(b_mtp_index, dtype=torch.int32, device="cpu")
    b_ready_cache_len = torch.tensor(b_ready_cache_len, dtype=torch.int32, device="cpu")
    b_q_seq_len = torch.tensor(b_q_seq_len, dtype=torch.int32, device="cpu")
    b_prefill_start_loc = b_q_seq_len.cumsum(dim=0, dtype=torch.int32) - b_q_seq_len

    # dynamic prompt cache 准备 token
    g_infer_state_lock.acquire()
    if g_infer_context.radix_cache is not None:
        g_infer_context.radix_cache.free_radix_cache_to_get_enough_token(input_ids.shape[0])
    mem_indexes = g_infer_context.req_manager.mem_manager.alloc(input_ids.shape[0])
    g_infer_state_lock.release()

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
        b_ready_cache_len=b_ready_cache_len,
        b_prefill_start_loc=b_prefill_start_loc,
        is_prefill=True,
        original_num_reqs=len(req_objs),
        b_prefill_has_output_cpu=b_prefill_has_output,
        prefix_total_token_num=prefix_total_token_num,
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
        # 动态 MTP 模式：使用动态 mtp_size 构建 batch
        for step in range(req.mtp_size):
            run_reqs.append(req)
            b_req_idx.append(req.req_idx)
            seq_len += 1
            b_seq_len.append(seq_len)
            total_token_num += seq_len
            b_mtp_index.append(step + 1)
            multimodal_params.append(req.multimodal_params)
            b_q_seq_len.append(1)

    max_kv_seq_len = max(b_seq_len)
    max_q_seq_len = max(b_q_seq_len)

    b_req_idx = torch.tensor(b_req_idx, dtype=torch.int32, device="cpu")
    b_seq_len = torch.tensor(b_seq_len, dtype=torch.int32, device="cpu")
    b_mtp_index = torch.tensor(b_mtp_index, dtype=torch.int32, device="cpu")

    # diverse mode 和 dynamic MTP mode 使用不同的 shared group 构建逻辑
    if enable_diverse_mode_gqa_decode_fast_kernel():
        b_shared_seq_len, b_mark_shared_group = build_diverse_shared_group_infos(run_reqs=run_reqs)
    elif enable_dynamic_mtp_verify() or enable_triton_mtp_kernel():
        # MTP 模式下，使用专门的 shared group 构建函数
        b_shared_seq_len = None  # MTP 模式不需要 b_shared_seq_len
        b_mark_shared_group = build_mtp_shared_group_infos(b_mtp_index=b_mtp_index)
    else:
        b_shared_seq_len = None
        b_mark_shared_group = None 

    # dynamic prompt cache 准备 token
    g_infer_state_lock.acquire()
    if g_infer_context.radix_cache is not None:
        g_infer_context.radix_cache.free_radix_cache_to_get_enough_token(b_seq_len.shape[0])
    mem_indexes = g_infer_context.req_manager.mem_manager.alloc(b_seq_len.shape[0])
    g_infer_state_lock.release()

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
        b_shared_seq_len=b_shared_seq_len,
        b_mark_shared_group=b_mark_shared_group,
        is_prefill=False,
        original_num_reqs=len(req_objs),
        multimodal_params=multimodal_params,
    )
    return model_input, run_reqs


def build_diverse_shared_group_infos(run_reqs: List[InferReq]) -> Tuple[torch.Tensor, torch.Tensor]:
    # b_shared_seq_len 和 b_mark_shared_group 只会在 diverse_mode 下的 decode 阶段真正被使用的参数,
    # 用于记录请求间的共享关系。
    # 举列说明:
    # b_shared_seq_len : [10, 10, 10, 11, 11, 11, 11]
    # b_mark_shared_group: [0, 0, 3, 0, 0, 0, 4]
    # b_mark_shared_group 中每一个不为0的位置都代表其与前面多少个请求形成一个共享前缀组。属于
    # 同一个共享前缀组的请求, 其在对应的 b_shared_seq_len 中的内容必然相同。某些模式可以利用这两个
    # 输入加速算子的运行。
    max_batch_shared_group_size = get_diverse_max_batch_shared_group_size()
    b_shared_seq_len = [req.get_radix_cache_shared_len() for req in run_reqs]
    b_mark_shared_group = []
    shared_nodes = [req.shared_kv_node for req in run_reqs]
    _current_group = []
    for node in shared_nodes:
        if not _current_group:
            _current_group.append(node)
        elif node == _current_group[-1]:
            _current_group.append(node)
        else:
            b_mark_shared_group.extend([0 for _ in range(len(_current_group))])
            b_mark_shared_group[-1] = len(_current_group)
            _current_group.clear()
            _current_group.append(node)

        if len(_current_group) == max_batch_shared_group_size:
            b_mark_shared_group.extend([0 for _ in range(len(_current_group))])
            b_mark_shared_group[-1] = len(_current_group)
            _current_group.clear()
    if _current_group:
        b_mark_shared_group.extend([0 for _ in range(len(_current_group))])
        b_mark_shared_group[-1] = len(_current_group)
        _current_group.clear()

    assert len(b_mark_shared_group) == len(run_reqs)
    # 如果一个 shared group 的长度为1， 则将其共享长度强制修改为0， 避免无效计算，提升
    # 算子执行效率。
    b_shared_seq_len = [
        0 if group_size == 1 else shared_len for shared_len, group_size in zip(b_shared_seq_len, b_mark_shared_group)
    ]
    b_shared_seq_len = torch.tensor(b_shared_seq_len, dtype=torch.int32, device="cpu")
    b_mark_shared_group = torch.tensor(b_mark_shared_group, dtype=torch.int32, device="cpu")
    return b_shared_seq_len, b_mark_shared_group


def build_mtp_shared_group_infos(
    b_mtp_index: torch.Tensor,
) -> torch.Tensor:
    """
    构建 b_mark_shared_group（支持 max shared group size 截断）

    规则：
    - 组边界由 b_mtp_index == 0 定义（每个 0 是新组起点）
    - 默认在每个组的最后一个位置写入组大小，其余位置为 0
    - 若组大小 > max_batch_shared_group_size，则切分为多个连续子组
      （每个子组大小 <= max_batch_shared_group_size），并在每个子组末尾写入子组大小

    返回：
    - res: shape [batch_size], int32
      0 表示非子组末尾；N>=1 表示该子组大小
    """
    device = b_mtp_index.device
    batch_size = b_mtp_index.numel()
    res = torch.zeros(batch_size, dtype=torch.int32, device=device)

    if batch_size == 0:
        return res

    max_size = int(get_diverse_max_batch_shared_group_size())
    if max_size <= 0:
        max_size = 1

    # 1) 找每个原始组起点（b_mtp_index == 0）
    is_start = (b_mtp_index == 0)
    starts = torch.nonzero(is_start, as_tuple=False).flatten()

    # 防御：若输入不规范（没有 0），则把整个 batch 当一组
    if starts.numel() == 0:
        starts = torch.zeros(1, dtype=torch.long, device=device)

    # 2) 原始组长度
    # ends = [starts[1]-1, starts[2]-1, ..., batch_size-1]
    ends = torch.empty_like(starts)
    if starts.numel() > 1:
        ends[:-1] = starts[1:] - 1
    ends[-1] = batch_size - 1
    lens = ends - starts + 1  # [num_groups]

    # 3) 每个原始组需要切成多少个子组
    # sub_cnt[g] = ceil(lens[g] / max_size)
    sub_cnt = (lens + max_size - 1) // max_size  # [num_groups], int64

    # 4) 展平到“子组级别”（全向量化）
    num_groups = starts.numel()
    group_ids = torch.repeat_interleave(
        torch.arange(num_groups, device=device, dtype=torch.long),
        sub_cnt
    )  # [num_subgroups]

    # 子组在原始组内的序号：0,1,2,...
    sub_prefix = torch.cumsum(sub_cnt, dim=0) - sub_cnt               # 每个组的子组起始全局序号
    sub_prefix_rep = torch.repeat_interleave(sub_prefix, sub_cnt)
    global_sub_idx = torch.arange(group_ids.numel(), device=device, dtype=torch.long)
    sub_idx_in_group = global_sub_idx - sub_prefix_rep                # [num_subgroups]

    # 5) 计算每个子组大小与末尾位置
    # size = min(max_size, lens - sub_idx*max_size)
    rem = lens[group_ids] - sub_idx_in_group * max_size
    sub_size = torch.minimum(rem, torch.full_like(rem, max_size))     # [num_subgroups]
    sub_end = starts[group_ids] + sub_idx_in_group * max_size + sub_size - 1

    # 6) 写回结果：子组末尾位置写子组大小
    res[sub_end] = sub_size.to(torch.int32)
    return res