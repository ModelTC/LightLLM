#!/usr/bin/env python3
"""
静态测试脚本：对比无MTP、静态Eagle MTP和动态Eagle MTP的吞吐差异
严格按照服务端 chunked_prefill/impl.py 实现复刻

Usage:
    python benchmark_mtp_throughput.py \
        --model_dir /path/to/model \
        --mtp_draft_model_dir /path/to/draft/model \
        --dataset /path/to/dataset.json \
        --batch_sizes 1 4 8 16 \
        --output_lens 128 512 \
        --tp 2
"""

import os
import sys
import json
import time
import argparse
import torch
import numpy as np
from multiprocessing import Queue
import multiprocessing
from typing import List, Tuple, Dict, Optional
from transformers import PretrainedConfig
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lightllm.utils.dist_utils import init_distributed_env, get_current_rank_in_dp
from lightllm.utils.envs_utils import get_env_start_args, set_env_start_args
from lightllm.server.api_cli import make_argument_parser
from lightllm.models import get_model
from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.models.qwen3_eagle.model import Qwen3EagleModel
from lightllm.utils.log_utils import init_logger
from transformers import AutoTokenizer
from lightllm.common.basemodel.triton_kernel.mtp_utils import mtp_verify, mtp_scatter_next_token_ids
from lightllm.common.basemodel.triton_kernel.gen_mtp_prefill_params import gen_mtp_new_input_ids

logger = init_logger(__name__)


@dataclass
class BenchmarkResult:
    """存储单次benchmark结果"""
    batch_size: int
    input_len: int
    output_len: int
    prefill_time_ms: float
    prefill_throughput: float
    decode_time_ms: float
    decode_throughput: float
    total_time_ms: float
    overall_throughput: float
    mode: str  # "no_mtp", "static_mtp", "dynamic_mtp"
    avg_accepted_steps: float = 0.0


def load_dataset_samples(dataset_path: str, num_samples: int, tokenizer) -> List[List[int]]:
    """从数据集加载样本，返回 input_ids 列表"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    samples = []
    for i, data in enumerate(dataset[:num_samples * 2]):
        if len(samples) >= num_samples:
            break

        conversations = data.get("conversations", [])
        if len(conversations) < 2:
            continue

        prompt = None
        for turn in conversations:
            role = turn.get("from", "").lower()
            value = turn.get("value", "")
            if role in ["human", "user"] and not prompt:
                prompt = value
                break

        if not prompt:
            continue

        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_tokens) < 4:
            continue

        samples.append(prompt_tokens)

    return samples[:num_samples]


def compute_dynamic_mtp_size(
    main_probs: torch.Tensor,
    max_mtp_size: int,
    draft_probs: List[torch.Tensor],
) -> List[int]:
    """
    根据 prob 分布计算每个请求的动态 MTP 验证长度
    与服务端 compute_dynamic_mtp_size 保持一致
    """
    import random
    batch_size = main_probs.shape[0]
    dynamic_mtp_sizes = []

    for i in range(batch_size):
        mtp_size = 0
        for step in range(max_mtp_size):
            if step < len(draft_probs):
                step_probs = draft_probs[step][i]
                max_prob = torch.max(step_probs).item()
                r = random.random()
                if r < max_prob or max_prob > 0.5:
                    mtp_size += 1
                else:
                    break
        dynamic_mtp_sizes.append(mtp_size)

    return dynamic_mtp_sizes


def build_mtp_shared_group_infos(
    b_mtp_index: torch.Tensor,
) -> torch.Tensor:
    """
    为 MTP 动态验证模式构建 b_mark_shared_group 信息
    与服务端 build_mtp_shared_group_infos 保持一致

    MTP 模式下，每个原始请求会扩展成 (1 + mtp_size) 个请求，这些请求共享相同的 KV 前缀
    例如：mtp_size=3 时，4 个请求 [A0, A1, A2, A3] 共享前缀，标记为 [0, 0, 0, 4]
    """
    batch_size = b_mtp_index.shape[0]
    b_mark_shared_group = [0] * batch_size

    group_start = 0
    for i in range(1, batch_size + 1):
        if i == batch_size or b_mtp_index[i] == 0:
            group_size = i - group_start
            b_mark_shared_group[i - 1] = group_size
            group_start = i

    return torch.tensor(b_mark_shared_group, dtype=torch.int32, device="cpu")


def prepare_mtp_prefill_inputs_for_benchmark(
    model_input: ModelInput,
    b_next_token_ids: torch.Tensor,
    mtp_draft_input_hiddens: torch.Tensor,
) -> ModelInput:
    """
    模拟服务端的 prepare_mtp_prefill_inputs 函数
    """
    import copy
    new_model_input = copy.copy(model_input)
    new_input_ids = gen_mtp_new_input_ids(
        input_ids=model_input.input_ids,
        b_next_token_ids=b_next_token_ids,
        b_seq_len=model_input.b_seq_len,
        b_ready_cache_len=model_input.b_ready_cache_len,
    )
    new_model_input.input_ids = new_input_ids
    new_model_input.mtp_draft_input_hiddens = mtp_draft_input_hiddens
    return new_model_input


def init_mtp_model(args, kvargs, main_model):
    """初始化 MTP draft model（Eagle模式只有一个draft model）"""
    args.mtp_draft_model_dir = args.mtp_draft_model_dir[0] if isinstance(args.mtp_draft_model_dir, list) else args.mtp_draft_model_dir

    os.environ["DISABLE_CHECK_MAX_LEN_INFER"] = "1"

    mtp_model_cfg, _ = PretrainedConfig.get_config_dict(args.mtp_draft_model_dir)

    mtp_model_kvargs = kvargs.copy()
    mtp_model_kvargs.update({
        "weight_dir": args.mtp_draft_model_dir,
        "max_total_token_num": main_model.mem_manager.size,
        "disable_chunked_prefill": True,
        "mtp_mode": args.mtp_mode,
        "main_model": main_model,
        "mtp_previous_draft_models": [],
        "is_mtp_draft_model": True,
        "run_mode": "normal",
    })

    model_type = mtp_model_cfg.get("model_type", "")

    if model_type == "llama" and args.mtp_mode == "eagle3":
        draft_model = Qwen3EagleModel(mtp_model_kvargs)
    else:
        raise ValueError(f"Unsupported model_type {model_type} or mtp_mode {args.mtp_mode}")

    return draft_model


def run_benchmark_no_mtp(
    args,
    input_ids_list: List[List[int]],
    output_len: int,
    batch_size: int,
    main_model,
    warmup: bool = False
) -> BenchmarkResult:
    """无 MTP 模式的 benchmark"""
    rank_id = get_current_rank_in_dp()
    actual_batch_size = min(batch_size, len(input_ids_list))
    input_lens = [len(input_ids_list[i]) for i in range(actual_batch_size)]
    max_input_len = max(input_lens)

    # Padding - 与prepare_prefill_inputs一致
    input_ids = []
    for i in range(actual_batch_size):
        ids = input_ids_list[i]
        input_ids.extend(ids)

    test_data = torch.tensor(input_ids, dtype=torch.long, device="cpu")

    # Prefill
    torch.cuda.synchronize()
    prefill_start = time.time()

    b_req_idx = torch.tensor(
        [main_model.req_manager.alloc() for _ in range(actual_batch_size)],
        dtype=torch.int32, device="cpu"
    )
    b_seq_len = torch.tensor(input_lens, dtype=torch.int32, device="cpu")
    b_ready_cache_len = torch.zeros(actual_batch_size, dtype=torch.int32, device="cpu")

    total_token_num = sum(input_lens)

    # 分配内存索引
    mem_indexes = main_model.req_manager.mem_manager.alloc(test_data.shape[0])
    mem_indexes_cpu = mem_indexes

    b_q_seq_len = torch.tensor(input_lens, dtype=torch.int32, device="cpu")
    b_prefill_start_loc = b_q_seq_len.cumsum(dim=0, dtype=torch.int32) - b_q_seq_len

    model_input = ModelInput(
        batch_size=actual_batch_size,
        total_token_num=total_token_num,
        prefix_total_token_num=0,
        max_q_seq_len=max(input_lens),
        max_kv_seq_len=max(input_lens),
        max_cache_len=None,
        input_ids=test_data,
        mem_indexes_cpu=mem_indexes_cpu,
        b_req_idx=b_req_idx,
        b_seq_len=b_seq_len,
        b_mtp_index=torch.zeros(actual_batch_size, dtype=torch.int32, device="cpu"),
        b_prefill_start_loc=b_prefill_start_loc,
        is_prefill=True,
        b_ready_cache_len=b_ready_cache_len,
        multimodal_params=[{"images": [], "audios": []} for _ in range(actual_batch_size)],
    )

    model_output = main_model.forward(model_input)
    next_token_ids = torch.argmax(model_output.logits, dim=-1, keepdim=True)

    torch.cuda.synchronize()
    prefill_end = time.time()
    prefill_time = (prefill_end - prefill_start) * 1000
    prefill_throughput = actual_batch_size * sum(input_lens) / (prefill_end - prefill_start)

    if rank_id == 0 and not warmup:
        print(f"  Prefill: {prefill_time:.2f} ms, throughput: {prefill_throughput:.2f} tokens/s")

    # Decode - 与prepare_decode_inputs一致
    torch.cuda.synchronize()
    decode_start = time.time()

    steps_completed = 0
    while steps_completed < output_len:
        # 构建 decode 输入
        b_seq_len_list = []
        for i in range(actual_batch_size):
            seq_len = input_lens[i] + steps_completed + 1
            b_seq_len_list.append(seq_len)

        decode_b_seq_len = torch.tensor(b_seq_len_list, dtype=torch.int32, device="cpu")
        total_token_num = sum(b_seq_len_list)
        max_kv_seq_len = max(b_seq_len_list)

        # 分配内存索引
        mem_indexes = main_model.req_manager.mem_manager.alloc(actual_batch_size)
        mem_indexes_cpu = mem_indexes

        decode_input = ModelInput(
            batch_size=actual_batch_size,
            total_token_num=total_token_num,
            max_q_seq_len=1,
            max_kv_seq_len=max_kv_seq_len,
            max_cache_len=None,
            input_ids=next_token_ids.reshape(-1),
            mem_indexes_cpu=mem_indexes_cpu,
            b_req_idx=b_req_idx,
            b_seq_len=decode_b_seq_len,
            b_mtp_index=torch.zeros(actual_batch_size, dtype=torch.int32, device="cpu"),
            is_prefill=False,
            b_ready_cache_len=None,
            multimodal_params=[{"images": [], "audios": []} for _ in range(actual_batch_size)],
        )

        model_output = main_model.forward(decode_input)
        next_token_ids = torch.argmax(model_output.logits, dim=-1, keepdim=True)
        steps_completed += 1

    torch.cuda.synchronize()
    decode_end = time.time()
    decode_time = (decode_end - decode_start) * 1000
    decode_throughput = actual_batch_size * steps_completed / (decode_end - decode_start)

    if rank_id == 0 and not warmup:
        print(f"  Decode: {decode_time:.2f} ms, throughput: {decode_throughput:.2f} tokens/s")

    total_time = prefill_time + decode_time
    total_tokens = actual_batch_size * (sum(input_lens) + steps_completed)
    overall_throughput = total_tokens / (total_time / 1000)

    # Cleanup
    main_model.mem_manager.free_all()
    main_model.req_manager.free_all()

    return BenchmarkResult(
        batch_size=actual_batch_size,
        input_len=sum(input_lens) // actual_batch_size,
        output_len=output_len,
        prefill_time_ms=prefill_time,
        prefill_throughput=prefill_throughput,
        decode_time_ms=decode_time,
        decode_throughput=decode_throughput,
        total_time_ms=total_time,
        overall_throughput=overall_throughput,
        mode="no_mtp",
    )


def _gen_argmax_token_ids(model_output: ModelOutput):
    """模拟服务端的 _gen_argmax_token_ids 函数"""
    logits = model_output.logits
    probs = torch.softmax(logits, dim=-1)
    draft_next_token_ids_gpu = torch.argmax(probs, dim=-1)
    return draft_next_token_ids_gpu


def run_benchmark_eagle_mtp(
    args,
    input_ids_list: List[List[int]],
    output_len: int,
    batch_size: int,
    main_model,
    draft_model,
    enable_dynamic_mtp: bool = False,
    warmup: bool = False
) -> BenchmarkResult:
    """
    Eagle MTP 模式的 benchmark
    严格按照服务端 decode_mtp 和 _draft_decode_eagle 实现
    支持静态和动态 MTP
    """
    rank_id = get_current_rank_in_dp()
    actual_batch_size = min(batch_size, len(input_ids_list))
    input_lens = [len(input_ids_list[i]) for i in range(actual_batch_size)]
    max_input_len = max(input_lens)
    mtp_step = args.mtp_step

    mode_str = "dynamic_mtp" if enable_dynamic_mtp else "static_mtp"

    # 输入数据
    input_ids = []
    for i in range(actual_batch_size):
        ids = input_ids_list[i]
        input_ids.extend(ids)

    test_data = torch.tensor(input_ids, dtype=torch.long, device="cpu")

    # ==================== Prefill 阶段 ====================
    torch.cuda.synchronize()
    prefill_start = time.time()

    # 分配请求索引
    b_req_idx = torch.tensor(
        [main_model.req_manager.alloc() for _ in range(actual_batch_size)],
        dtype=torch.int32, device="cpu"
    )
    b_seq_len = torch.tensor(input_lens, dtype=torch.int32, device="cpu")
    b_ready_cache_len = torch.zeros(actual_batch_size, dtype=torch.int32, device="cpu")

    total_token_num = sum(input_lens)

    # 分配内存索引
    mem_indexes = main_model.req_manager.mem_manager.alloc(test_data.shape[0])
    mem_indexes_cpu = mem_indexes

    b_q_seq_len = torch.tensor(input_lens, dtype=torch.int32, device="cpu")
    b_prefill_start_loc = b_q_seq_len.cumsum(dim=0, dtype=torch.int32) - b_q_seq_len

    # 主模型 prefill 输入
    model_input = ModelInput(
        batch_size=actual_batch_size,
        total_token_num=total_token_num,
        prefix_total_token_num=0,
        max_q_seq_len=max(input_lens),
        max_kv_seq_len=max(input_lens),
        max_cache_len=None,
        input_ids=test_data,
        mem_indexes_cpu=mem_indexes_cpu,
        b_req_idx=b_req_idx,
        b_seq_len=b_seq_len,
        b_mtp_index=torch.zeros(actual_batch_size, dtype=torch.int32, device="cpu"),
        b_prefill_start_loc=b_prefill_start_loc,
        is_prefill=True,
        b_ready_cache_len=b_ready_cache_len,
        multimodal_params=[{"images": [], "audios": []} for _ in range(actual_batch_size)],
    )

    # 主模型 prefill forward
    model_output = main_model.forward(model_input)
    next_token_ids = torch.argmax(model_output.logits, dim=-1, keepdim=True)

    # Draft 模型 prefill - 填充 KV cache
    draft_model_input = model_input
    draft_model_output = model_output
    draft_next_token_ids = next_token_ids.reshape(-1)

    for _ in range(1):
        draft_model_input = prepare_mtp_prefill_inputs_for_benchmark(
            model_input=draft_model_input,
            b_next_token_ids=draft_next_token_ids,
            mtp_draft_input_hiddens=draft_model_output.mtp_main_output_hiddens,
        )
        draft_model_output = draft_model.forward(draft_model_input)
        draft_next_token_ids = _gen_argmax_token_ids(draft_model_output)

    torch.cuda.synchronize()
    prefill_end = time.time()
    prefill_time = (prefill_end - prefill_start) * 1000
    prefill_throughput = actual_batch_size * sum(input_lens) / (prefill_end - prefill_start)

    if rank_id == 0 and not warmup:
        print(f"  Prefill: {prefill_time:.2f} ms, throughput: {prefill_throughput:.2f} tokens/s")

    # ==================== Decode 阶段 ====================
    torch.cuda.synchronize()
    decode_start = time.time()

    # 初始化 decode 状态
    current_step = 0
    steps_completed = 0
    total_accepted_steps = 0

    # 每个请求的当前 MTP size（动态模式下会更新）
    req_mtp_sizes = [mtp_step] * actual_batch_size

    # 创建模拟的 req_to_next_token_ids (max_req_num, max_mtp_step)
    max_req_num = main_model.req_manager.req_sampling_params_manager.req_to_next_token_ids.shape[0]
    req_to_next_token_ids = torch.zeros((max_req_num, mtp_step + 1), dtype=torch.int32, device="cuda")

    while steps_completed < output_len:
        # 构建 decode 输入 - 严格按照 prepare_decode_inputs 实现
        run_reqs_ids = []
        b_req_idx_list = []
        b_seq_len_list = []
        b_mtp_index = []
        b_q_seq_len = []

        for i in range(actual_batch_size):
            current_mtp_size = req_mtp_sizes[i]
            current_seq_len = input_lens[i] + current_step + 1

            # 主模型 token (mtp_index=0)
            run_reqs_ids.append(i)
            b_req_idx_list.append(b_req_idx[i].item())
            b_seq_len_list.append(current_seq_len)
            b_mtp_index.append(0)
            b_q_seq_len.append(1)

            # Draft tokens (mtp_index=1..current_mtp_size)
            for step in range(current_mtp_size):
                run_reqs_ids.append(i)
                b_req_idx_list.append(b_req_idx[i].item())
                b_seq_len_list.append(current_seq_len + step + 1)
                b_mtp_index.append(step + 1)
                b_q_seq_len.append(1)

        # 计算 batch_size（扩展后的）
        expanded_batch_size = len(b_mtp_index)
        num_reqs = actual_batch_size  # 原始请求数量

        # 构建 b_mark_shared_group
        b_mtp_index_tensor = torch.tensor(b_mtp_index, dtype=torch.int32, device="cpu")
        b_mark_shared_group = build_mtp_shared_group_infos(b_mtp_index_tensor)

        b_req_idx_expanded = torch.tensor(b_req_idx_list, dtype=torch.int32, device="cpu")
        b_seq_len_expanded = torch.tensor(b_seq_len_list, dtype=torch.int32, device="cpu")

        total_token_num = sum(b_seq_len_list)
        max_kv_seq_len = max(b_seq_len_list)
        max_q_seq_len = 1

        # 分配内存索引
        mem_indexes = main_model.req_manager.mem_manager.alloc(expanded_batch_size)
        mem_indexes_cpu = mem_indexes

        # 获取上一步的主模型 token 作为输入
        if current_step == 0:
            main_next_ids = next_token_ids.reshape(-1)
        else:
            main_next_ids = next_token_ids[:actual_batch_size]

        # 构建 input_ids：主模型位置放 main_next_ids
        decode_input_ids = torch.zeros(expanded_batch_size, dtype=torch.long, device="cuda")
        main_positions = (b_mtp_index_tensor == 0).nonzero(as_tuple=True)[0]
        decode_input_ids[main_positions] = main_next_ids

        decode_input = ModelInput(
            batch_size=expanded_batch_size,
            total_token_num=total_token_num,
            max_q_seq_len=max_q_seq_len,
            max_kv_seq_len=max_kv_seq_len,
            max_cache_len=None,
            input_ids=decode_input_ids.cpu(),
            mem_indexes_cpu=mem_indexes_cpu,
            b_req_idx=b_req_idx_expanded,
            b_seq_len=b_seq_len_expanded,
            b_mtp_index=b_mtp_index_tensor,
            b_mark_shared_group=b_mark_shared_group,
            is_prefill=False,
            b_ready_cache_len=None,
            multimodal_params=[{"images": [], "audios": []} for _ in range(expanded_batch_size)],
        )

        # 主模型 decode forward
        model_output = main_model.forward(decode_input)
        main_logits = model_output.logits

        # 获取主模型预测的 token（mtp_index=0 的位置）
        b_mtp_index_gpu = b_mtp_index_tensor.cuda()
        main_positions_gpu = (b_mtp_index_gpu == 0).nonzero(as_tuple=True)[0]
        main_predict_logits = main_logits[main_positions_gpu]
        main_next_token_ids = torch.argmax(main_predict_logits, dim=-1, keepdim=True)

        # 构建 b_req_mtp_start_loc
        b_req_mtp_start_loc = (b_mtp_index_gpu == 0).nonzero(as_tuple=True)[0]

        # ==================== MTP Verify ====================
        # 先将 draft token 放入 req_to_next_token_ids
        # Eagle模式下，需要先进行 draft decode 生成 draft tokens

        # ==================== Eagle Draft Decode ====================
        # 分配 eagle_mem_indexes
        eagle_mem_indexes = main_model.req_manager.mem_manager.alloc(num_reqs * mtp_step)
        eagle_mem_indexes_gpu = eagle_mem_indexes.cuda()

        draft_model_input = decode_input
        draft_model_output = model_output
        draft_next_token_ids = main_next_token_ids.reshape(-1)
        all_next_token_ids = []
        all_next_token_ids.append(draft_next_token_ids)

        # 用于收集每个 step 的 probs（动态模式）
        draft_probs_list = [] if enable_dynamic_mtp else None

        # 计算 mtp_group_sizes（动态模式）
        if enable_dynamic_mtp:
            mtp_group_sizes = []
            current_group_size = 0
            for mtp_idx in b_mtp_index:
                if mtp_idx == 0:
                    if current_group_size > 0:
                        mtp_group_sizes.append(current_group_size)
                    current_group_size = 1
                else:
                    current_group_size += 1
            if current_group_size > 0:
                mtp_group_sizes.append(current_group_size)

        for _step in range(mtp_step):
            draft_model_input_cpu = draft_model_input
            draft_model_input_cpu.input_ids = draft_next_token_ids.cpu()
            draft_model_input_cpu.mtp_draft_input_hiddens = draft_model_output.mtp_main_output_hiddens

            draft_model_output = draft_model.forward(draft_model_input_cpu)
            draft_logits = draft_model_output.logits

            # 收集 probs（如果需要）
            if enable_dynamic_mtp:
                draft_probs = torch.softmax(draft_logits, dim=-1)
                draft_probs_list.append(draft_probs[main_positions_gpu])

            draft_next_token_ids = _gen_argmax_token_ids(draft_model_output)
            draft_model_input_cpu.b_seq_len = draft_model_input_cpu.b_seq_len + 1
            draft_model_input_cpu.max_kv_seq_len += 1

            # 更新 mem_indexes（Eagle模式）
            eagle_mem_indexes_i = eagle_mem_indexes_gpu[_step * num_reqs:(_step + 1) * num_reqs]

            if enable_dynamic_mtp:
                # 动态模式：根据 group_sizes 拆分和重新组合
                chunks = torch.split(draft_model_input_cpu.mem_indexes.cuda(), mtp_group_sizes)
                new_chunks = []
                for i, chunk in enumerate(chunks):
                    updated_chunk = torch.cat([chunk[1:], eagle_mem_indexes_i[i:i+1]], dim=0)
                    new_chunks.append(updated_chunk)
                draft_model_input_cpu.mem_indexes = torch.cat(new_chunks, dim=0).cpu()
                draft_model_input_cpu.mem_indexes_cpu = draft_model_input_cpu.mem_indexes
            else:
                # 静态模式
                mem_indexes_view = draft_model_input_cpu.mem_indexes.cuda().view(-1, mtp_step + 1)
                new_mem_indexes = torch.cat(
                    [mem_indexes_view[:, 1:], eagle_mem_indexes_i.view(-1, 1)],
                    dim=1,
                ).view(-1)
                draft_model_input_cpu.mem_indexes = new_mem_indexes.cpu()
                draft_model_input_cpu.mem_indexes_cpu = draft_model_input_cpu.mem_indexes

            all_next_token_ids.append(draft_next_token_ids)

        all_next_token_ids = torch.stack(all_next_token_ids, dim=1)  # [batch_size, mtp_step + 1]

        # 将 draft tokens scatter 到 req_to_next_token_ids
        mtp_scatter_next_token_ids(
            req_to_next_token_ids=req_to_next_token_ids,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            all_next_token_ids=all_next_token_ids,
            b_req_idx=b_req_idx_expanded.cuda(),
            mtp_accept_len=torch.full((num_reqs,), mtp_step + 1, dtype=torch.int32, device="cuda"),
        )

        # 进行 MTP Verify
        mtp_accept_len, accepted_index = mtp_verify(
            req_to_next_token_ids=req_to_next_token_ids,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            new_next_token_ids=main_next_token_ids.reshape(-1),
            b_req_idx=b_req_idx_expanded.cuda(),
        )

        # 计算接受的 token 数量
        accepted_len = mtp_accept_len.float().mean().item()

        # 动态 MTP：根据验证结果更新每个请求的 MTP size
        if enable_dynamic_mtp:
            # 获取验证通过的请求
            verify_ok_mask = accepted_index[main_positions_gpu] == 1
            if verify_ok_mask.any():
                verify_ok_logits = main_predict_logits[verify_ok_mask]
                verify_ok_draft_probs = [probs[verify_ok_mask] for probs in draft_probs_list]

                dynamic_mtp_sizes = compute_dynamic_mtp_size(
                    main_probs=verify_ok_logits,
                    max_mtp_size=mtp_step,
                    draft_probs=verify_ok_draft_probs,
                )

                # 更新每个请求的 MTP size
                req_idx = 0
                for i in range(actual_batch_size):
                    if verify_ok_mask[i]:
                        req_mtp_sizes[i] = dynamic_mtp_sizes[req_idx]
                        req_idx += 1

        total_accepted_steps += int(accepted_len) - 1 if accepted_len > 0 else 0
        steps_completed += int(accepted_len)
        current_step += int(accepted_len)

        # 准备下一次迭代的输入
        next_token_ids = main_next_token_ids

        # 释放 eagle_mem_indexes
        main_model.mem_manager.free(eagle_mem_indexes)

    torch.cuda.synchronize()
    decode_end = time.time()
    decode_time = (decode_end - decode_start) * 1000
    decode_throughput = actual_batch_size * steps_completed / (decode_end - decode_start)

    avg_accepted_steps = total_accepted_steps / max(steps_completed // (mtp_step + 1), 1) if steps_completed > 0 else 0

    if rank_id == 0 and not warmup:
        print(f"  Decode: {decode_time:.2f} ms, throughput: {decode_throughput:.2f} tokens/s")
        if enable_dynamic_mtp:
            print(f"  Avg accepted steps: {avg_accepted_steps:.2f} / {mtp_step}")

    total_time = prefill_time + decode_time
    total_tokens = actual_batch_size * (sum(input_lens) + steps_completed)
    overall_throughput = total_tokens / (total_time / 1000)

    # Cleanup
    main_model.mem_manager.free_all()
    main_model.req_manager.free_all()

    return BenchmarkResult(
        batch_size=actual_batch_size,
        input_len=sum(input_lens) // actual_batch_size,
        output_len=output_len,
        prefill_time_ms=prefill_time,
        prefill_throughput=prefill_throughput,
        decode_time_ms=decode_time,
        decode_throughput=decode_throughput,
        total_time_ms=total_time,
        overall_throughput=overall_throughput,
        mode=mode_str,
        avg_accepted_steps=avg_accepted_steps,
    )


def worker_process(args, model_kvargs, input_ids_list, output_len, batch_size, ans_queue, mode):
    """工作进程函数"""
    import torch.distributed as dist
    from lightllm.distributed import dist_group_manager

    # 设置模式
    args.mtp_dynamic_verify = (mode == "dynamic_mtp")
    set_env_start_args(args)

    init_distributed_env(model_kvargs)
    dist_group_manager.create_groups(group_size=1)
    model_cfg, _ = PretrainedConfig.get_config_dict(model_kvargs["weight_dir"])
    dist.barrier()

    torch.cuda.empty_cache()

    main_model, _ = get_model(model_cfg, model_kvargs)

    rank_id = model_kvargs["rank_id"]

    if rank_id == 0:
        print(f"\n{'='*60}")
        print(f"Testing {mode} - Batch Size: {batch_size}")
        print(f"{'='*60}")

    # 根据模式运行不同的 benchmark
    if mode == "no_mtp":
        # Warmup
        run_benchmark_no_mtp(args, input_ids_list, output_len, batch_size, main_model, warmup=True)
        dist.barrier()

        # Actual benchmark
        result = run_benchmark_no_mtp(args, input_ids_list, output_len, batch_size, main_model, warmup=False)
    else:
        # MTP 模式需要初始化 draft model
        draft_model = init_mtp_model(args, model_kvargs, main_model)

        enable_dynamic = (mode == "dynamic_mtp")

        # Warmup
        run_benchmark_eagle_mtp(args, input_ids_list, output_len, batch_size, main_model, draft_model,
                                 enable_dynamic_mtp=enable_dynamic, warmup=True)
        dist.barrier()

        # Actual benchmark
        result = run_benchmark_eagle_mtp(args, input_ids_list, output_len, batch_size, main_model, draft_model,
                                          enable_dynamic_mtp=enable_dynamic, warmup=False)

    dist.barrier()
    ans_queue.put(result if rank_id == 0 else None)
    return


def run_single_config(
    args,
    input_ids_list: List[List[int]],
    output_len: int,
    batch_size: int,
    mode: str
) -> BenchmarkResult:
    """运行单种配置的 benchmark"""
    from easydict import EasyDict

    if not isinstance(args, EasyDict):
        args = EasyDict(vars(args))

    ans_queue = Queue()
    workers = []

    dp_size = args.get("dp", 1)

    for rank_id in range(args.node_rank * args.tp, (args.node_rank + 1) * args.tp):
        model_kvargs = {
            "args": args,
            "nccl_host": args.nccl_host,
            "data_type": args.data_type,
            "nccl_port": args.nccl_port,
            "rank_id": rank_id,
            "world_size": args.tp,
            "dp_size": dp_size,
            "weight_dir": args.model_dir,
            "quant_type": getattr(args, 'quant_type', None),
            "load_way": "HF",
            "max_total_token_num": args.max_total_token_num,
            "graph_max_len_in_batch": args.max_req_total_len,
            "graph_max_batch_size": getattr(args, 'graph_max_batch_size', 16),
            "mem_faction": getattr(args, 'mem_fraction', 0.9),
            "max_req_num": 2000,
            "batch_max_tokens": 2048,
            "run_mode": "normal",
            "max_seq_length": args.max_req_total_len,
            "spec_algo": getattr(args, 'spec_algo', None),
            "disable_cudagraph": getattr(args, 'disable_cudagraph', True),
            "mtp_mode": getattr(args, 'mtp_mode', None),
            "mtp_draft_model_dir": getattr(args, 'mtp_draft_model_dir', None),
            "llm_decode_att_backend": getattr(args, 'llm_decode_att_backend', 'triton'),
            "llm_prefill_att_backend": getattr(args, 'llm_prefill_att_backend', ['fa3']),
        }

        proc = multiprocessing.Process(
            target=worker_process,
            args=(args, model_kvargs, input_ids_list, output_len, batch_size, ans_queue, mode)
        )
        proc.start()
        workers.append(proc)

    for proc in workers:
        proc.join()

    while not ans_queue.empty():
        result = ans_queue.get()
        if result is not None:
            return result

    return None


def print_comparison_report(results: List[BenchmarkResult], output_file: str = None):
    """打印对比报告"""
    report_lines = []

    def log(msg):
        print(msg)
        report_lines.append(msg)

    log("\n" + "="*100)
    log("MTP Throughput Benchmark Report")
    log("="*100)

    # Group by batch size
    by_config = {}
    for r in results:
        key = (r.batch_size, r.input_len, r.output_len)
        if key not in by_config:
            by_config[key] = {}
        by_config[key][r.mode] = r

    log(f"\n{'Batch':>8} {'Input':>8} {'Output':>8} {'Mode':>18} {'Prefill(t/s)':>14} {'Decode(t/s)':>14} {'Overall(t/s)':>14} {'Avg Steps':>12}")
    log("-" * 110)

    for (bs, in_len, out_len), modes in sorted(by_config.items()):
        for mode in ["no_mtp", "static_mtp", "dynamic_mtp"]:
            if mode in modes:
                r = modes[mode]
                log(f"{bs:>8} {in_len:>8} {out_len:>8} {mode:>18} {r.prefill_throughput:>14.2f} {r.decode_throughput:>14.2f} {r.overall_throughput:>14.2f} {r.avg_accepted_steps:>12.2f}")

        # Calculate improvement
        if "no_mtp" in modes and "static_mtp" in modes:
            r_no = modes["no_mtp"]
            r_static = modes["static_mtp"]
            static_improvement = ((r_static.overall_throughput - r_no.overall_throughput) / r_no.overall_throughput) * 100
            log(f"{'':>8} {'':>8} {'':>8} {'Static vs No MTP':>18} {'':>14} {'':>14} {static_improvement:>13.2f}% {'':>12}")

        if "static_mtp" in modes and "dynamic_mtp" in modes:
            r_static = modes["static_mtp"]
            r_dynamic = modes["dynamic_mtp"]
            dynamic_improvement = ((r_dynamic.overall_throughput - r_static.overall_throughput) / r_static.overall_throughput) * 100
            log(f"{'':>8} {'':>8} {'':>8} {'Dynamic vs Static':>18} {'':>14} {'':>14} {dynamic_improvement:>13.2f}% {'':>12}")

        log("")

    log("="*100)

    if output_file:
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))
        print(f"\nReport saved to {output_file}")


def main():
    parser = make_argument_parser()

    # Benchmark specific arguments
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 4, 8, 16],
                        help="List of batch sizes to test")
    parser.add_argument("--output_lens", type=int, nargs="+", default=[128, 512],
                        help="List of output lengths to test")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to dataset JSON file")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to load from dataset")
    parser.add_argument("--report_file", type=str, default="mtp_benchmark_report.txt",
                        help="Output file for benchmark report")
    parser.add_argument("--skip_no_mtp", action="store_true",
                        help="Skip no MTP benchmark")
    parser.add_argument("--skip_static_mtp", action="store_true",
                        help="Skip static MTP benchmark")
    parser.add_argument("--skip_dynamic_mtp", action="store_true",
                        help="Skip dynamic MTP benchmark")

    args = parser.parse_args()
    set_env_start_args(args)

    torch.multiprocessing.set_start_method("spawn", force=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    # Load dataset
    if args.dataset and os.path.exists(args.dataset):
        print(f"Loading dataset from {args.dataset}")
        samples = load_dataset_samples(args.dataset, args.num_samples, tokenizer)
        print(f"Loaded {len(samples)} samples")
    else:
        print("No dataset provided, using random data")
        samples = None

    all_results = []

    for output_len in args.output_lens:
        for batch_size in args.batch_sizes:
            # Prepare input data
            if samples:
                input_ids_list = [samples[i % len(samples)] for i in range(batch_size)]
            else:
                input_len = 256
                input_ids_list = [np.random.randint(0, 50256, input_len).tolist() for _ in range(batch_size)]

            # Test no MTP
            if not args.skip_no_mtp:
                print(f"\n[No MTP] Batch={batch_size}, Output={output_len}")
                result = run_single_config(args, input_ids_list, output_len, batch_size, mode="no_mtp")
                if result:
                    all_results.append(result)

            # Test static MTP
            if not args.skip_static_mtp and args.mtp_mode:
                print(f"\n[Static MTP] Batch={batch_size}, Output={output_len}")
                result = run_single_config(args, input_ids_list, output_len, batch_size, mode="static_mtp")
                if result:
                    all_results.append(result)

            # Test dynamic MTP
            if not args.skip_dynamic_mtp and args.mtp_mode:
                print(f"\n[Dynamic MTP] Batch={batch_size}, Output={output_len}")
                result = run_single_config(args, input_ids_list, output_len, batch_size, mode="dynamic_mtp")
                if result:
                    all_results.append(result)

    # Print report
    if all_results:
        print_comparison_report(all_results, args.report_file)


if __name__ == "__main__":
    main()
