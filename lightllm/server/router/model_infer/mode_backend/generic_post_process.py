import torch
from typing import List, Optional, Union
from lightllm.common.basemodel.triton_kernel.post_process.apply_penalty import apply_penalty
from lightllm.common.basemodel.triton_kernel.post_process.apply_penalty_gpu_cache import apply_penalty_gpu_cache
from lightllm.common.basemodel.triton_kernel.post_process.apply_invalid_token import apply_invalid_token_ids
from lightllm.server.router.model_infer.infer_batch import InferReq, g_infer_context
from lightllm.server.router.model_infer.pin_mem_manager import g_pin_mem_manager
from lightllm.utils.envs_utils import get_env_start_args

_flashinfer_top_k_top_p_sampling_from_logits = None
_flashinfer_top_k_top_p_sampling_from_logits_checked = False
_flashinfer_top_k_top_p_sampling_from_probs = None
_flashinfer_top_k_top_p_sampling_from_probs_checked = False
_flashinfer_top_p_sampling_from_probs = None
_flashinfer_top_p_sampling_from_probs_checked = False
_flashinfer_top_k_sampling_from_probs = None
_flashinfer_top_k_sampling_from_probs_checked = False
_uniform_tensor_cache = {}
_softmax_out_cache = {}
_is_flashinfer_sampling_backend = None


def sample(logits: torch.Tensor, reqs: List[InferReq], eos_id: List[int] = [2]):
    fast_next_token_ids = _try_flashinfer_sample_without_penalty(logits, reqs)
    if fast_next_token_ids is not None:
        return fast_next_token_ids.view(-1), None

    (
        b_req_idx,
        b_temperatures,
        b_top_ps,
        b_top_ks,
        b_length_penalty_param,
        b_mask_eos_reqs,
        invalid_token_ids,
        cu_invalid_token_num,
        is_all_greedy,
        has_invalid_token_ids,
        skip_top_k,
        skip_top_p,
        exist_req_use_random_seed,
    ) = _get_post_sample_tensors(reqs)
    eos_ids = g_pin_mem_manager.gen_from_list(key="eos_ids", data=eos_id, dtype=torch.int32).cuda(non_blocking=True)

    sampling_params_manager = g_infer_context.req_manager.req_sampling_params_manager

    # 这里需要区分历史token的频率惩罚类的系数的生效模式，目前支持两种在线统计方式:
    # 一种是基于 cpu 的，每个 req 对象利用其上绑定的dict对象out_token_id_count，每生成一个token就进行相应
    # 的计数更新，当进行使用的时候, 对一个需要处理的req list, 会生成对应的3个 triton kernel 需要使用的惩罚系数
    # 输入参数  p_token_ids, p_token_counts, p_cumsum_seq_len，这种方式的特点是占用的显存少，在请求输出不长的时候，
    # 速度快且没有代价，但是目前 RL 采样场景下，需要进行大量的长输出生成，这时候，cpu进行的处理操作会形成一些瓶颈，影响
    # 推理的速度。
    # 一种是基于 gpu buffer的，每个请求都会被分配一个 vocab_size 大小的 cuda tensor 用于出现过的token进行计数，
    # 然后在直接使用 triton kernel 在对应的logits上进行相应的惩罚操作，这种方法的特点是，处理速度快，但是需要预先
    # 分配较大的显存空间用于token的计数，如果以常见的词表大小 vocab_size = 500000, 预分配1000个请求的cuda tensor，
    # 使用int32类型进行计数大概需要600M的空间，这也不是一笔不菲的开销。
    # 所以需要根据具体的显卡，使用场景，来判断使用那种方式，默认情况下 为gpu模式，可以调整args.penalty_counter_mode
    # 参数来控制使用方式。
    if sampling_params_manager.penalty_counter_mode == "cpu_counter":
        (
            p_token_ids,
            p_token_counts,
            p_cumsum_seq_len,
        ) = sampling_params_manager.gen_cpu_out_token_counter_sampling_params(req_objs=reqs)

        apply_penalty(
            Logits=logits,
            b_req_idx=b_req_idx,
            b_length_penalty_param=b_length_penalty_param,
            b_mask_eos_reqs=b_mask_eos_reqs,
            p_token_ids=p_token_ids,
            p_token_counts=p_token_counts,
            p_cumsum_seq_len=p_cumsum_seq_len,
            eos_ids=eos_ids,
            sampling_params_manager=sampling_params_manager,
        )
    else:
        apply_penalty_gpu_cache(
            Logits=logits,
            b_req_idx=b_req_idx,
            b_length_penalty_param=b_length_penalty_param,
            b_mask_eos_reqs=b_mask_eos_reqs,
            eos_ids=eos_ids,
            sampling_params_manager=sampling_params_manager,
        )

    if has_invalid_token_ids:
        apply_invalid_token_ids(
            Logits=logits,
            invalid_token_ids=invalid_token_ids,
            cu_invalid_token_num=cu_invalid_token_num,
        )

    if b_temperatures is not None:
        logits.div_(b_temperatures.view((-1, 1)))

    if is_all_greedy:
        batch_next_token_ids = torch.argmax(logits, -1)
        return batch_next_token_ids.view(-1), None

    probs = torch.softmax(logits, dim=-1)

    if skip_top_k and skip_top_p:
        # topk 等于整个词表，topp 等于1.0，等价于不进行topk topp过滤，直接进行随机采样，可以提升采样速度
        batch_next_token_ids = _random_sample(probs, reqs, exist_req_use_random_seed)
        return batch_next_token_ids.view(-1), None

    else:
        batch_next_token_ids = _top_p_top_k_sample(
            reqs,
            probs,
            b_top_ps,
            b_top_ks,
            skip_top_k,
            skip_top_p,
            exist_req_use_random_seed,
        )
        return batch_next_token_ids.view(-1), None


def _try_flashinfer_sample_without_penalty(logits: torch.Tensor, reqs: List[InferReq]) -> Optional[torch.Tensor]:
    if not _is_flashinfer_sampling() or not reqs:
        return None

    first_param = reqs[0].sampling_param.shm_param
    top_p = first_param.top_p
    top_k = first_param.top_k
    temperature = first_param.temperature
    vocab_size = reqs[0].vocab_size

    if top_k <= 1 or (top_k == vocab_size and top_p == 1.0):
        return None

    for req in reqs:
        shm_param = req.sampling_param.shm_param
        if req.generator is not None:
            return None
        if len(req.sampling_param.invalid_token_ids) != 0:
            return None
        if not shm_param.ignore_eos:
            return None
        if shm_param.presence_penalty != 0.0:
            return None
        if shm_param.frequency_penalty != 0.0:
            return None
        if shm_param.repetition_penalty != 1.0:
            return None
        if shm_param.temperature != temperature:
            return None
        if shm_param.top_p != top_p:
            return None
        if shm_param.top_k != top_k:
            return None

    if temperature != 1.0:
        logits.div_(temperature)

    if top_k == vocab_size and top_p != 1.0:
        top_p_tensor = _get_uniform_tensor(top_p, logits.shape[0], torch.float32, logits.device)
        return _flashinfer_top_p_sample_from_logits(logits, top_p_tensor)

    top_p_tensor = _get_uniform_tensor(top_p, logits.shape[0], torch.float32, logits.device)
    top_k_tensor = _get_uniform_tensor(top_k, logits.shape[0], torch.int32, logits.device)
    return _flashinfer_top_p_top_k_sample_from_logits(logits, top_p_tensor, top_k_tensor)


def _is_flashinfer_sampling() -> bool:
    global _is_flashinfer_sampling_backend
    if _is_flashinfer_sampling_backend is None:
        _is_flashinfer_sampling_backend = get_env_start_args().sampling_backend == "flashinfer"
    return _is_flashinfer_sampling_backend


def _top_p_top_k(probs: torch.Tensor, top_ps: torch.Tensor, top_ks: torch.Tensor):
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)

    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0

    probs_sort[torch.arange(0, probs.shape[-1], device="cuda").view(1, -1) >= top_ks.view(-1, 1)] = 0.0

    return probs_sort, probs_idx


def _flashinfer_top_p_top_k_sample_from_logits(
    logits: torch.Tensor,
    b_top_ps: Union[torch.Tensor, float],
    b_top_ks: Union[torch.Tensor, int],
) -> Optional[torch.Tensor]:
    global _flashinfer_top_k_top_p_sampling_from_logits
    global _flashinfer_top_k_top_p_sampling_from_logits_checked

    if not _flashinfer_top_k_top_p_sampling_from_logits_checked:
        try:
            from flashinfer.sampling import top_k_top_p_sampling_from_logits
        except ImportError:
            top_k_top_p_sampling_from_logits = None
        _flashinfer_top_k_top_p_sampling_from_logits = top_k_top_p_sampling_from_logits
        _flashinfer_top_k_top_p_sampling_from_logits_checked = True

    if _flashinfer_top_k_top_p_sampling_from_logits is None:
        return None

    return _flashinfer_top_k_top_p_sampling_from_logits(
        logits,
        b_top_ks,
        b_top_ps,
        filter_apply_order="joint",
        deterministic=True,
        check_nan=False,
    )


def _flashinfer_top_p_sample_from_logits(
    logits: torch.Tensor, top_p: Union[torch.Tensor, float]
) -> Optional[torch.Tensor]:
    probs = _softmax_out(logits)
    return _flashinfer_top_p_sample_from_probs(probs, top_p)


def _get_uniform_tensor(value: Union[float, int], size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    key = (str(device), dtype, size, value)
    tensor = _uniform_tensor_cache.get(key)
    if tensor is None:
        tensor = torch.full((size,), value, dtype=dtype, device=device)
        _uniform_tensor_cache[key] = tensor
    return tensor


def _softmax_out(logits: torch.Tensor) -> torch.Tensor:
    key = (str(logits.device), logits.dtype, tuple(logits.shape))
    probs = _softmax_out_cache.get(key)
    if probs is None:
        probs = torch.empty_like(logits)
        _softmax_out_cache[key] = probs
    torch.ops.aten._softmax.out(logits, -1, False, out=probs)
    return probs


def _get_flashinfer_top_k_top_p_sampling_from_probs():
    global _flashinfer_top_k_top_p_sampling_from_probs
    global _flashinfer_top_k_top_p_sampling_from_probs_checked

    if not _flashinfer_top_k_top_p_sampling_from_probs_checked:
        try:
            from flashinfer.sampling import top_k_top_p_sampling_from_probs
        except ImportError:
            top_k_top_p_sampling_from_probs = None
        _flashinfer_top_k_top_p_sampling_from_probs = top_k_top_p_sampling_from_probs
        _flashinfer_top_k_top_p_sampling_from_probs_checked = True
    return _flashinfer_top_k_top_p_sampling_from_probs


def _flashinfer_top_p_sample_from_probs(
    probs: torch.Tensor, top_p: Union[torch.Tensor, float]
) -> Optional[torch.Tensor]:
    global _flashinfer_top_p_sampling_from_probs
    global _flashinfer_top_p_sampling_from_probs_checked

    if not _flashinfer_top_p_sampling_from_probs_checked:
        try:
            from flashinfer.sampling import top_p_sampling_from_probs
        except ImportError:
            top_p_sampling_from_probs = None
        _flashinfer_top_p_sampling_from_probs = top_p_sampling_from_probs
        _flashinfer_top_p_sampling_from_probs_checked = True

    if _flashinfer_top_p_sampling_from_probs is None:
        return None

    return _flashinfer_top_p_sampling_from_probs(probs, top_p, deterministic=True, check_nan=False)


def _flashinfer_top_k_sample_from_probs(probs: torch.Tensor, top_k: Union[torch.Tensor, int]) -> Optional[torch.Tensor]:
    global _flashinfer_top_k_sampling_from_probs
    global _flashinfer_top_k_sampling_from_probs_checked

    if not _flashinfer_top_k_sampling_from_probs_checked:
        try:
            from flashinfer.sampling import top_k_sampling_from_probs
        except ImportError:
            top_k_sampling_from_probs = None
        _flashinfer_top_k_sampling_from_probs = top_k_sampling_from_probs
        _flashinfer_top_k_sampling_from_probs_checked = True

    if _flashinfer_top_k_sampling_from_probs is None:
        return None

    return _flashinfer_top_k_sampling_from_probs(probs, top_k, deterministic=True, check_nan=False)


def _top_p_top_k_sample(
    reqs: List[InferReq],
    probs: torch.Tensor,
    b_top_ps: Union[torch.Tensor, float],
    b_top_ks: Union[torch.Tensor, int],
    skip_top_k: bool,
    skip_top_p: bool,
    exist_req_use_random_seed: bool,
) -> torch.Tensor:
    sampling_backend = get_env_start_args().sampling_backend

    if sampling_backend == "triton":
        probs_sort, probs_idx = _top_p_top_k(probs, b_top_ps, b_top_ks)
        if not exist_req_use_random_seed:
            sampled_index = torch.multinomial(probs_sort, num_samples=1, replacement=True)
        else:
            sampled_index = _random_sample(probs_sort, reqs, exist_req_use_random_seed).view(-1, 1)
        next_token_ids = torch.gather(probs_idx, dim=1, index=sampled_index)
        return next_token_ids.view(-1)

    elif sampling_backend == "flashinfer":
        if skip_top_k:
            batch_next_token_ids = _flashinfer_top_p_sample_from_probs(probs, b_top_ps)
        elif skip_top_p:
            batch_next_token_ids = _flashinfer_top_k_sample_from_probs(probs, b_top_ks)
        else:
            top_k_top_p_sampling_from_probs = _get_flashinfer_top_k_top_p_sampling_from_probs()
            if top_k_top_p_sampling_from_probs is None:
                raise ImportError("flashinfer.sampling.top_k_top_p_sampling_from_probs is not available")
            batch_next_token_ids = top_k_top_p_sampling_from_probs(
                probs,
                b_top_ks,
                b_top_ps,
                filter_apply_order="joint",
                deterministic=True,
                check_nan=False,
            )
        if batch_next_token_ids is None:
            raise ImportError("flashinfer sampling op is not available")
        return batch_next_token_ids.view(-1)
    else:
        assert False, "Unsupported sampling backend for top_p_top_k_sample"


def _random_sample(probs: torch.Tensor, reqs: List[InferReq], exist_req_use_random_seed: bool):
    q = torch.empty_like(probs)
    q.exponential_()
    if exist_req_use_random_seed:
        for i, req in enumerate(reqs):
            if req.generator is not None:
                q[i].exponential_(generator=req.generator)
    return probs.div(q).argmax(dim=-1).view(-1)


def _get_post_sample_tensors(reqs: List[InferReq]):
    req_idxes: List[int] = []
    temperatures: List[float] = []
    top_ps: List[float] = []
    top_ks: List[int] = []
    length_penalty_param: List[int] = []
    mask_eos_reqs: List[bool] = []
    is_all_greedy = True
    skip_top_k = True
    skip_top_p = True
    exist_req_use_random_seed = False
    all_temperature_one = True

    # invalid token ids
    invalid_token_ids: List[int] = []
    has_invalid_token_ids = False
    cu_invalid_token_num = [0]
    invalid_token_num_start = 0

    for i, req_obj in enumerate(reqs):
        sample_param = req_obj.sampling_param
        shm_param = sample_param.shm_param
        exponential_decay_length_penalty = shm_param.exponential_decay_length_penalty.to_tuple()
        out_token_len = req_obj.get_cur_total_len() - req_obj.shm_req.input_len
        length_penalty_param.append(max(out_token_len - exponential_decay_length_penalty[0], 0))
        mask_eos_reqs.append(out_token_len < shm_param.min_new_tokens - 1)

        temperatures.append(shm_param.temperature)
        top_ps.append(shm_param.top_p)
        top_k_val = shm_param.top_k
        top_ks.append(top_k_val)
        if top_k_val > 1:
            is_all_greedy = False
        if top_k_val != req_obj.vocab_size:
            skip_top_k = False
        if shm_param.top_p != 1.0:
            skip_top_p = False
        if req_obj.generator is not None:
            exist_req_use_random_seed = True
        if shm_param.temperature != 1.0:
            all_temperature_one = False
        req_idxes.append(req_obj.req_idx)
        invalid_token_num_start += len(req_obj.sampling_param.invalid_token_ids)
        cu_invalid_token_num.append(invalid_token_num_start)
        if len(req_obj.sampling_param.invalid_token_ids) > 0:
            has_invalid_token_ids = True
            invalid_token_ids.extend(req_obj.sampling_param.invalid_token_ids)

    req_idxes_cpu = g_pin_mem_manager.gen_from_list(key="req_idxes", data=req_idxes, dtype=torch.int32)
    length_penalty_param_cpu = g_pin_mem_manager.gen_from_list(
        key="length_penalty_param", data=length_penalty_param, dtype=torch.int32
    )
    mask_eos_reqs_cpu = g_pin_mem_manager.gen_from_list(key="mask_eos_reqs", data=mask_eos_reqs, dtype=torch.bool)
    temperatures_cpu = (
        None
        if all_temperature_one
        else g_pin_mem_manager.gen_from_list(key="temperatures", data=temperatures, dtype=torch.float32)
    )
    sampling_backend = get_env_start_args().sampling_backend
    need_top_k_top_p_tensors = (not is_all_greedy) and (not (skip_top_k and skip_top_p))
    need_top_ps_tensor = need_top_k_top_p_tensors and (sampling_backend != "flashinfer" or not skip_top_p)
    need_top_ks_tensor = need_top_k_top_p_tensors and (sampling_backend != "flashinfer" or not skip_top_k)
    top_ps_cpu = (
        g_pin_mem_manager.gen_from_list(key="top_ps", data=top_ps, dtype=torch.float32) if need_top_ps_tensor else None
    )
    top_ks_cpu = (
        g_pin_mem_manager.gen_from_list(key="top_ks", data=top_ks, dtype=torch.int32) if need_top_ks_tensor else None
    )

    if has_invalid_token_ids:
        invalid_token_ids_cpu = g_pin_mem_manager.gen_from_list(
            key="invalid_token_ids", data=invalid_token_ids, dtype=torch.int32
        )
        cu_invalid_token_num_cpu = g_pin_mem_manager.gen_from_list(
            key="cu_invalid_token_num", data=cu_invalid_token_num, dtype=torch.int32
        )

    return (
        req_idxes_cpu.cuda(non_blocking=True),
        temperatures_cpu.cuda(non_blocking=True) if temperatures_cpu is not None else None,
        top_ps_cpu.cuda(non_blocking=True) if top_ps_cpu is not None else None,
        top_ks_cpu.cuda(non_blocking=True) if top_ks_cpu is not None else None,
        length_penalty_param_cpu.cuda(non_blocking=True),
        mask_eos_reqs_cpu.cuda(non_blocking=True),
        invalid_token_ids_cpu.cuda(non_blocking=True) if has_invalid_token_ids else None,
        cu_invalid_token_num_cpu.cuda(non_blocking=True) if has_invalid_token_ids else None,
        is_all_greedy,
        has_invalid_token_ids,
        skip_top_k,
        skip_top_p,
        exist_req_use_random_seed,
    )
