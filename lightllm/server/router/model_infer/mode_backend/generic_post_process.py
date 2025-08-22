import torch
from typing import List
from lightllm.common.basemodel.triton_kernel.apply_penalty import apply_penalty
from lightllm.common.basemodel.triton_kernel.apply_penalty_gpu_cache import apply_penalty_gpu_cache
from lightllm.server.router.model_infer.infer_batch import InferReq, g_infer_context
from lightllm.utils.envs_utils import get_env_start_args


def sample(logits: torch.Tensor, reqs: List[InferReq], eos_id: List[int] = [2]):
    (
        b_req_idx,
        b_temperatures,
        b_top_ps,
        b_top_ks,
        b_length_penalty_param,
        b_mask_eos_reqs,
        is_all_greedy,
    ) = _get_post_sample_tensors(reqs)
    eos_ids = torch.tensor(eos_id, dtype=torch.int32, device="cpu", pin_memory=True).cuda(non_blocking=True)

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
    logits.div_(b_temperatures.view((-1, 1)))
    probs = torch.softmax(logits, dim=-1)

    if is_all_greedy:
        batch_next_token_ids = torch.argmax(logits, -1)
        batch_next_token_probs = torch.gather(probs, dim=1, index=batch_next_token_ids.view(-1, 1))
        return batch_next_token_ids.view(-1), torch.log(batch_next_token_probs).view(-1)

    elif get_env_start_args().sampling_backend == "triton":
        probs_sort, probs_idx = _top_p_top_k(probs, b_top_ps, b_top_ks)
        sampled_index = torch.multinomial(probs_sort, num_samples=1, replacement=True)
        next_token_ids = torch.gather(probs_idx, dim=1, index=sampled_index)
        next_token_logprobs = torch.log(torch.gather(probs_sort, dim=1, index=sampled_index))
        return next_token_ids.view(-1), next_token_logprobs.view(-1)

    elif get_env_start_args().sampling_backend == "sglang_kernel":
        from sgl_kernel import top_k_top_p_sampling_from_probs

        batch_next_token_ids = top_k_top_p_sampling_from_probs(
            probs,
            b_top_ks,
            b_top_ps,
            filter_apply_order="joint",
            check_nan=False,
        )
        int64_batch_next_token_ids = torch.empty_like(batch_next_token_ids, dtype=torch.int64)
        int64_batch_next_token_ids[:] = batch_next_token_ids
        batch_next_token_probs = torch.gather(probs, dim=1, index=int64_batch_next_token_ids.view(-1, 1))
        return batch_next_token_ids.view(-1), torch.log(batch_next_token_probs).view(-1)
    else:
        assert False, "dead path"


def _top_p_top_k(probs: torch.Tensor, top_ps: torch.Tensor, top_ks: torch.Tensor):
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)

    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0

    probs_sort[torch.arange(0, probs.shape[-1], device="cuda").view(1, -1) >= top_ks.view(-1, 1)] = 0.0

    return probs_sort, probs_idx


def _get_post_sample_tensors(reqs: List[InferReq]):
    req_idxes: List[int] = []
    temperatures: List[float] = []
    top_ps: List[float] = []
    top_ks: List[int] = []
    length_penalty_param: List[int] = []
    mask_eos_reqs: List[bool] = []
    is_all_greedy = True

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
        req_idxes.append(req_obj.req_idx)

    req_idxes_cpu = torch.tensor(req_idxes, dtype=torch.int32, device="cpu", pin_memory=True)
    temperatures_cpu = torch.tensor(temperatures, dtype=torch.float, device="cpu", pin_memory=True)
    top_ps_cpu = torch.tensor(top_ps, dtype=torch.float, device="cpu", pin_memory=True)
    top_ks_cpu = torch.tensor(top_ks, dtype=torch.int32, device="cpu", pin_memory=True)
    length_penalty_param_cpu = torch.tensor(length_penalty_param, dtype=torch.int32, device="cpu", pin_memory=True)
    mask_eos_reqs_cpu = torch.tensor(mask_eos_reqs, dtype=torch.bool, device="cpu", pin_memory=True)

    return (
        req_idxes_cpu.cuda(non_blocking=True),
        temperatures_cpu.cuda(non_blocking=True),
        top_ps_cpu.cuda(non_blocking=True),
        top_ks_cpu.cuda(non_blocking=True),
        length_penalty_param_cpu.cuda(non_blocking=True),
        mask_eos_reqs_cpu.cuda(non_blocking=True),
        is_all_greedy,
    )
