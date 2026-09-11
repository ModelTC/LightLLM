import argparse
import json
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

from lightllm.common.basemodel.batch_objs import ModelInput
from lightllm.distributed.communication_op import dist_group_manager
from lightllm.models.pi0.model import Pi0ActionExpertModel, Pi0VLMModel
from lightllm.server.understanding_transfer import (
    PrefixKVHandle,
    gather_dense_prefix_kv,
)
from lightllm.models.pi0.tokenizer import Pi0Tokenizer, resolve_tokenizer_path
from lightllm.models.pi0.visual import Pi0VisionEncoder
from lightllm.models.pi05.model import Pi05ActionExpertModel, Pi05VLMModel
from lightllm.models.pi05.tokenizer import Pi05Tokenizer
from openpi_torch_reference import (
    OpenPiTorchReference,
    reference_tokenize,
    reference_vision_encode,
)
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.utils.dist_utils import (
    set_current_device_id,
    set_current_rank_in_dp,
    set_current_rank_in_node,
    set_dp_rank_in_node,
    set_dp_size,
    set_dp_world_size,
    set_global_dp_rank,
    set_global_rank,
    set_global_world_size,
    set_node_world_size,
)
from lightllm.utils.envs_utils import (
    get_unique_server_name,
    set_env_start_args,
    set_unique_server_name,
)
from lightllm.utils.config_utils import get_pi_config, with_pi_config_overrides
from lightllm.utils.torch_dtype_utils import get_torch_dtype


def _init_single_rank_model_runtime(args: StartArgs, device_id: int) -> None:
    """Minimal rank setup for this direct model-vs-reference accuracy test."""

    set_env_start_args(args)
    set_global_rank(0)
    set_global_world_size(1)
    set_global_dp_rank(0)
    set_dp_size(1)
    set_dp_world_size(1)
    set_dp_rank_in_node(0)
    set_current_rank_in_dp(0)
    set_current_rank_in_node(0)
    set_node_world_size(1)
    set_current_device_id(device_id)
    torch.cuda.set_device(device_id)
    if not dist.is_initialized():
        init_file = f"/tmp/{get_unique_server_name()}_accuracy_{os.getpid()}_torch_dist"
        dist.init_process_group(
            "nccl",
            init_method=f"file://{init_file}",
            rank=0,
            world_size=1,
            device_id=torch.device(f"cuda:{device_id}"),
        )
    if len(dist_group_manager) == 0:
        dist_group_manager.create_groups(group_size=1)


def _run_lightllm_prefix_prefill(
    model: Pi0VLMModel,
    *,
    image_embeds: list[torch.Tensor],
    image_masks: list[torch.Tensor],
    tokenized_prompt: torch.Tensor,
    tokenized_prompt_mask: torch.Tensor,
    prefix_req_indexes: torch.Tensor,
    prefix_mem_indexes: torch.Tensor,
) -> None:
    """Build the packed ModelInput used only by this direct accuracy oracle."""

    batch_size = tokenized_prompt.shape[0]
    device = model.pre_post_weight.wte_weight_.weight.device
    dtype = model.data_type
    hidden_size = model.config["hidden_size"]
    cache_rows = []
    packed_ids = []
    multimodal_params = []
    sequence_lengths = []
    next_token_id = 100_000_000
    cache_token_count = 0
    for batch_index in range(batch_size):
        images = []
        image_token_ids = []
        for embedding, mask in zip(image_embeds, image_masks, strict=True):
            if not bool(mask[batch_index].item()):
                continue
            image_embedding = embedding[batch_index].to(
                device=device,
                dtype=dtype,
                non_blocking=True,
            )
            token_num = image_embedding.shape[0]
            cache_rows.append(image_embedding)
            images.append(
                {
                    "token_id": next_token_id,
                    "token_num": token_num,
                    "start_index_in_embed_cache": cache_token_count,
                }
            )
            image_token_ids.append(
                torch.arange(
                    next_token_id,
                    next_token_id + token_num,
                    dtype=torch.long,
                    device=tokenized_prompt.device,
                )
            )
            next_token_id += token_num
            cache_token_count += token_num

        text_ids = tokenized_prompt[batch_index, tokenized_prompt_mask[batch_index].bool()]
        input_ids = torch.cat([*image_token_ids, text_ids])
        packed_ids.append(input_ids)
        sequence_lengths.append(input_ids.numel())
        multimodal_params.append({"images": images, "audios": []})

    embed_cache = (
        torch.cat(cache_rows).unsqueeze(1)
        if cache_rows
        else torch.empty((0, 1, hidden_size), dtype=dtype, device=device)
    )
    from lightllm.server.router.model_infer.infer_batch import g_infer_context

    g_infer_context.cpu_embed_cache_client = SimpleNamespace(
        cpu_embed_cache_tensor=embed_cache
    )

    b_seq_len = torch.tensor(sequence_lengths, dtype=torch.int32)
    b_ready_cache_len = torch.zeros_like(b_seq_len)
    model.forward(
        ModelInput(
            batch_size=batch_size,
            total_token_num=sum(sequence_lengths),
            max_q_seq_len=max(sequence_lengths),
            max_kv_seq_len=max(sequence_lengths),
            max_cache_len=0,
            prefix_total_token_num=0,
            input_ids=torch.cat(packed_ids),
            b_req_idx=prefix_req_indexes.to(dtype=torch.int32),
            b_mtp_index=torch.zeros(batch_size, dtype=torch.int32),
            b_seq_len=b_seq_len,
            mem_indexes=None,
            mem_indexes_cpu=prefix_mem_indexes,
            is_prefill=True,
            b_ready_cache_len=b_ready_cache_len,
            b_prefill_start_loc=b_seq_len.cumsum(0) - b_seq_len,
            multimodal_params=multimodal_params,
            b_prefill_has_output_cpu=[True] * batch_size,
        )
    )


def _benchmark_cuda(fn, *, warmup: int, runs: int, device: torch.device) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    elapsed = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        elapsed.append(start.elapsed_time(end))
    return elapsed


def _timing_summary(values: list[float]) -> dict[str, float]:
    samples = torch.tensor(values, dtype=torch.float64)
    return {
        "mean_ms": samples.mean().item(),
        "p50_ms": samples.quantile(0.5).item(),
        "p90_ms": samples.quantile(0.9).item(),
        "min_ms": samples.min().item(),
        "max_ms": samples.max().item(),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Deterministic OpenPI vs LightLLM pi0 action diff")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--action-horizon", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=2)
    parser.add_argument("--num-images", type=int, default=3)
    parser.add_argument(
        "--valid-images",
        type=int,
        default=None,
        help=(
            "Mark only the first N image slots valid. This models OpenPI "
            "domains such as LIBERO/DROID that pad the third slot."
        ),
    )
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=2e-5)
    parser.add_argument("--prefix-kv-atol", type=float, default=1e-3)
    parser.add_argument("--vision-atol", type=float, default=2e-5)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--enable-action-cudagraph", action="store_true")
    parser.add_argument("--benchmark-warmup", type=int, default=0)
    parser.add_argument("--benchmark-runs", type=int, default=0)
    parser.add_argument("--profile-action", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(1234)
    config = with_pi_config_overrides(
        get_pi_config(args.model_dir, dtype=args.dtype),
        action_horizon=args.action_horizon,
        num_denoise_steps=args.num_steps,
    )
    device = torch.device(args.device)
    start_args = StartArgs(
        run_mode="normal",
        model_dir=config["model_dir"],
        data_type=config["dtype"],
        max_total_token_num=32768,
        running_max_req_size=max(args.batch_size, 4),
    )
    start_args.hardware_platform = "cuda"
    # This direct oracle bypasses the normal CLI parser, which owns this
    # pre-existing runtime option.
    start_args.performance_mode = None
    start_args.enable_torch_fallback = False
    start_args.enable_triton_fallback = False
    start_args.disable_cudagraph = True
    if get_torch_dtype(config["dtype"]) is torch.float32:
        start_args.llm_prefill_att_backend = ["triton"]
        start_args.llm_decode_att_backend = ["triton"]
    set_unique_server_name(start_args)
    _init_single_rank_model_runtime(start_args, device.index or 0)
    os.environ["DISABLE_CHECK_MAX_LEN_INFER"] = "1"
    vision = Pi0VisionEncoder(config, device=device)
    image = torch.arange(180 * 260 * 3, dtype=torch.int64, device=device)
    image = (image % 256).to(torch.uint8).reshape(1, 180, 260, 3)
    image = image.expand(args.batch_size, -1, -1, -1).contiguous()
    lightllm_image_embedding = vision.encode(image)
    stage_benchmark = {}
    if args.benchmark_runs:
        profile_images = image.repeat(args.num_images, 1, 1, 1)
        vision_ms = _benchmark_cuda(
            lambda encoder=vision: encoder.encode(profile_images),
            warmup=args.benchmark_warmup,
            runs=args.benchmark_runs,
            device=device,
        )
        stage_benchmark["vision"] = _timing_summary(vision_ms)
    del vision
    torch.cuda.empty_cache()
    reference_image_embedding = reference_vision_encode(config, image, device=device)
    vision_diff = (lightllm_image_embedding - reference_image_embedding).abs()
    vision_max_abs = vision_diff.max().item()
    vision_mean_abs = vision_diff.mean().item()
    lightllm_image_embeds = [lightllm_image_embedding.clone() for _ in range(args.num_images)]
    reference_image_embeds = [reference_image_embedding.clone() for _ in range(args.num_images)]
    image_masks = []
    for image_index in range(args.num_images):
        mask = torch.ones(args.batch_size, dtype=torch.bool, device=device)
        if args.valid_images is not None:
            if not 0 < args.valid_images <= args.num_images:
                raise ValueError("valid-images must be in [1, num-images]")
            if image_index >= args.valid_images:
                mask.fill_(False)
        elif args.batch_size > 1 and image_index == args.num_images - 1:
            mask[-1] = False
        image_masks.append(mask)
    observation_state = torch.linspace(-1.0, 1.0, config["action_dim"], device=device)[None].expand(args.batch_size, -1)
    tokenizer_path = resolve_tokenizer_path(args.model_dir, args.tokenizer_path)
    prompts = [
        "pick_up\nthe block" if index % 2 == 0 else "move the block to the left" for index in range(args.batch_size)
    ]
    if config["model_type"] == "pi05":
        lightllm_tokenizer = Pi05Tokenizer(
            config["tokenizer_max_length"], tokenizer_path
        )
        lightllm_tokens, lightllm_token_mask = lightllm_tokenizer.tokenize(
            prompts,
            states=observation_state.cpu(),
        )
        vlm_class = Pi05VLMModel
        expert_class = Pi05ActionExpertModel
    else:
        lightllm_tokenizer = Pi0Tokenizer(
            config["tokenizer_max_length"], tokenizer_path
        )
        lightllm_tokens, lightllm_token_mask = lightllm_tokenizer.tokenize(prompts)
        vlm_class = Pi0VLMModel
        expert_class = Pi0ActionExpertModel
    reference_tokens, reference_token_mask = reference_tokenize(
        config,
        prompts,
        observation_state,
        tokenizer_path,
    )
    token_ids_equal = torch.equal(lightllm_tokens, reference_tokens) and torch.equal(
        lightllm_token_mask, reference_token_mask
    )
    lightllm_tokens = lightllm_tokens.to(device)
    lightllm_token_mask = lightllm_token_mask.to(device)
    reference_tokens = reference_tokens.to(device)
    reference_token_mask = reference_token_mask.to(device)
    token_length = config["tokenizer_max_length"]
    prefix_length = args.num_images * lightllm_image_embedding.shape[1] + token_length
    suffix_length = args.action_horizon + (0 if config["model_type"] == "pi05" else 1)
    prefix_seq_lens = lightllm_token_mask.sum(dim=-1, dtype=torch.int32)
    for mask in image_masks:
        prefix_seq_lens += mask.to(torch.int32) * lightllm_image_embedding.shape[1]
    max_sequence_length = prefix_length + suffix_length
    max_total_tokens = max(
        max_sequence_length,
        int(prefix_seq_lens.sum().item()) + args.batch_size * suffix_length + 64,
    )
    lightllm_vlm = vlm_class(
        {
            "vla_config": config,
            "run_mode": "normal",
            "weight_dir": config["model_dir"],
            "max_total_token_num": max_total_tokens,
            "batch_max_tokens": int(prefix_seq_lens.sum().item()),
            "load_way": "HF",
            "max_req_num": max(args.batch_size + 1, 4),
            "max_seq_length": max_sequence_length,
            "return_all_prompt_logics": False,
            "disable_cudagraph": True,
            "graph_max_batch_size": args.batch_size,
            "graph_max_len_in_batch": max_sequence_length,
            "quant_type": "none",
            "mem_fraction": 0.8,
        }
    )
    cache = lightllm_vlm.mem_manager
    req_manager = lightllm_vlm.req_manager
    prefix_req_indexes = torch.tensor([req_manager.alloc() for _ in range(args.batch_size)], dtype=torch.int32)
    mem_indexes = cache.alloc(int(prefix_seq_lens.sum().item()))

    _run_lightllm_prefix_prefill(
        lightllm_vlm,
        image_embeds=lightllm_image_embeds,
        image_masks=image_masks,
        tokenized_prompt=lightllm_tokens,
        tokenized_prompt_mask=lightllm_token_mask,
        prefix_req_indexes=prefix_req_indexes,
        prefix_mem_indexes=mem_indexes,
    )
    prefix_handles = tuple(
        PrefixKVHandle(
            batch_index,
            batch_index,
            int(req_index),
            int(seq_len),
        )
        for batch_index, (req_index, seq_len) in enumerate(
            zip(prefix_req_indexes.tolist(), prefix_seq_lens.tolist(), strict=True)
        )
    )
    dense_prefix = gather_dense_prefix_kv(
        cache,
        req_manager.req_to_token_indexs,
        prefix_handles,
    )
    split_dense_prefixes = [
        gather_dense_prefix_kv(
            cache,
            req_manager.req_to_token_indexs,
            handle,
        )
        for handle in prefix_handles
    ]
    if args.benchmark_runs:
        dense_gather_ms = _benchmark_cuda(
            lambda: gather_dense_prefix_kv(
                cache,
                req_manager.req_to_token_indexs,
                prefix_handles,
            ),
            warmup=args.benchmark_warmup,
            runs=args.benchmark_runs,
            device=device,
        )
        stage_benchmark["dense_prefix_gather"] = _timing_summary(
            dense_gather_ms
        )
        prefill_ms = _benchmark_cuda(
            lambda model=lightllm_vlm: _run_lightllm_prefix_prefill(
                model,
                image_embeds=lightllm_image_embeds,
                image_masks=image_masks,
                tokenized_prompt=lightllm_tokens,
                tokenized_prompt_mask=lightllm_token_mask,
                prefix_req_indexes=prefix_req_indexes,
                prefix_mem_indexes=mem_indexes,
            ),
            warmup=args.benchmark_warmup,
            runs=args.benchmark_runs,
            device=device,
        )
        stage_benchmark["vlm_prefill"] = _timing_summary(prefill_ms)
    del lightllm_vlm
    torch.cuda.empty_cache()

    reference = OpenPiTorchReference(config, device=device)
    reference_cache, reference_pad_mask = reference.prefill(
        reference_image_embeds,
        image_masks,
        reference_tokens,
        reference_token_mask,
    )
    reference_kv = reference_cache.to_legacy_cache()
    prefix_kv_max_abs = 0.0
    prefix_kv_layer_max_abs = []
    for layer_index, (key, value) in enumerate(reference_kv):
        layer_diff = 0.0
        token_offset = 0
        for batch_index, seq_len in enumerate(prefix_seq_lens.tolist()):
            pages = mem_indexes[token_offset : token_offset + seq_len].long()
            cached = cache.kv_buffer[layer_index, pages]
            reference_key = key[batch_index, 0, reference_pad_mask[batch_index]]
            reference_value = value[batch_index, 0, reference_pad_mask[batch_index]]
            layer_diff = max(
                layer_diff,
                (cached[:, 0] - reference_key).abs().max().item(),
                (cached[:, 1] - reference_value).abs().max().item(),
            )
            token_offset += seq_len
        prefix_kv_layer_max_abs.append(layer_diff)
        prefix_kv_max_abs = max(prefix_kv_max_abs, layer_diff)

    state = None if config["model_type"] == "pi05" else observation_state
    noise = torch.randn(
        args.batch_size, args.action_horizon, config["max_action_dim"], device=device
    )
    reference_actions = reference.sample_actions(
        reference_cache,
        reference_pad_mask,
        state,
        noise.clone(),
        args.num_steps,
    )
    del reference
    torch.cuda.empty_cache()

    target_row_indexes = prefix_req_indexes.to(device=req_manager.req_to_token_indexs.device, dtype=torch.long)
    target_logical_rows_before = req_manager.req_to_token_indexs[
        target_row_indexes
    ].clone()
    lightllm_expert = expert_class(
        config,
        max_seq_length=max_sequence_length,
        device=device,
        enable_cuda_graph=args.enable_action_cudagraph,
    )
    action_graph_enabled = lightllm_expert.enable_cuda_graph
    eager_actions = None
    if action_graph_enabled:
        lightllm_expert.enable_cuda_graph = False
        eager_output = lightllm_expert.forward(
            prefix_kv=dense_prefix,
            state=state,
            noise=noise.clone(),
            num_steps=args.num_steps,
            action_dim=config["action_dim"],
            action_horizon=args.action_horizon,
        )
        eager_actions = eager_output.actions
        lightllm_expert.enable_cuda_graph = True

    output = lightllm_expert.forward(
        prefix_kv=dense_prefix,
        state=state,
        noise=noise.clone(),
        num_steps=args.num_steps,
        action_dim=config["action_dim"],
        action_horizon=args.action_horizon,
    )
    action_diff = (output.actions - reference_actions).abs()
    batch_split_diff = None
    if args.batch_size > 1:
        split_actions = []
        for batch_index in range(args.batch_size):
            split_output = lightllm_expert.forward(
                prefix_kv=split_dense_prefixes[batch_index],
                state=None if state is None else state[batch_index : batch_index + 1],
                noise=noise[batch_index : batch_index + 1].clone(),
                num_steps=args.num_steps,
                action_dim=config["action_dim"],
                action_horizon=args.action_horizon,
            )
            # CUDA Graph outputs are stable until the same graph key replays.
            # Preserve each row before evaluating the next split request.
            split_actions.append(split_output.actions.clone())
        batch_split_diff = (output.actions - torch.cat(split_actions)).abs()

    benchmark_ms = []
    if args.benchmark_runs:
        for _ in range(args.benchmark_warmup):
            warmup_output = lightllm_expert.forward(
                prefix_kv=dense_prefix,
                state=state,
                noise=noise,
                num_steps=args.num_steps,
                action_dim=config["action_dim"],
                action_horizon=args.action_horizon,
            )
            torch.cuda.synchronize(device)
            warmup_output.resolve_timing()
        for _ in range(args.benchmark_runs):
            benchmark_output = lightllm_expert.forward(
                prefix_kv=dense_prefix,
                state=state,
                noise=noise,
                num_steps=args.num_steps,
                action_dim=config["action_dim"],
                action_horizon=args.action_horizon,
            )
            torch.cuda.synchronize(device)
            benchmark_ms.append(benchmark_output.resolve_timing())
    if args.profile_action:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        ) as action_profile:
            profile_output = lightllm_expert.forward(
                prefix_kv=dense_prefix,
                state=state,
                noise=noise,
                num_steps=args.num_steps,
                action_dim=config["action_dim"],
                action_horizon=args.action_horizon,
            )
            torch.cuda.synchronize(device)
            profile_output.resolve_timing()
        print(
            action_profile.key_averages(group_by_input_shape=True).table(
                sort_by="self_cuda_time_total",
                row_limit=40,
            )
        )
    target_logical_mapping_unchanged = torch.equal(
        req_manager.req_to_token_indexs[target_row_indexes],
        target_logical_rows_before,
    )
    result = {
        "model_type": config["model_type"],
        "batch_size": args.batch_size,
        "action_horizon": args.action_horizon,
        "num_steps": args.num_steps,
        "token_ids_equal": token_ids_equal,
        "vision_max_abs": vision_max_abs,
        "vision_mean_abs": vision_mean_abs,
        "prefix_kv_max_abs": prefix_kv_max_abs,
        "prefix_kv_layer_max_abs": prefix_kv_layer_max_abs,
        "actions_max_abs": action_diff.max().item(),
        "actions_mean_abs": action_diff.mean().item(),
        "actions_batch_max_abs": action_diff.amax(dim=(1, 2)).tolist(),
        "actions_batch_mean_abs": action_diff.mean(dim=(1, 2)).tolist(),
        "atol": args.atol,
        "rtol": args.rtol,
        "prefix_kv_atol": args.prefix_kv_atol,
        "vision_atol": args.vision_atol,
        "action_logical_mapping": "shared_req_to_token",
        "target_logical_mapping_unchanged": target_logical_mapping_unchanged,
        "action_cudagraph": action_graph_enabled,
    }
    if eager_actions is not None:
        graph_eager_diff = (output.actions - eager_actions).abs()
        result["action_graph_eager_max_abs"] = graph_eager_diff.max().item()
        result["action_graph_eager_mean_abs"] = graph_eager_diff.mean().item()
    if batch_split_diff is not None:
        result["action_batch_split_max_abs"] = batch_split_diff.max().item()
    if benchmark_ms:
        stage_benchmark["action_expert"] = _timing_summary(benchmark_ms)
        result["benchmark"] = {
            "warmup": args.benchmark_warmup,
            "runs": args.benchmark_runs,
            "stages": stage_benchmark,
        }
    print(json.dumps(result, indent=2))
    if not token_ids_equal:
        raise AssertionError("LightLLM and OpenPI tokenization differ")
    if vision_max_abs > args.vision_atol:
        raise AssertionError(f"vision max abs diff {vision_max_abs} exceeds tolerance {args.vision_atol}")
    if prefix_kv_max_abs > args.prefix_kv_atol:
        raise AssertionError(f"prefix KV max abs diff {prefix_kv_max_abs} exceeds tolerance {args.prefix_kv_atol}")
    if not target_logical_mapping_unchanged:
        raise AssertionError("action sampling mutated the target logical KV mapping")
    torch.testing.assert_close(output.actions, reference_actions, atol=args.atol, rtol=args.rtol)
    torch.cuda.synchronize()
    del lightllm_expert
    cache.free(mem_indexes)
    for req_index in prefix_req_indexes.tolist():
        req_manager.free_req(req_index)
    allocator_stat = cache.allocator.shared_can_use_token_num
    allocator_stat.arr = None
    try:
        allocator_stat.shm.unlink()
    except FileNotFoundError:
        pass
    allocator_stat.shm.close()
    torch.cuda.synchronize()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
