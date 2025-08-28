import os
import torch
import numpy as np
from multiprocessing import Queue
import multiprocessing
from transformers import PretrainedConfig
from lightllm.utils.dist_utils import init_distributed_env, get_current_rank_in_dp
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.models import get_model
from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.server.core.objs.start_args_type import StartArgs
from torch.profiler import profile, record_function, ProfilerActivity
from lightllm.utils.log_utils import init_logger
from lightllm.models.deepseek_mtp.model import Deepseek3MTPModel
import torch.cuda as cuda

logger = init_logger(__name__)


def init_mtp_model(args: StartArgs, kvargs, main_model):
    mtp_step = args.mtp_step
    draft_models = []

    logger.info(f"Initializing {mtp_step} MTP draft models")

    os.environ["DISABLE_CHECK_MAX_LEN_INFER"] = "1"
    mtp_model_kvargs = kvargs
    mtp_model_kvargs.update(
        {
            "weight_dir": args.mtp_draft_model_dir,
            "max_total_token_num": main_model.mem_manager.size,
            "disable_chunked_prefill": True,
            "mtp_mode": args.mtp_mode,
            "main_model": main_model,
        }
    )
    for i in range(mtp_step):
        try:
            mtp_model_cfg, _ = PretrainedConfig.get_config_dict(args.mtp_draft_model_dir)
            mtp_model_kvargs.update(
                {
                    "weight_dir": args.mtp_draft_model_dir,
                    "max_total_token_num": main_model.mem_manager.size,
                    "disable_chunked_prefill": True,
                    "mtp_mode": args.mtp_mode,
                    "main_model": main_model,
                    "mem_layer_start": main_model.config["num_hidden_layers"] + i * mtp_model_cfg["num_hidden_layers"],
                }
            )
            draft_model = Deepseek3MTPModel(mtp_model_kvargs)
            draft_models.append(draft_model)
            logger.info(f"Successfully initialized draft model {i+1}/{mtp_step}")
        except Exception as e:
            logger.error(f"Failed to initialize draft model {i+1}: {str(e)}")
            raise

    logger.info(f"Successfully initialized all {len(draft_models)} draft models")
    return draft_models


def test_model_inference_mtp(args):
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
            "quant_type": args.quant_type,
            "load_way": "HF",
            "max_total_token_num": args.max_total_token_num,
            "graph_max_len_in_batch": args.max_req_total_len,
            "graph_max_batch_size": args.graph_max_batch_size,
            "mem_fraction": args.mem_fraction,
            "max_req_num": 2000,
            "batch_max_tokens": 2048,
            "run_mode": "normal",
            "max_seq_length": args.max_req_total_len,
            "disable_cudagraph": args.disable_cudagraph,
        }
        proc = multiprocessing.Process(
            target=tppart_model_infer,
            args=(args, model_kvargs, args.batch_size, args.input_len, args.output_len, ans_queue),
        )
        proc.start()
        workers.append(proc)

    for proc in workers:
        proc.join()

    assert not ans_queue.empty()
    while not ans_queue.empty():
        assert ans_queue.get()
    return


def torch_profile(fn, batch_size, log_dir=None):
    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
    ) as prof:
        fn()
    torch.cuda.synchronize()
    if get_current_rank_in_dp() == 0:
        logger.info(f"batch_size {batch_size}\n{prof.key_averages().table(sort_by='cuda_time_total', row_limit=20)}")
        table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20)
        logger.info(table if table else "<empty profile> (no ops recorded)")


def run_forward_once(
    args,
    input_len,
    output_len,
    batch_size,
    main_model,
    draft_models,
    warmup=False,
    enable_torch_profile=False,
    skip_prefill=False,
):
    import time
    import torch.distributed as dist

    dist.barrier()

    torch.cuda.synchronize()
    prefill_start_time = time.time()

    b_req_idx = torch.tensor(
        [main_model.req_manager.alloc() for _ in range(batch_size)], dtype=torch.int32, device="cuda"
    )
    b_mtp_index = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    b_seq_len = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    b_ready_cache_len = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    for i in range(batch_size):
        b_seq_len[i] = input_len

    total_token_num = input_len * batch_size

    if skip_prefill:
        # Skip prefill computation but simulate the state after prefill
        # Generate dummy output tokens as if prefill happened
        draft_ids = []

        # Generate dummy token IDs for main model and draft models
        # Simulate one token output per model (main + draft models)
        for model_idx in range(len(draft_models) + 1):
            # Generate random token IDs as if they were predicted
            dummy_predict_ids = np.random.randint(1000, 10000, (batch_size, 1))
            draft_ids.append(dummy_predict_ids)

        # Update sequence lengths to reflect that prefill "happened"
        # No need to update b_seq_len as it already contains input_len

        if get_current_rank_in_dp() == 0 and not warmup:
            logger.info(f"Skipped prefill phase, simulated {len(draft_ids)} draft outputs")
    else:
        # Generate test data for actual prefill
        test_data = np.vstack([np.random.randint(0, 50256, input_len) for _ in range(batch_size)])
        test_data = test_data.reshape(-1)
        test_data = torch.from_numpy(test_data).cuda()

        # Allocate memory for prefill tokens
        mem_indexes = main_model.req_manager.mem_manager.alloc(test_data.shape[0]).cuda()
        # Main model Prefill
        model_input = ModelInput(
            batch_size=batch_size,
            total_token_num=total_token_num,
            max_len_in_batch=input_len,
            input_ids=test_data,
            b_req_idx=b_req_idx,
            b_mtp_index=b_mtp_index,
            b_seq_len=b_seq_len,
            mem_indexes=mem_indexes,
            is_prefill=True,
            b_ready_cache_len=b_ready_cache_len,
        )

        model_output: ModelOutput = main_model.forward(model_input)
        prob_out = torch.softmax(model_output.logits, dim=-1)
        predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
        predict_ids = predict_ids.detach().cpu().numpy()

        draft_ids = [predict_ids]

        # Draft model Prefill
        # For simplicity, we'll just take the input of main_model to draft model.
        model_input.deepseekv3_mtp_draft_input_hiddens = model_output.deepseekv3_mtp_main_output_hiddens
        for draft_model_id in range(len(draft_models)):
            draft_model = draft_models[draft_model_id]
            model_output = draft_model.forward(model_input)
            prob_out = torch.softmax(model_output.logits, dim=-1)
            predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
            predict_ids = predict_ids.detach().cpu().numpy()
            draft_ids.append(predict_ids)
            model_input.deepseekv3_mtp_draft_input_hiddens = model_output.deepseekv3_mtp_main_output_hiddens

    torch.cuda.synchronize()
    prefill_end_time = time.time()

    rank_id = get_current_rank_in_dp()

    if rank_id == 0 and not warmup and not skip_prefill:
        prefill_time = (prefill_end_time - prefill_start_time) * 1000
        dp_size = getattr(args, "dp", 1)
        throughput = dp_size * batch_size * input_len / (prefill_end_time - prefill_start_time)
        logger.info(f"prefill time cost: {prefill_time:.2f} ms, prefill throughput: {throughput:.2f} tokens/s")

    # Add profiling support for prefill
    if enable_torch_profile and not warmup and not skip_prefill:
        logger.info("Profile Prefill")
        try:
            torch_profile(
                lambda: main_model.forward(model_input),
                batch_size,
                log_dir=f"./logs/forward_prefill_mtp_bs{batch_size}_{rank_id}",
            )
        except Exception as e:
            logger.error(f"Profiling error: {str(e)}")
            # Continue without profiling

    torch.cuda.synchronize()

    decode_input_ids = np.stack(draft_ids, axis=-1).reshape(-1)
    decode_input_ids = torch.from_numpy(decode_input_ids).cuda()

    # build main decode input:
    nopad_b_seq_idx = []
    nopad_b_mtp_index = []
    nopad_b_seq_len = []
    nopad_total_token_num = 0
    nopad_max_len_in_batch = 0

    for i in range(batch_size):
        nopad_b_seq_idx.append(b_req_idx[i])
        nopad_b_mtp_index.append(0)
        seq_len = b_seq_len[i].item()
        nopad_b_seq_len.append(seq_len + 1)
        nopad_total_token_num += seq_len + 1
        nopad_max_len_in_batch = max(nopad_max_len_in_batch, b_seq_len[i] + 1)

        for step in range(len(draft_models)):
            nopad_b_seq_idx.append(b_req_idx[i])
            nopad_b_mtp_index.append(step + 1)
            nopad_b_seq_len.append(seq_len + step + 2)
            nopad_total_token_num += seq_len + step + 2
            nopad_max_len_in_batch = max(nopad_max_len_in_batch, seq_len + step + 2)

    nopad_b_seq_idx = torch.tensor(nopad_b_seq_idx, dtype=torch.int32, device="cuda")
    nopad_b_mtp_index = torch.tensor(nopad_b_mtp_index, dtype=torch.int32, device="cuda")
    nopad_b_seq_len = torch.tensor(nopad_b_seq_len, dtype=torch.int32, device="cuda")
    mem_indexes = main_model.req_manager.mem_manager.alloc(batch_size * (len(draft_models) + 1)).cuda()

    model_input = ModelInput(
        batch_size=batch_size * (len(draft_models) + 1),
        total_token_num=nopad_total_token_num,
        max_len_in_batch=nopad_max_len_in_batch,
        input_ids=decode_input_ids,
        b_req_idx=nopad_b_seq_idx,
        b_mtp_index=nopad_b_mtp_index,
        b_seq_len=nopad_b_seq_len,
        mem_indexes=mem_indexes,
        is_prefill=False,
    )

    # Main decode
    for i in range(0, output_len, len(draft_models) + 1):
        torch.cuda.synchronize()
        step_start_time = time.time()
        model_output = main_model.forward(
            model_input,
        )
        prob_out = torch.softmax(model_output.logits, dim=-1)
        predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)

        # draft decode
        model_input.input_ids = predict_ids.reshape(-1)
        model_input.deepseekv3_mtp_draft_input_hiddens = model_output.deepseekv3_mtp_main_output_hiddens

        for draft_model_id in range(len(draft_models)):
            draft_model = draft_models[draft_model_id]
            model_output = draft_model.forward(
                model_input,
            )
            prob_out = torch.softmax(model_output.logits, dim=-1)
            predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
            model_input.input_ids = predict_ids.reshape(-1)
            model_input.deepseekv3_mtp_draft_input_hiddens = model_output.deepseekv3_mtp_main_output_hiddens

        # accept all draft ids by default.
        model_input.input_ids = predict_ids.reshape(-1)
        model_input.deepseekv3_mtp_draft_input_hiddens = model_output.deepseekv3_mtp_main_output_hiddens
        torch.cuda.synchronize()
        if i % 100 == 0 or i == output_len - (len(draft_models) + 1):
            step_end_time = time.time()
            if rank_id == 0 and not warmup:
                step_time = step_end_time - step_start_time
                dp_size = getattr(args, "dp", 1)
                throughput = dp_size * batch_size * (len(draft_models) + 1) / step_time
                logger.info(f"i: {i}, step cost time: {step_time * 1000:.2f} ms, throughput: {throughput:.2f} tokens/s")

        # Add profiling support for decode on last step
        if enable_torch_profile and not warmup and i == output_len - (len(draft_models) + 1):
            logger.info("Profile Decode")
            try:
                torch_profile(
                    lambda: main_model.forward(model_input),
                    batch_size,
                    log_dir=f"./logs/forward_decode_mtp_bs{batch_size}_{rank_id}",
                )
            except Exception as e:
                logger.error(f"Profiling error: {str(e)}")
                # Continue without profiling

    main_model.mem_manager.free_all()
    main_model.req_manager.free_all()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def tppart_model_infer(args, model_kvargs, batch_sizes, input_len, output_len, ans_queue):
    args = get_env_start_args()
    import triton.profiler as proton
    import torch
    from lightllm.distributed import dist_group_manager
    from lightllm.utils.dist_utils import set_current_device_id

    # Handle batch_sizes as either int or list
    if isinstance(batch_sizes, int):
        batch_sizes = [batch_sizes]
    else:
        # Default batch sizes for comprehensive testing
        batch_sizes = [16, 32, 64]

    logger.info(f"Testing batch sizes: {batch_sizes}")

    import torch.distributed as dist

    enable_decode_overlap = args.enable_decode_microbatch_overlap
    group_size = 1
    if enable_decode_overlap or args.enable_prefill_microbatch_overlap:
        for bs in batch_sizes:
            assert bs % 2 == 0, f"batch size {bs} must be even number for overlap mode"
        group_size = 2
    init_distributed_env(model_kvargs)
    dist_group_manager.create_groups(group_size=group_size)
    model_cfg, _ = PretrainedConfig.get_config_dict(model_kvargs["weight_dir"])
    dist.barrier()

    torch.cuda.empty_cache()

    main_model, _ = get_model(model_cfg, model_kvargs)
    draft_models = init_mtp_model(args, model_kvargs, main_model)
    rank_id = model_kvargs["rank_id"]
    skip_prefill = getattr(args, "skip_prefill", False)
    for batch_size in batch_sizes:
        if rank_id == 0:
            logger.info(f"Testing batch size {batch_size}")

        # warm up
        run_forward_once(
            args,
            input_len,
            10,
            batch_size,
            main_model,
            draft_models,
            warmup=True,
            enable_torch_profile=False,
            skip_prefill=skip_prefill,
        )
        torch.cuda.synchronize()

        # actual test
        enable_profiling = getattr(args, "torch_profile", False)
        run_forward_once(
            args,
            input_len,
            output_len,
            batch_size,
            main_model,
            draft_models,
            warmup=False,
            enable_torch_profile=enable_profiling,
            skip_prefill=skip_prefill,
        )
        if rank_id == 0:
            logger.info("=" * 50)
        dist.barrier()

    ans_queue.put(True)
    return
