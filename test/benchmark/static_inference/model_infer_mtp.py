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
        mtp_model_cfg, _ = PretrainedConfig.get_config_dict(args.mtp_draft_model_dir)
        mtp_model_kvargs.update(
            {
                "weight_dir": args.spec_model_dir,
                "max_total_token_num": main_model.mem_manager.size,
                "disable_chunked_prefill": True,
                "mtp_mode": args.mtp_mode,
                "main_model": main_model,
                "mem_layer_start": main_model.config["num_hidden_layers"] + i * mtp_model_cfg["num_hidden_layers"],
            }
        )
        draft_models.append(Deepseek3MTPModel(mtp_model_kvargs))
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
            "mem_faction": args.mem_fraction,
            "max_req_num": 2000,
            "batch_max_tokens": 2048,
            "run_mode": "normal",
            "max_seq_length": args.max_req_total_len,
            "spec_algo": args.spec_algo,
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


def torch_profile(fn, log_dir=None):
    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
    ) as prof:
        fn()
    if get_current_rank_in_dp() == 0:
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


def run_forward_once(args, input_len, output_len, batch_size, main_model, draft_models, warmup=False):
    import time

    torch.cuda.synchronize()
    prefill_start_time = time.time()

    test_data = np.vstack([np.random.randint(0, 50256, input_len) for _ in range(batch_size)])
    test_data = test_data.reshape(-1)
    test_data = torch.from_numpy(test_data).cuda()

    b_req_idx = torch.tensor(
        [main_model.req_manager.alloc() for _ in range(batch_size)], dtype=torch.int32, device="cuda"
    )
    b_seq_len = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    b_ready_cache_len = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    for i in range(batch_size):
        b_seq_len[i] = input_len

    total_token_num = input_len * batch_size
    mem_indexes = main_model.req_manager.mem_manager.alloc(test_data.shape[0]).cuda()
    # Main model Prefill
    model_input = ModelInput(
        batch_size=batch_size,
        total_token_num=total_token_num,
        max_len_in_batch=input_len,
        input_ids=test_data,
        mem_indexes=mem_indexes,
        b_req_idx=b_req_idx,
        b_seq_len=b_seq_len,
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
    if get_current_rank_in_dp() == 0 and not warmup:
        print("prefill time cost:", (prefill_end_time - prefill_start_time) * 1000)
        print(
            f"Prefill throughput: {batch_size * input_len * args.dp / (prefill_end_time - prefill_start_time)} tokens/s"
        )

    torch.cuda.synchronize()

    decode_input_ids = np.stack(draft_ids, axis=-1).reshape(-1)
    decode_input_ids = torch.from_numpy(decode_input_ids).cuda()

    # build main decode input:
    nopad_b_seq_idx = []
    nopad_b_seq_len = []
    nopad_total_token_num = 0
    nopad_max_len_in_batch = 0

    for i in range(batch_size):
        nopad_b_seq_idx.append(b_req_idx[i])
        seq_len = b_seq_len[i].item()
        nopad_b_seq_len.append(seq_len + 1)
        nopad_total_token_num += seq_len + 1
        nopad_max_len_in_batch = max(nopad_max_len_in_batch, b_seq_len[i] + 1)

        for step in range(len(draft_models)):
            nopad_b_seq_idx.append(b_req_idx[i])
            nopad_b_seq_len.append(seq_len + step + 2)
            nopad_total_token_num += seq_len + step + 2
            nopad_max_len_in_batch = max(nopad_max_len_in_batch, seq_len + step + 2)

    nopad_b_seq_idx = torch.tensor(nopad_b_seq_idx, dtype=torch.int32, device="cuda")
    nopad_b_seq_len = torch.tensor(nopad_b_seq_len, dtype=torch.int32, device="cuda")
    mem_indexes = main_model.req_manager.mem_manager.alloc(batch_size * (len(draft_models) + 1)).cuda()

    model_input = ModelInput(
        batch_size=batch_size * (len(draft_models) + 1),
        total_token_num=nopad_total_token_num,
        max_len_in_batch=nopad_max_len_in_batch,
        input_ids=decode_input_ids,
        mem_indexes=mem_indexes,
        b_req_idx=nopad_b_seq_idx,
        b_seq_len=nopad_b_seq_len,
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
        if i % 100 == 0 or i == output_len - 1:
            step_end_time = time.time()
            if get_current_rank_in_dp() == 0 and not warmup:
                step_time = step_end_time - step_start_time
                print(i, " step cost time:", step_time * 1000)
                print(f"Decode throughput: {batch_size * (len(draft_models) + 1) * args.dp / step_time} tokens/s")

    main_model.mem_manager.free_all()
    main_model.req_manager.free_all()


def tppart_model_infer(args, model_kvargs, batch_sizes, input_len, output_len, ans_queue):
    args = get_env_start_args()
    import triton.profiler as proton
    import torch
    from lightllm.distributed import dist_group_manager
    from lightllm.utils.dist_utils import set_current_device_id

    import torch.distributed as dist

    enable_decode_overlap = args.enable_decode_microbatch_overlap
    group_size = 1
    if enable_decode_overlap or args.enable_prefill_microbatch_overlap:
        group_size = 2
    init_distributed_env(model_kvargs)
    dist_group_manager.create_groups(group_size=group_size)
    model_cfg, _ = PretrainedConfig.get_config_dict(model_kvargs["weight_dir"])
    dist.barrier()

    torch.cuda.empty_cache()

    main_model, _ = get_model(model_cfg, model_kvargs)
    draft_models = init_mtp_model(args, model_kvargs, main_model)
    if isinstance(batch_sizes, int):
        batch_sizes = [batch_sizes]

    for batch_size in batch_sizes:
        # warm up
        run_forward_once(args, input_len, output_len, batch_size, main_model, draft_models, warmup=True)
        torch.cuda.synchronize()
        run_forward_once(args, input_len, output_len, batch_size, main_model, draft_models, warmup=False)
        dist.barrier()

    ans_queue.put(True)
    return
