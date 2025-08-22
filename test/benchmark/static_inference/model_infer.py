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
from torch.profiler import profile, record_function, ProfilerActivity
from lightllm.utils.log_utils import init_logger
import torch.cuda as cuda

logger = init_logger(__name__)


def test_model_inference(args):
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
            "max_req_num": 2048,
            "batch_max_tokens": 1024,
            "run_mode": "normal",
            "max_seq_length": args.max_req_total_len,
            "disable_cudagraph": args.disable_cudagraph,
            "mode": args.mode,
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


def overlap_prefill(
    model_part,
    batch_size,
    max_len_in_batch,
    input_ids,
    mem_indexes,
    b_req_idx,
    b_mtp_index,
    b_seq_len,
    total_token_num,
    b_ready_cache_len,
):
    _0_batch_size = batch_size // 2
    _0_total_token_num = total_token_num // 2
    _0_max_len_in_batch = max_len_in_batch
    _0_input_ids = input_ids[: total_token_num // 2]
    _0_mem_indexes = mem_indexes[: total_token_num // 2]
    _0_b_req_idx = b_req_idx[: batch_size // 2]
    _0_b_mtp_index = b_mtp_index[: batch_size // 2]
    _0_b_seq_len = b_seq_len[: batch_size // 2]
    _o_b_ready_cache_len = b_ready_cache_len[: batch_size // 2]
    micro_batch1 = ModelInput(
        batch_size=_0_batch_size,
        total_token_num=_0_total_token_num,
        max_len_in_batch=_0_max_len_in_batch,
        input_ids=_0_input_ids,
        b_req_idx=_0_b_req_idx,
        b_mtp_index=_0_b_mtp_index,
        b_seq_len=_0_b_seq_len,
        is_prefill=True,
        b_ready_cache_len=_o_b_ready_cache_len,
        multimodal_params={},
        mem_indexes_cpu=_0_mem_indexes,
    )

    _1_batch_size = batch_size - batch_size // 2
    _1_total_token_num = total_token_num - total_token_num // 2
    _1_max_len_in_batch = max_len_in_batch
    _1_input_ids = input_ids[total_token_num // 2 :]
    _1_mem_indexes = mem_indexes[total_token_num // 2 :]
    _1_b_req_idx = b_req_idx[batch_size // 2 :]
    _1_b_mtp_index = b_mtp_index[batch_size // 2 :]
    _1_b_seq_len = b_seq_len[batch_size // 2 :]
    _1_b_ready_cache_len = b_ready_cache_len[batch_size // 2 :]

    micro_batch2 = ModelInput(
        batch_size=_1_batch_size,
        total_token_num=_1_total_token_num,
        max_len_in_batch=_1_max_len_in_batch,
        input_ids=_1_input_ids,
        b_req_idx=_1_b_req_idx,
        b_mtp_index=_1_b_mtp_index,
        b_seq_len=_1_b_seq_len,
        is_prefill=True,
        b_ready_cache_len=_1_b_ready_cache_len,
        multimodal_params={},
        mem_indexes_cpu=_1_mem_indexes,
    )

    output, output1 = model_part.microbatch_overlap_prefill(micro_batch1, micro_batch2)
    logits = output.logits
    logits1 = output1.logits
    return torch.cat((logits, logits1), dim=0)


def overlap_decode(
    model_part, batch_size, max_len_in_batch, input_ids, mem_indexes, b_req_idx, b_mtp_index, b_seq_len, total_token_num
):
    _0_batch_size = batch_size // 2
    _0_total_token_num = total_token_num // 2
    _0_max_len_in_batch = max_len_in_batch
    _0_input_ids = input_ids[: batch_size // 2]
    _0_mem_indexes = mem_indexes[: batch_size // 2]
    _0_b_req_idx = b_req_idx[: batch_size // 2]
    _0_b_mtp_index = b_mtp_index[: batch_size // 2]
    _0_b_seq_len = b_seq_len[: batch_size // 2]
    micro_batch1 = ModelInput(
        batch_size=_0_batch_size,
        total_token_num=_0_total_token_num,
        max_len_in_batch=_0_max_len_in_batch,
        input_ids=_0_input_ids,
        b_req_idx=_0_b_req_idx,
        b_mtp_index=_0_b_mtp_index,
        b_seq_len=_0_b_seq_len,
        mem_indexes_cpu=_0_mem_indexes,
    )

    _1_batch_size = batch_size - batch_size // 2
    _1_total_token_num = total_token_num - total_token_num // 2
    _1_max_len_in_batch = max_len_in_batch
    _1_input_ids = input_ids[batch_size // 2 :]
    _1_mem_indexes = mem_indexes[batch_size // 2 :]
    _1_b_req_idx = b_req_idx[batch_size // 2 :]
    _1_b_mtp_index = b_mtp_index[batch_size // 2 :]
    _1_b_seq_len = b_seq_len[batch_size // 2 :]

    micro_batch2 = ModelInput(
        batch_size=_1_batch_size,
        total_token_num=_1_total_token_num,
        max_len_in_batch=_1_max_len_in_batch,
        input_ids=_1_input_ids,
        b_req_idx=_1_b_req_idx,
        b_mtp_index=_1_b_mtp_index,
        b_seq_len=_1_b_seq_len,
        mem_indexes_cpu=_1_mem_indexes,
    )

    output, output1 = model_part.microbatch_overlap_decode(micro_batch1, micro_batch2)
    logits = output.logits
    logits1 = output1.logits
    return torch.cat((logits, logits1), dim=0)


def prefill(
    model_part,
    batch_size,
    max_len_in_batch,
    input_ids,
    mem_indexes,
    b_req_idx,
    b_mtp_index,
    b_seq_len,
    total_token_num,
    b_ready_cache_len,
):
    b_mtp_index = torch.zeros(batch_size, dtype=torch.int32, device="cpu")
    model_input = ModelInput(
        batch_size=batch_size,
        total_token_num=total_token_num,
        max_len_in_batch=max_len_in_batch,
        input_ids=input_ids,
        b_req_idx=b_req_idx,
        b_seq_len=b_seq_len,
        b_mtp_index=b_mtp_index,
        mem_indexes_cpu=mem_indexes,
        is_prefill=True,
        b_ready_cache_len=b_ready_cache_len,  # b_ready_cache_len
    )

    model_output = model_part.forward(model_input)
    return model_output.logits


def decode(
    model_part, batch_size, max_len_in_batch, input_ids, mem_indexes, b_req_idx, b_mtp_index, b_seq_len, total_token_num
):
    model_input = ModelInput(
        batch_size=batch_size,
        total_token_num=total_token_num,
        max_len_in_batch=max_len_in_batch,
        input_ids=input_ids,
        b_req_idx=b_req_idx,
        b_seq_len=b_seq_len,
        b_mtp_index=b_mtp_index,
        mem_indexes_cpu=mem_indexes,
        is_prefill=False,
    )
    model_output = model_part.forward(model_input)
    return model_output.logits


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
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


def run_forward_once(
    model_kvargs, input_len, output_len, batch_size, model_part, enable_overlap, enable_torch_profile=False
):
    test_data = np.vstack([np.random.randint(0, 50256, input_len) for _ in range(batch_size)])
    test_data = test_data.reshape(-1)
    test_data = torch.from_numpy(test_data)
    import torch.distributed as dist

    dist.barrier()
    import time

    dp_size = model_kvargs["dp_size"]

    torch.cuda.synchronize()
    prefill_start_time = time.time()

    b_req_idx = torch.tensor(
        [model_part.req_manager.alloc() for _ in range(batch_size)], dtype=torch.int32, device="cpu"
    )
    b_seq_len = torch.zeros(batch_size, dtype=torch.int32, device="cpu")
    b_ready_cache_len = torch.zeros(batch_size, dtype=torch.int32, device="cpu")
    for i in range(batch_size):
        b_seq_len[i] = input_len

    total_token_num = batch_size * input_len
    mem_indexes = model_part.req_manager.mem_manager.alloc(test_data.shape[0])
    b_mtp_index = torch.zeros(batch_size, dtype=torch.int32, device="cpu")
    rank_id = model_kvargs["rank_id"]

    if enable_overlap:
        prefill_fn = overlap_prefill
        decode_fn = overlap_decode
    else:
        prefill_fn = prefill
        decode_fn = decode

    logits = prefill_fn(
        model_part,
        batch_size,
        input_len,
        test_data,
        mem_indexes,
        b_req_idx,
        b_mtp_index,
        b_seq_len,
        total_token_num,
        b_ready_cache_len,  # b_ready_cache_len
    )

    prob_out = torch.softmax(logits, dim=-1)
    predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
    _ = predict_ids.detach().cpu().numpy()

    torch.cuda.synchronize()

    if rank_id == 0:
        print(
            f"prefill time cost: {(time.time() - prefill_start_time) * 1000}, "
            f"prefill throughput: {dp_size * batch_size * input_len / (time.time() - prefill_start_time)} tokens/s"
        )

    if enable_torch_profile:
        print("Profile Prefill")
        try:
            torch_profile(
                lambda: prefill_fn(
                    model_part,
                    batch_size,
                    input_len,
                    test_data,
                    mem_indexes,
                    b_req_idx,
                    b_mtp_index,
                    b_seq_len,
                    total_token_num,
                    b_ready_cache_len,  # b_ready_cache_len
                ),
                log_dir=f"./logs/forward_prefill_{model_kvargs['rank_id']}",
            )
        except Exception as e:
            print(str(e))
            raise

    for i in range(output_len):
        torch.cuda.synchronize()
        step_start = time.time()
        total_token_num += batch_size
        b_seq_len += 1
        mem_indexes = model_part.req_manager.mem_manager.alloc(predict_ids.shape[0])
        max_len_in_batch = input_len + i + 1
        logits = decode_fn(
            model_part,
            batch_size,
            max_len_in_batch,
            predict_ids.view(-1),
            mem_indexes,
            b_req_idx,
            b_mtp_index,
            b_seq_len,
            total_token_num,
        )
        if enable_torch_profile and i == output_len - 1:
            try:
                torch_profile(
                    lambda: decode_fn(
                        model_part,
                        batch_size,
                        max_len_in_batch,
                        predict_ids.view(-1),
                        mem_indexes,
                        b_req_idx,
                        b_mtp_index,
                        b_seq_len,
                        total_token_num,
                    ),
                    log_dir=f"./logs/forward_decode_{model_kvargs['rank_id']}",
                )
            except Exception as e:
                print(str(e))
                raise

        prob_out = torch.softmax(logits, dim=-1)
        predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
        _ = predict_ids.detach().cpu().numpy()
        torch.cuda.synchronize()
        if i % 100 == 0 or i == output_len - 1:
            if rank_id == 0:
                print(
                    f"i: {i}, step cost time: {(time.time() - step_start) * 1000} ms, "
                    f"throughput: {dp_size * batch_size / (time.time() - step_start)} tokens/s"
                )

    model_part.mem_manager.free_all()
    model_part.req_manager.free_all()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def tppart_model_infer(args, model_kvargs, batch_size, input_len, output_len, ans_queue):
    args = get_env_start_args()
    import triton.profiler as proton
    import torch
    from lightllm.distributed import dist_group_manager
    from lightllm.utils.dist_utils import set_current_device_id

    if isinstance(batch_size, int):
        batch_size = [batch_size]
    else:
        batch_size = [2, 8, 16, 32, 64, 128]
    print(batch_size)

    import torch.distributed as dist

    enable_decode_overlap = args.enable_decode_microbatch_overlap
    group_size = 1
    if enable_decode_overlap or args.enable_prefill_microbatch_overlap:
        for bs in batch_size:
            assert bs % 2 == 0, "batch size must be even number"
        group_size = 2
    init_distributed_env(model_kvargs)
    dist_group_manager.create_groups(group_size=group_size)
    model_cfg, _ = PretrainedConfig.get_config_dict(model_kvargs["weight_dir"])
    dist.barrier()

    torch.cuda.empty_cache()
    enable_overlap = args.enable_decode_microbatch_overlap or args.enable_prefill_microbatch_overlap

    model_part, _ = get_model(model_cfg, model_kvargs)

    rank_id = model_kvargs["rank_id"]
    for b in batch_size:
        if rank_id == 0:
            print(f"Testing batch size {b}")

        # warm up
        run_forward_once(
            model_kvargs,
            input_len,
            output_len=10,
            batch_size=b,
            model_part=model_part,
            enable_overlap=enable_overlap,
            enable_torch_profile=False,
        )

        # test
        run_forward_once(
            model_kvargs,
            input_len,
            output_len,
            batch_size=b,
            model_part=model_part,
            enable_overlap=enable_overlap,
            enable_torch_profile=args.torch_profile,
        )
        if rank_id == 0:
            print("=" * 50)

    ans_queue.put(True)

    return
