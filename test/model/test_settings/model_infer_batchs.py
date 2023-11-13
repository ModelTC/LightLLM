import os
import numpy as np
from multiprocessing import Queue
import multiprocessing

def test_model_inference(world_size, model_dir, model_class, batch_sizes, input_len, output_len, mode, log_path):
    ans_queue = Queue()
    workers = []
    for rank_id in range(world_size):
        model_kvargs = {
            "tp_rank": rank_id,
            "world_size": world_size,
            "weight_dir": model_dir,
            "max_total_token_num":0 * (input_len + output_len),
            "load_way": "HF",
            "mode": mode,
            "max_req_num": max(batch_sizes),
            "max_seq_length": (input_len + output_len)
        }
        
        proc = multiprocessing.Process(target=tppart_model_infer, args=(model_class, model_kvargs, batch_sizes, input_len, output_len, ans_queue, log_path))
        proc.start()
        workers.append(proc)

    while True:
        import time
        exist_dead = any([not proc.is_alive() for proc in workers])
        if exist_dead:
            time.sleep(4)
            exist_err = any([proc.exitcode != 0 for proc in workers])
            if exist_err:
                return -1
            else:
                break
        time.sleep(1)            
        
    while not ans_queue.empty():
        if not ans_queue.get():
            return -1
    return 0


def tppart_model_infer(model_class, model_kvargs, batch_sizes, input_len, output_len, ans_queue, log_path):
    assert log_path is not None
    need_run_batch_sizes = []
    for batch_size in batch_sizes:
        new_log_path = log_path.replace("batch_size", str(batch_size))
        if os.path.exists(new_log_path):
            with open(new_log_path, "r") as fp_file:
                lines = fp_file.readlines()
                if len(lines) >= 2 and "time total cost(ms):" in lines[-1]: # 说明已经跑过了，直接过滤掉。
                    continue
                else:
                    need_run_batch_sizes.append(batch_size)
        else:
            need_run_batch_sizes.append(batch_size)
    
    if len(need_run_batch_sizes) == 0:
        return

    import torch
    import torch.distributed as dist
    rank_id = model_kvargs["tp_rank"]
    world_size = model_kvargs["world_size"]

    dist.init_process_group('nccl', init_method='tcp://127.0.0.1:28765', rank=rank_id, world_size=world_size)
    torch.cuda.set_device(rank_id)

    import torch.distributed as dist
    dist.barrier()
    torch.cuda.empty_cache()

    model_part = model_class(model_kvargs)

    for batch_size in need_run_batch_sizes:
        model_part.mem_manager.free_all()
        model_part.req_manager.free_all()
        model_part.mem_manager.resize_mem(batch_size * (input_len + output_len))
        # warm up
        test_data = np.vstack([np.arange(5, input_len + 5) for _ in range(batch_size)])
        test_data = test_data.reshape(-1)
        test_data = torch.from_numpy(test_data).cuda()

        b_req_idx = model_part.req_manager.alloc(batch_size).int()
        b_start_loc = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
        b_seq_len = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
        for i in range(batch_size):
            b_start_loc[i] = i * input_len
            b_seq_len[i] = input_len

        total_token_num = input_len * batch_size
        logics = model_part.forward(batch_size, 
                                    total_token_num, 
                                    input_len, 
                                    test_data,
                                    b_req_idx,
                                    b_start_loc,
                                    b_seq_len,
                                    is_prefill=True)
        prob_out = torch.softmax(logics, dim=-1)
        predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
        predict_ids = predict_ids.detach().cpu().numpy()

        for i in range(output_len):
            b_start_loc = b_start_loc + torch.arange(0, batch_size, dtype=torch.int32, device="cuda")
            total_token_num += batch_size
            b_seq_len += 1
            logics = model_part.forward(batch_size, total_token_num, input_len + i + 1, torch.from_numpy(
                predict_ids).cuda().reshape(-1), b_req_idx, b_start_loc, b_seq_len, is_prefill=False)
            prob_out = torch.softmax(logics, dim=-1)
            predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
            predict_ids = predict_ids.detach().cpu().numpy()

        model_part.mem_manager.free_all()
        model_part.req_manager.free_all()
        
        if rank_id == 0:
            print("can use mem size:", model_part.mem_manager.can_use_mem_size)
            print("can use req size:", model_part.req_manager.can_use_req_size)
            
        b_req_idx = None
        b_start_loc = None
        b_seq_len = None
        
        dist.barrier()
        if rank_id == 0:
            new_log_path = log_path.replace("batch_size", str(batch_size))
            fp_file = open(new_log_path, "w+")
        
        import time
        torch.cuda.synchronize()
        start_time = time.time()

        prefill_start_time = time.time()

        b_req_idx = model_part.req_manager.alloc(batch_size).int()
        b_start_loc = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
        b_seq_len = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
        for i in range(batch_size):
            b_start_loc[i] = i * input_len
            b_seq_len[i] = input_len

        total_token_num = batch_size * input_len
        logics = model_part.forward(batch_size, total_token_num, input_len, test_data,
                                                    b_req_idx, b_start_loc, b_seq_len, is_prefill=True)
        prob_out = torch.softmax(logics, dim=-1)
        predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
        predict_ids = predict_ids.detach().cpu().numpy()

        torch.cuda.synchronize()
        if rank_id == 0:
            print("prefill time cost:", (time.time() - prefill_start_time) * 1000, file=fp_file)

        for i in range(output_len):
            torch.cuda.synchronize()
            step_start = time.time()
            b_start_loc = b_start_loc + torch.arange(0, batch_size, dtype=torch.int32, device="cuda")
            total_token_num += batch_size
            b_seq_len += 1

            logics = model_part.forward(batch_size, total_token_num, input_len + i + 1, torch.from_numpy(
                predict_ids).cuda().reshape(-1), b_req_idx, b_start_loc, b_seq_len, is_prefill=False)
            prob_out = torch.softmax(logics, dim=-1)
            predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
            predict_ids = predict_ids.detach().cpu().numpy()
            torch.cuda.synchronize()
            if i % 100 == 0 or i == output_len - 1:
                if rank_id == 0:
                    print(i, "step cost time:", (time.time() - step_start) * 1000, file=fp_file)

        torch.cuda.synchronize()
        end_time = time.time()

        if rank_id == 0:
            print("time total cost(ms):", (end_time - start_time) * 1000, file=fp_file)
            import sys
            if fp_file is not sys.stdout:
                fp_file.flush()
                fp_file.close()
                while not fp_file.closed:
                    fp_file.close()
        
        b_req_idx = None
        b_start_loc = None
        b_seq_len = None
        test_data = None

        ans_queue.put(True)

    return


