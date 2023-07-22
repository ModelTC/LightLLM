
import torch
import numpy as np
from multiprocessing import Queue
import multiprocessing
import unittest
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


test_model_dir = "/path/llama2-70b-chat"


def func(rank_id, world_size, ans_queue):
    import torch
    import torch.distributed as dist
    dist.init_process_group('nccl', init_method='tcp://127.0.0.1:28765', rank=rank_id, world_size=world_size)
    torch.cuda.set_device(rank_id)

    import torch.distributed as dist
    dist.barrier()
    torch.cuda.empty_cache()
    from lightllm.models.llama2.layer_infer.model import Llama2TpPartModel
    model_part = Llama2TpPartModel(dist.get_rank(), dist.get_world_size(), max_total_token_num= 2048 * 50, weight_dir=test_model_dir, load_way="HF")
    input_length_ = 1024
    output_length_ = 1024

    test_batch_size = 50
    test_data = np.vstack([np.arange(5, input_length_ + 5) for _ in range(test_batch_size)])
    test_data = test_data.reshape(-1)
    test_data = torch.from_numpy(test_data).cuda()

    b_loc = torch.zeros(test_batch_size, 6000, dtype=torch.long, device="cuda")
    b_start_loc = torch.zeros(test_batch_size, dtype=torch.int32, device="cuda")
    b_seq_len = torch.zeros(test_batch_size, dtype=torch.int32, device="cuda")
    for i in range(test_batch_size):
        b_loc[i, 0:input_length_] = i * input_length_ + torch.arange(0, input_length_, dtype=torch.int32, device="cuda")
        b_start_loc[i] = i * input_length_
        b_seq_len[i] = input_length_

    total_token_num = test_batch_size * input_length_
    logics = model_part.forward(test_batch_size, total_token_num, input_length_, test_data,
                                                 b_loc, b_start_loc, b_seq_len, is_prefill=True)
    prob_out = torch.softmax(logics, dim=-1)
    predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
    predict_ids = predict_ids.detach().cpu().numpy()

    for i in range(output_length_):
        b_loc[:, input_length_ + i] = total_token_num + torch.arange(0, test_batch_size, dtype=torch.int32, device="cuda")
        b_start_loc = b_start_loc + torch.arange(0, test_batch_size, dtype=torch.int32, device="cuda")
        total_token_num += test_batch_size
        b_seq_len += 1

        logics = model_part.forward(test_batch_size, total_token_num, input_length_ + i + 1, torch.from_numpy(
            predict_ids).cuda().reshape(-1), b_loc, b_start_loc, b_seq_len, is_prefill=False)
        prob_out = torch.softmax(logics, dim=-1)
        predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
        predict_ids = predict_ids.detach().cpu().numpy()
    
    max_len_in_batch = input_length_ + output_length_
    for i in range(test_batch_size):
        model_part.mem_manager.free(b_loc[i, max_len_in_batch - b_seq_len[i]:max_len_in_batch])
        
    print("can use mem size:", model_part.mem_manager.can_use_mem_size)
        
    b_loc = None
    b_start_loc = None
    b_seq_len = None
    
    dist.barrier()
    import time
    torch.cuda.synchronize()
    start_time = time.time()

    prefill_start_time = time.time()

    b_loc = torch.zeros(test_batch_size, 6000, dtype=torch.long, device="cuda")
    b_start_loc = torch.zeros(test_batch_size, dtype=torch.int32, device="cuda")
    b_seq_len = torch.zeros(test_batch_size, dtype=torch.int32, device="cuda")
    for i in range(test_batch_size):
        b_start_loc[i] = i * input_length_
        b_seq_len[i] = input_length_

    total_token_num = test_batch_size * input_length_
    logics = model_part.forward(test_batch_size, total_token_num, input_length_, test_data,
                                                 b_loc, b_start_loc, b_seq_len, is_prefill=True)
    prob_out = torch.softmax(logics, dim=-1)
    predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
    predict_ids = predict_ids.detach().cpu().numpy()

    torch.cuda.synchronize()
    print("prefill time cost:", (time.time() - prefill_start_time) * 1000)

    for i in range(output_length_):
        torch.cuda.synchronize()
        step_start = time.time()
        b_start_loc = b_start_loc + torch.arange(0, test_batch_size, dtype=torch.int32, device="cuda")
        total_token_num += test_batch_size
        b_seq_len += 1

        logics = model_part.forward(test_batch_size, total_token_num, input_length_ + i + 1, torch.from_numpy(
            predict_ids).cuda().reshape(-1), b_loc, b_start_loc, b_seq_len, is_prefill=False)
        prob_out = torch.softmax(logics, dim=-1)
        predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
        predict_ids = predict_ids.detach().cpu().numpy()
        torch.cuda.synchronize()
        if i % 100 == 0 or i == output_length_ - 1:
            print(i, "cost time:", (time.time() - step_start) * 1000)

    torch.cuda.synchronize()
    end_time = time.time()

    print("time cost(ms):", (end_time - start_time) * 1000)
    ans_queue.put(True)

    return


class TestLlamaInfer(unittest.TestCase):

    def test_llama_infer(self):
        ans_queue = Queue()
        workers = []
        world_size = 8
        for rank_id in range(world_size):
            proc = multiprocessing.Process(target=func, args=(rank_id, world_size, ans_queue))
            proc.start()
            workers.append(proc)

        for proc in workers:
            proc.join()

        assert not ans_queue.empty()
        while not ans_queue.empty():
            assert ans_queue.get()
        return


if __name__ == '__main__':
    unittest.main()
