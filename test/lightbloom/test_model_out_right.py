import torch
import numpy as np
from multiprocessing import Queue
import multiprocessing
import unittest
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


test_model_dir = "/path/bloomz-560m/"


def func(rank_id, world_size, ans_queue):
    import torch
    import torch.distributed as dist
    dist.init_process_group('nccl', init_method='tcp://127.0.0.1:28765', rank=rank_id, world_size=world_size)
    torch.cuda.set_device(rank_id)
    from lightllm.models.bloom.layer_infer.model import BloomTpPartModel
    model_part = BloomTpPartModel(dist.get_rank(), dist.get_world_size(), weight_dir=test_model_dir, load_way="HF")

    output_ids = [
        [12, 210, 1777, 11, 705, 12, 210, 4951],
        [27534, 2321, 15, 408, 458, 2773, 93719, 267],
        [384, 2813, 384, 2813, 384, 2813, 384, 2813],
        [86, 375, 87, 12, 2, 761, 10476, 867],
        [232, 607, 189, 790, 15012, 761, 15, 5026],
        [30, 6, 1135, 62163, 37234, 6, 1135, 62163],
        [182, 189, 182, 189, 182, 189, 182, 189],
        [11945, 12, 5026, 369, 527, 5026, 369, 527]
    ]

    input_ids = [
        [818, 262, 938, 3155, 286, 1528, 11, 257],
        [198, 464, 968, 8221, 2732, 286, 15198, 318],
        [464, 968, 1971, 12056, 423, 257, 649, 1182],
        [464, 968, 1971, 3782, 468, 3199, 663, 5079],
        [818, 257, 1445, 326, 481, 1884, 787, 340],
        [464, 968, 1971, 12056, 6, 5859, 41683, 423],
        [198, 198, 464, 5398, 4332, 628, 628, 198],
        [464, 717, 640, 314, 2497, 262, 3807, 11]]

    input_ids = np.array(input_ids)
    max_len_in_batch = input_ids.shape[-1]
    test_batch_size = 8

    input_ids = torch.from_numpy(input_ids).cuda().reshape(-1)

    # warm_up
    ans_token_ids = []
    # warm_up
    max_new_tokens = 20
    b_loc = torch.zeros(test_batch_size, max_len_in_batch + max_new_tokens, dtype=torch.int32, device="cuda")
    b_start_loc = torch.zeros(test_batch_size, dtype=torch.int32, device="cuda")
    b_seq_len = torch.zeros(test_batch_size, dtype=torch.int32, device="cuda")
    b_seq_len[:] = max_len_in_batch
    # b_seq_len[0], b_seq_len[1] = max_len_in_batch, max_len_in_batch
    for i in range(test_batch_size):
        if i != 0:
            b_start_loc[i] = b_start_loc[i - 1] + b_seq_len[i - 1]
        b_loc[i, max_len_in_batch - b_seq_len[i]: max_len_in_batch] = b_start_loc[i] + \
            torch.arange(0, b_seq_len[i], dtype=torch.int32, device="cuda")
    # print(b_loc, b_start_loc, b_seq_len, input_ids)
    total_token_num = input_ids.shape[0]
    logics, past_key_values = model_part.forward(test_batch_size, total_token_num, max_len_in_batch, input_ids,
                                                 b_loc, b_start_loc, b_seq_len, is_prefill=True, past_key_values=None)
    prob_out = torch.softmax(logics, dim=-1)
    predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
    predict_ids = predict_ids.detach().cpu().numpy()
    # if dist.get_rank() == 0:
    #     print(predict_ids)

    ans_token_ids.append(predict_ids)

    for i in range(12):

        b_loc[:, max_len_in_batch] = total_token_num + torch.arange(0, test_batch_size, dtype=torch.int32, device="cuda")
        b_start_loc = b_start_loc + torch.arange(0, test_batch_size, dtype=torch.int32, device="cuda")
        total_token_num += test_batch_size
        max_len_in_batch += 1
        b_seq_len += 1

        logics, past_key_values = model_part.forward(test_batch_size, total_token_num, max_len_in_batch, torch.from_numpy(
            predict_ids).cuda().reshape(-1), b_loc, b_start_loc, b_seq_len, is_prefill=False, past_key_values=past_key_values)
        prob_out = torch.softmax(logics, dim=-1)
        predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
        predict_ids = predict_ids.detach().cpu().numpy()
        ans_token_ids.append(predict_ids)
        # if i < 8:
        # if (dist.get_rank() == 0):
        #     print(predict_ids)

    ans_token_ids = np.concatenate(ans_token_ids, axis=1)

    ans_queue.put(np.array_equal(ans_token_ids[:, 0:8], np.array(output_ids)))
    return


class TestBloomInfer(unittest.TestCase):

    def test_bloom_infer(self):
        ans_queue = Queue()
        world_size = 8
        workers = []
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
