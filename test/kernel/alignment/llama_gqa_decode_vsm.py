import unittest
import random
import torch
from tqdm import tqdm
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.common.req_manager import ReqManager
from lightllm.models.llama.triton_kernel.gqa_flash_decoding_vsm import (
    gqa_token_decode_attention_flash_decoding_vsm,
)
from lightllm.models.llama.triton_kernel.gqa_flash_decoding import (
    gqa_token_decode_attention_flash_decoding,
)


class TestVSMGQADecoding(unittest.TestCase):
    def test_vsm_gqa_decoding_align(self):
        random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        bs_list = [1, 8, 16, 32, 64, 128, 256]
        group_size_list = [16, 32, 64]
        seq_len_list = [128, 512, 1024, 2048, 4096, 8192]
        q_head_dim_list = [64, 128]
        q_head_num_list = [8, 16, 32]

        def get_test_configs():
            for bs in bs_list:
                for group_size in group_size_list:
                    for seq_len_m in seq_len_list:
                        for q_head_dim in q_head_dim_list:
                            for q_head_num in q_head_num_list:
                                if q_head_num < group_size:
                                    continue
                                yield bs, group_size, seq_len_m, q_head_dim, q_head_num

        for bs, group_size, seq_len_m, q_head_dim, q_head_num in tqdm(list(get_test_configs())):
            kv_head_num = q_head_num // group_size
            q_head_dim = q_head_dim
            kv_head_dim = q_head_dim
            seq_len = (torch.zeros(bs, dtype=torch.int32) + seq_len_m).to(torch.int32)
            total_token_in_the_batch = seq_len.sum().item()
            rounded_total_token_in_the_batch = (total_token_in_the_batch + 128 - 1) // 128 * 128

            q_shape = [bs, q_head_num, q_head_dim]
            kv_shape = [
                rounded_total_token_in_the_batch,
                kv_head_num,
                kv_head_dim,
            ]
            qkv_dtype = torch.float16

            q, k, v = (
                torch.randn(q_shape, dtype=qkv_dtype, device="cuda"),
                torch.randn(kv_shape, dtype=qkv_dtype, device="cuda"),
                torch.randn(kv_shape, dtype=qkv_dtype, device="cuda"),
            )
            q, k, v = q / 10, k / 10, v / 10

            req_to_token_index = torch.zeros((bs, seq_len_m)) - 1
            token_index = torch.arange(rounded_total_token_in_the_batch)

            total_count = 0
            for i in range(bs):
                req_to_token_index[i, : seq_len[i]] = token_index[total_count : total_count + seq_len[i]]
                total_count += seq_len[i]

            req_to_token_index = req_to_token_index.long().cuda()

            b_req_idx = torch.arange(bs, device="cuda")
            infer_state = InferStateInfo()
            infer_state.req_manager = ReqManager(bs, 2048, None)
            infer_state.req_manager.req_to_token_indexs = req_to_token_index
            infer_state.b_req_idx = b_req_idx.cuda()
            infer_state.b_seq_len = seq_len.cuda()
            infer_state.max_len_in_batch = seq_len_m
            infer_state.batch_size = bs
            infer_state.q_head_num = q_head_num
            infer_state.q_head_dim = q_head_dim
            infer_state.kv_head_num = kv_head_num
            infer_state.softmax_scale = 1 / (q_head_dim ** 0.5)
            infer_state.total_token_num = torch.tensor([total_token_in_the_batch], dtype=torch.int32).cuda()
            new_out = gqa_token_decode_attention_flash_decoding_vsm(q, k, v, infer_state)
            old_out = gqa_token_decode_attention_flash_decoding(
                q,
                infer_state,
                infer_state.q_head_num,
                infer_state.q_head_dim,
                k,
                v,
            )
            cos_sim = torch.nn.functional.cosine_similarity(new_out, old_out, dim=-1).mean().cpu().item()
            self.assertGreaterEqual(
                cos_sim,
                0.9,
                f"bs={bs},group_size={group_size},seq_len={seq_len_m},q_head_dim={q_head_dim},q_head_num={q_head_num}",
            )


if __name__ == "__main__":
    unittest.main()
