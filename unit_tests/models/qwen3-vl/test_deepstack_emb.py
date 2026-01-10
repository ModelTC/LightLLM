import torch
import triton
import triton.language as tl
from lightllm.models.qwen3_vl.infer_struct import Qwen3VLInferStateInfo
from lightllm.models.qwen3_vl.triton_kernel.deepstack_multimodal_emb import add_deepstack_embs


def test_deepstack_same_image_twice():
    device = "cuda"

    # 1. 构造 input_ids，包含两段相同的 image token 范围 [100, 101, 102]
    input_ids = torch.tensor(
        [1, 100, 101, 102, 2, 100, 101, 102, 3],
        device=device,
        dtype=torch.long,
    )
    seq_len = input_ids.shape[0]

    hidden_size = 4
    token_len = 3  # 每张图 3 个 token

    # 2. 构造初始 embedding，全 0，方便看增量
    input_embeddings = torch.zeros(seq_len, hidden_size, device=device, dtype=torch.float32)

    # 3. 构造 deepstack_embs（这一层的 deepstack）
    #    只有一张图片，所以 deepstack_embs 形状是 [token_len, hidden_size]
    #    每一行是 [1,1,1,1], [2,2,2,2], [3,3,3,3]
    deepstack_embs = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],  # 对应 token_id = 100
            [2.0, 2.0, 2.0, 2.0],  # 对应 token_id = 101
            [3.0, 3.0, 3.0, 3.0],  # 对应 token_id = 102
        ],
        device=device,
        dtype=torch.float32,
    )

    # 4. image 相关索引信息（与 multimodal_emb 一致的语义）
    img_start_token_ids = torch.tensor([100], device=device, dtype=torch.long)  # 只有一个 image handle，从 100 开始
    img_token_lens = torch.tensor([token_len], device=device, dtype=torch.long)
    img_start_locs = torch.tensor([0], device=device, dtype=torch.long)  # deepstack_embs 从第 0 行开始是这张图的

    # 5. 保存一份原始 embedding，方便求差
    before = input_embeddings.clone()

    # 6. 调用 Triton 算子
    add_deepstack_embs(
        out=input_embeddings,
        input_ids=input_ids,
        deepstack_embs=deepstack_embs,
        img_token_lens=img_token_lens,
        img_start_token_ids=img_start_token_ids,
        img_start_locs_in_cache=img_start_locs,
    )

    # 7. 看看相同图片两段上的增量
    delta = input_embeddings - before

    print("input_ids:", input_ids)
    print("delta:\n", delta)

    # 第一次 image：位置 1,2,3
    print("first image span delta:\n", delta[1:4])
    # 第二次 image：位置 5,6,7
    print("second image span delta:\n", delta[5:8])

    # 8. 断言它们和预期一致
    expected = deepstack_embs  # [3, 4]

    assert torch.allclose(delta[1:4], expected), "first image span does not match expected deepstack"
    assert torch.allclose(delta[5:8], expected), "second image span does not match expected deepstack"

    # 其他位置应该仍然是 0
    assert torch.all(delta[0] == 0)
    assert torch.all(delta[4] == 0)
    assert torch.all(delta[8] == 0)

    print("OK: same image appears twice, both spans get deepstack added correctly.")


if __name__ == "__main__":
    test_deepstack_same_image_twice()
