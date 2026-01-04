import torch
import triton
import triton.language as tl


@triton.jit
def embedding_kernel(
    weight,
    input_ids,
    out,
    vob_start_id,
    vob_end_id,
    stride_weight_seq,
    stride_out_seq,
    n_ctx,
    hiden_size: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,  # 32
    BLOCK_NN: tl.constexpr,
):
    start_n = tl.program_id(0) * BLOCK_N

    offs_nn = start_n + tl.arange(0, BLOCK_NN)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    for start_nn in range(0, BLOCK_N, BLOCK_NN):
        start_nn = tl.multiple_of(start_nn, BLOCK_NN)
        offs_seq = start_nn + offs_nn
        n_ctx_mask = offs_seq < n_ctx
        token_ids = tl.load(input_ids + offs_seq, mask=n_ctx_mask, other=vob_end_id)
        id_mask = (token_ids >= vob_start_id) & (token_ids < vob_end_id)
        token_ids = token_ids - vob_start_id
        dim_mask = offs_d < hiden_size
        load_mask = id_mask[:, None] & dim_mask[None, :]
        store_mask = n_ctx_mask[:, None] & dim_mask[None, :]
        vecs = tl.load(weight + token_ids[:, None] * stride_weight_seq + offs_d[None, :], mask=load_mask, other=0.0)
        tl.store(out + offs_seq[:, None] * stride_out_seq + offs_d[None, :], vecs, mask=store_mask)


@torch.no_grad()
def embedding(input_ids, weight: torch.Tensor, vob_start_id, vob_end_id, out: torch.Tensor):

    BLOCK_N = 64
    BLOCK_NN = 1
    BLOCK_DMODEL = triton.next_power_of_2(weight.shape[1])
    n_ctx = input_ids.shape[0]

    grid = (triton.cdiv(n_ctx, BLOCK_N), 1, 1)

    embedding_kernel[grid](
        weight,
        input_ids,
        out,
        vob_start_id,
        vob_end_id,
        weight.stride(0),
        out.stride(0),
        n_ctx=n_ctx,
        hiden_size=weight.shape[1],
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_N=BLOCK_N,
        BLOCK_NN=BLOCK_NN,
        num_warps=1,
        num_stages=1,
    )


@torch.no_grad()
def embedding_new(input_ids, weight, vob_start_id, vob_end_id):
    # out = self.alloc_tensor((N_CTX, DIM), dtype=torch.float32)
    out = torch.empty((N_CTX, DIM), device="cuda", requires_grad=False)

    embedding(input_ids, weight, vob_start_id, vob_end_id, out)
    return out


@torch.no_grad()
def embedding_old(input_ids, wte_weight, vob_start_id, vob_end_id):
    input_mask = torch.empty(input_ids.shape, dtype=torch.bool, device="cuda")
    torch.logical_or(vob_start_id > input_ids, input_ids >= vob_end_id, out=input_mask)
    tmp_input_ids = torch.zeros_like(input_ids)
    torch.sub(input_ids, vob_start_id, out=tmp_input_ids)
    tmp_input_ids[input_mask] = 0
    # to do 将 embedding 操作替换为可以 out 参数的算子，可以自己申请tensor进行传入。
    input_embdings = torch.embedding(wte_weight, tmp_input_ids, padding_idx=-1)
    input_embdings[input_mask] = 0.0
    return input_embdings


if __name__ == "__main__":

    import time
    import random

    DIM = 4096
    VOB_SIZE = 151645
    N_CTX = 10 * 1024
    TEST_COUNT = 1000
    max_diff = 0

    t1 = 0
    t2 = 0

    wte_weight = torch.randn(VOB_SIZE, DIM, device="cuda")

    for TP in [1, 2, 4, 8]:
        for i in range(TEST_COUNT):
            for rank_id in range(TP):

                input_ids = torch.randint(0, VOB_SIZE, (N_CTX,), device="cuda")

                vob_start_id = VOB_SIZE // TP * rank_id
                vob_end_id = VOB_SIZE // TP * (rank_id + 1)

                torch.cuda.synchronize()
                sta_time = time.time()
                new_out = embedding_new(input_ids, wte_weight, vob_start_id, vob_end_id)
                torch.cuda.synchronize()
                t1 += time.time() - sta_time

                torch.cuda.synchronize()
                sta_time = time.time()
                old_out = embedding_old(input_ids, wte_weight, vob_start_id, vob_end_id)
                torch.cuda.synchronize()
                t2 += time.time() - sta_time

                if i == 0:
                    max_diff = max(max_diff, torch.max(torch.abs(new_out - old_out)).item())
                    t1 = 0
                    t2 = 0

        MFLOPS = int(DIM * N_CTX * TEST_COUNT / t1 / 1000 / 1000)
        print(f"TP={TP}, Diff={max_diff}, old_t:{t2:.5f}, new_t:{t1:.5f}, MFLOPS={MFLOPS}, SP={t2/t1:.5f}")
