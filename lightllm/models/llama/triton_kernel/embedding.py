import torch
import triton
import triton.language as tl


@triton.autotune([triton.Config({"BLOCK_N": bn}) for bn in [32, 64, 128, 256, 512, 1024]], key=["BLOCK_DMODEL"])
@triton.jit
def embedding_kernel(
    weight,
    input_ids,
    out,
    vob_start_id,
    vob_end_id,
    stride_weight_size,
    stride_weight_dim,
    stride_out_size,
    stride_out_dim,
    n_ctx,
    hiden_size: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_n = tl.program_id(0) * BLOCK_N

    out += start_n * stride_out_size
    input_ids += start_n

    offs_d = tl.arange(0, BLOCK_DMODEL)
    block_end = min(BLOCK_N, n_ctx - start_n)

    for i in range(0, block_end):
        token_id = tl.load(input_ids + i)
        if token_id >= vob_start_id and token_id < vob_end_id:
            token_id -= vob_start_id

            vec = tl.load(weight + token_id * stride_weight_size + offs_d * stride_weight_dim, mask=offs_d < hiden_size)
            tl.store(out + i * stride_out_size + offs_d * stride_out_dim, vec, mask=offs_d < hiden_size)


@torch.no_grad()
def embedding(input_ids, weight: torch.Tensor, vob_start_id, vob_end_id, out: torch.Tensor):
    out[...] = 0.0

    BLOCK_DMODEL = triton.next_power_of_2(weight.shape[1])
    n_ctx = input_ids.shape[0]

    grid = lambda mate: (triton.cdiv(n_ctx, mate["BLOCK_N"]), 1, 1)

    embedding_kernel[grid](
        weight,
        input_ids,
        out,
        vob_start_id,
        vob_end_id,
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        n_ctx=n_ctx,
        hiden_size=weight.shape[1],
        BLOCK_DMODEL=BLOCK_DMODEL,
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

    DIM = 3584
    VOB_SIZE = 151645
    N_CTX = 10240
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
