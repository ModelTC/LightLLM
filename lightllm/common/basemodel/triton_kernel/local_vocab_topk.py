"""从 [本地词表, token 行数] 布局的 logits 中筛选候选，优化大批量下的词表扫描。"""

import torch
import triton
import triton.language as tl


@triton.jit
def _transpose_vocab_logits_kernel(
    logits,
    transposed_logits,
    token_num,
    local_vocab_size: tl.constexpr,
    BLOCK_VOCAB: tl.constexpr,
    BLOCK_TOKEN: tl.constexpr,
):
    # B 使用运行时参数，避免请求批量变化时为每个 token 行数单独编译 kernel。
    # 每个 program 搬运一个二维 tile：输入按 token 方向连续，输出按词表方向连续。
    # tile 内的布局重排兼顾两侧的合并访存，避免逐列拷贝时的大步长读取。
    vocab_indexes = tl.program_id(0) * BLOCK_VOCAB + tl.arange(0, BLOCK_VOCAB)
    token_indexes = tl.program_id(1) * BLOCK_TOKEN + tl.arange(0, BLOCK_TOKEN)
    # 词表大小和 token 行数不必是 tile 的整数倍，边界只读写有效位置。
    valid = (vocab_indexes[:, None] < local_vocab_size) & (token_indexes[None, :] < token_num)

    # 将 logits[vocab, token] 原值写到 transposed_logits[token, vocab]，不改变数值精度。
    input_offsets = vocab_indexes[:, None] * token_num + token_indexes[None, :]
    output_offsets = token_indexes[None, :] * local_vocab_size + vocab_indexes[:, None]
    values = tl.load(logits + input_offsets, mask=valid, other=0)
    tl.store(transposed_logits + output_offsets, values, mask=valid)


@torch.no_grad()
def local_vocab_topk(
    local_logits: torch.Tensor,
    top_k: int,
    alloc_func=torch.empty,
) -> tuple[torch.Tensor, torch.Tensor]:
    """逐个 token 行选取本地词表中的 top-k，返回 [B, K] 分数和 int64 本地 token ID。

    输入与筛选语义：
        local_logits 的形状为 [V, B]，V 是当前 rank 的词表大小，B 是本次 forward
        的 token 行数，不一定等于请求并发数。例如 MTP3、并发 64 时，verify 和
        首次 draft 可以有 256 行。每一列独立选 K=top_k 个候选；本函数不做跨 rank
        通信。返回分数保持输入 dtype，ID 是原词表行号，同分项遵循 torch.topk 的
        语义，不保证稳定的候选顺序或同分边界 ID。

    为什么先转置：
        连续 [V, B] 张量的 stride 为 (B, 1)，原 dim=0 top-k 沿词表读取时跨 B 个
        元素。B=256、BF16 时，相邻词表元素相距 512 字节，当前 PyTorch kernel
        中相邻线程的读取难以合并。一个 warp 的 64 字节有效数据会分散到 32 个
        32-byte sector；这是访存请求粒度的放大，不能忽略缓存而等同于 HBM 流量。
        转成连续 [B, V] 后，dim=1 的词表扫描步长变为 1，多轮筛选都能受益。

    为什么需要分块转置 kernel，而不只写 .T.contiguous()：
        .T 只改变视图。实测所用 PyTorch 2.11 的 top-k 内部会调用 contiguous()，
        因此直接对 .T 做 dim=1 top-k 也会隐式复制，并没有省掉连续化成本。
        H200、真实 BF16 [62080, 256]、K64 的对照中，原生连续化约 112 微秒，
        分块转置约 18 微秒。计入转换成本后，原 dim=0 调用约 369 微秒，原生转置
        加 top-k 约 271 微秒，本组合约 176 微秒。这里统计的是 CUDA Graph 下
        整个算子组合的时间，不含 LM-head 和通信，也不是整段服务的加速比。
        top-k 还包含 radix 统计、原子计数、前缀和及同步，不能用一次 HBM 读取的
        理论时间代替它的成本。本方案只优化布局，筛选仍交给 torch.topk。

    缓冲区与返回布局：
        alloc_func 由调用方传入，使额外的 [B, V] 缓冲使用现有 tensor cache，并在
        CUDA Graph 捕获时纳入其内存池。每次调用都执行转置，不能在捕获前只填一次；
        新输入必须在后续 replay 中生效。BF16 [62080, 256] 额外需要约 30.3 MiB。
        返回布局统一为 [B, K]，与后续 candidate pack 和输出层的 token 行布局一致。
        优化路径直接返回连续的 top-k 输出；回退路径将 dim=0 的 [K, B] 结果变为
        [B, K] 转置视图，不搬运数据，也不改变 ID 含义。调用方不能假定输出连续；
        candidate pack 按显式 stride 读取，无须为回退路径额外复制。
    """
    local_vocab_size, token_num = local_logits.shape
    # 从 16 行启用分块转置。H200、V=62080 的 Graph 测试中，B=16 基本持平，
    # B>=32 已有收益；B=1～8 则有额外开销。B=16 在 eager 下仍可能因发射成本变慢，
    # 因此此阈值是取舍，并不保证所有词表大小和执行模式都加速。
    # BF16 是已验证的性能路径；kernel 的地址计算要求输入连续，其他 dtype/布局保留
    # 原调用。空词表或 K=0 也交给 torch.topk 处理，避免启动无意义的转置 grid。
    use_tiled_transpose = (
        token_num >= 16
        and local_vocab_size > 0
        and top_k > 0
        and local_logits.is_cuda
        and local_logits.dtype == torch.bfloat16
        and local_logits.is_contiguous()
    )
    if not use_tiled_transpose:
        values, token_ids = torch.topk(local_logits, k=top_k, dim=0, sorted=False)
        return values.T, token_ids.T

    # TODO: 开发并调优融合转置与 top-k 筛选的 Triton 算子，减少中间缓冲和访存，
    # 在保持候选分数与 token ID 正确性的前提下优化整体性能。
    # 只重排 BF16 数据，不升为 FP32；同时避免在此绕开调用方的 allocator 自行分配。
    transposed_logits = alloc_func((token_num, local_vocab_size), dtype=local_logits.dtype, device=local_logits.device)
    # H200 原型对照中，64×64 tile、8 个 warp 的组合在 B=64/256 时表现稳定。
    # 固定 tile 避免在推理时调优；它并不表示所有 GPU 和形状的最优配置都相同。
    block_vocab = 64
    block_token = 64
    grid = (triton.cdiv(local_vocab_size, block_vocab), triton.cdiv(token_num, block_token))
    _transpose_vocab_logits_kernel[grid](
        local_logits,
        transposed_logits,
        token_num,
        local_vocab_size=local_vocab_size,
        BLOCK_VOCAB=block_vocab,
        BLOCK_TOKEN=block_token,
        num_warps=8,
    )
    # 词表现在位于连续的最后一维；只需要候选集合，不额外要求候选按分数排序。
    return torch.topk(transposed_logits, k=top_k, dim=1, sorted=False)
