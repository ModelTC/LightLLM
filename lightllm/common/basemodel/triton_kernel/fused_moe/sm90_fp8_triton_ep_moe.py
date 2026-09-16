"""Single-node SM90 FP8 expert parallelism over CUDA symmetric memory."""

from typing import Callable, Optional, Tuple

import deep_gemm
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm
import triton
import triton.language as tl
from frozendict import frozendict

from lightllm.common.kernel_config import KernelConfigs


DEFAULT_CONFIG = {
    "publish_rows_large": 16,
    "publish_rows_small": 4,
    "publish_num_warps": 4,
    "histogram_block": 1024,
    "histogram_num_warps": 4,
    "pull_num_warps": 4,
    "activation_programs": 512,
    "activation_rows": 16,
    "activation_num_warps": 4,
    "return_num_warps": 4,
    "combine_rows_large": 4,
    "combine_rows_small": 1,
    "combine_block": 512,
    "combine_num_warps": 4,
}


class SM90FP8TritonEPMoEKernelConfig(KernelConfigs):
    kernel_name = "sm90_fp8_triton_ep_moe"

    @classmethod
    def _params(
        cls,
        hidden_size: int,
        intermediate_size: int,
        local_experts: int,
        topk: int,
        world_size: int,
        num_max_tokens_per_rank: int,
        alignment: int,
    ):
        return frozendict(
            {
                "hidden_size": hidden_size,
                "intermediate_size": intermediate_size,
                "local_experts": local_experts,
                "topk": topk,
                "world_size": world_size,
                "num_max_tokens_per_rank": num_max_tokens_per_rank,
                "alignment": alignment,
            }
        )

    @classmethod
    def try_to_get_best_config(cls, **kwargs) -> dict:
        config = cls.get_the_config(cls._params(**kwargs))
        return dict(DEFAULT_CONFIG if config is None else config)

    @classmethod
    def get_config_if_available(cls, **kwargs) -> Optional[dict]:
        config = cls.get_the_config(cls._params(**kwargs))
        return None if config is None else dict(config)

    @classmethod
    def save_config(cls, config: dict, **kwargs) -> None:
        cls.store_config(cls._params(**kwargs), config)


@triton.jit
def _publish(
    X,
    IDS,
    WEIGHTS,
    Q,
    SF,
    SID,
    SW,
    ROWS,
    COUNTS,
    M,
    H: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    BE: tl.constexpr,
    BK: tl.constexpr,
    BM: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    block = tl.program_id(1)
    cols = block * 128 + tl.arange(0, 128)
    values = tl.load(X + rows[:, None] * H + cols[None, :], rows[:, None] < M, 0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(values), 1), 1e-10) / 448.0
    quant = tl.clamp(tl.div_rn(values, scale[:, None]), -448.0, 448.0).to(Q.dtype.element_ty)
    tl.store(Q + rows[:, None] * H + cols[None, :], quant, rows[:, None] < M)
    tl.store(SF + rows * (H // 128) + block, scale, rows < M)
    if block == 0:
        slots = tl.arange(0, BK)
        offsets = rows[:, None] * K + slots[None, :]
        mask = (rows[:, None] < M) & (slots[None, :] < K)
        tl.store(SID + offsets, tl.load(IDS + offsets, mask, -1), mask)
        tl.store(SW + offsets, tl.load(WEIGHTS + offsets, mask, 0), mask)
        if tl.program_id(0) == 0:
            tl.store(ROWS, M)
            experts = tl.arange(0, BE)
            tl.store(COUNTS + experts, 0, experts < E)


@triton.jit
def _histogram(
    ID_PTRS,
    ROW_PTRS,
    COUNTS,
    CAP: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    RANK: tl.constexpr,
    BE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    peer = tl.program_id(1)
    ids = tl.load(ID_PTRS + peer).to(tl.pointer_type(tl.int64))
    rows = tl.load(tl.load(ROW_PTRS + peer).to(tl.pointer_type(tl.int32)))
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    expert = tl.load(ids + offsets, offsets < rows * K, -1).to(tl.int32) - RANK * E
    valid = (offsets < rows * K) & (expert >= 0) & (expert < E)
    counts = tl.histogram(expert, BE, mask=valid)
    experts = tl.arange(0, BE)
    tl.atomic_add(COUNTS + experts, counts, experts < E, sem="relaxed")


@triton.jit
def _prefix(COUNTS, ENDS, CURSOR, E: tl.constexpr, BE: tl.constexpr, ALIGN: tl.constexpr):
    experts = tl.arange(0, BE)
    counts = tl.load(COUNTS + experts, experts < E, 0)
    padded = tl.cdiv(counts, ALIGN) * ALIGN
    ends = tl.cumsum(padded)
    tl.store(ENDS + experts, ends, experts < E)
    tl.store(CURSOR + experts, ends - padded, experts < E)


@triton.jit
def _pull(
    X_PTRS,
    SF_PTRS,
    ID_PTRS,
    ROW_PTRS,
    CURSOR,
    X,
    SF,
    ROW_MAP,
    CAP: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    RANK: tl.constexpr,
    R: tl.constexpr,
    WORLD: tl.constexpr,
    BK: tl.constexpr,
):
    token = tl.program_id(0) // WORLD
    peer = (tl.program_id(0) % WORLD + RANK) % WORLD
    rows = tl.load(tl.load(ROW_PTRS + peer).to(tl.pointer_type(tl.int32)))
    if token < rows:
        ids = tl.load(ID_PTRS + peer).to(tl.pointer_type(tl.int64))
        slots = tl.arange(0, BK)
        experts = tl.load(ids + token * K + slots, slots < K, -1).to(tl.int32) - RANK * E
        local = (slots < K) & (experts >= 0) & (experts < E)
        if tl.sum(local.to(tl.int32), 0) > 0:
            source_x = tl.load(X_PTRS + peer).to(tl.pointer_type(tl.float8e4nv))
            source_sf = tl.load(SF_PTRS + peer).to(tl.pointer_type(tl.float32))
            cols = tl.arange(0, H)
            scale_cols = tl.arange(0, H // 128)
            values = tl.load(source_x + token * H + cols)
            scales = tl.load(source_sf + token * (H // 128) + scale_cols)
            for slot in range(K):
                expert = tl.load(ids + token * K + slot).to(tl.int32) - RANK * E
                if (expert >= 0) & (expert < E):
                    row = tl.atomic_add(CURSOR + expert, 1, sem="relaxed")
                    tl.store(X + row * H + cols, values)
                    tl.store(SF + row + scale_cols * R, scales)
                    tl.store(ROW_MAP + (peer * CAP + token) * K + slot, row)


@triton.jit
def _activation(
    X,
    Q,
    SF,
    ENDS,
    E: tl.constexpr,
    R: tl.constexpr,
    I: tl.constexpr,
    LIMIT: tl.constexpr,
    BM: tl.constexpr,
):
    total = tl.load(ENDS + E - 1)
    blocks_n = I // 128
    for tile in range(tl.program_id(0), tl.cdiv(total, BM) * blocks_n, tl.num_programs(0)):
        row = (tile // blocks_n) * BM + tl.arange(0, BM)
        block = tile % blocks_n
        col = block * 128 + tl.arange(0, 128)
        offset = row[:, None] * (2 * I) + col[None, :]
        gate = tl.load(X + offset, row[:, None] < total, 0).to(tl.float32)
        up = tl.load(X + offset + I, row[:, None] < total, 0)
        gate = tl.minimum(gate, LIMIT)
        up = tl.clamp(up, -LIMIT, LIMIT)
        gate = (gate / (1 + tl.exp(-gate))).to(tl.bfloat16)
        act = (up * gate).to(tl.bfloat16).to(tl.float32)
        scale = tl.maximum(tl.max(tl.abs(act), 1), 1e-10) / 448.0
        quant = tl.clamp(act / scale[:, None], -448.0, 448.0).to(Q.dtype.element_ty)
        tl.store(Q + row[:, None] * I + col[None, :], quant, row[:, None] < total)
        tl.store(SF + row + block * R, scale, row < total)


@triton.jit
def _return(
    X,
    ROW_MAP,
    ID_PTRS,
    WEIGHT_PTRS,
    ROW_PTRS,
    OUT_PTRS,
    CAP: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
):
    token = tl.program_id(0) // WORLD
    peer = (tl.program_id(0) % WORLD + RANK) % WORLD
    rows = tl.load(tl.load(ROW_PTRS + peer).to(tl.pointer_type(tl.int32)))
    if token < rows:
        ids = tl.load(ID_PTRS + peer).to(tl.pointer_type(tl.int64))
        weights = tl.load(WEIGHT_PTRS + peer).to(tl.pointer_type(tl.float32))
        cols = tl.arange(0, H)
        acc = tl.full((H,), 0.0, tl.float32)
        first = -1
        for slot in range(K):
            expert = tl.load(ids + token * K + slot).to(tl.int32) - RANK * E
            if (expert >= 0) & (expert < E):
                row = tl.load(ROW_MAP + (peer * CAP + token) * K + slot)
                weight = tl.load(weights + token * K + slot)
                acc += tl.load(X + row * H + cols).to(tl.float32) * weight
                if first < 0:
                    first = slot
        if first >= 0:
            output = tl.multiple_of(tl.load(OUT_PTRS + peer).to(tl.pointer_type(tl.bfloat16)), 16)
            tl.store(output + (token * K + first) * H + cols, acc.to(tl.bfloat16))


@triton.jit
def _combine(
    RETURNED,
    IDS,
    Y,
    M,
    H: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    BLOCK: tl.constexpr,
    BM: tl.constexpr,
):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.full((BM, BLOCK), 0, tl.float32)
    for slot in tl.static_range(K):
        expert = tl.load(IDS + row * K + slot, row < M, -1)
        valid = (row < M) & (expert >= 0)
        for earlier in tl.static_range(slot):
            previous = tl.load(IDS + row * K + earlier, row < M, -1)
            valid &= (previous < 0) | (previous // E != expert // E)
        values = tl.load(
            RETURNED + (row[:, None] * K + slot) * H + col[None, :],
            valid[:, None],
            0,
        ).to(tl.float32)
        acc += values
    tl.store(Y + row[:, None] * H + col[None, :], acc.to(tl.bfloat16), row[:, None] < M)


class SM90FP8TritonEPMoEBuffer:
    """SM90 FP8 Triton EP MoE 的共享通信与计算缓冲区。

    每个 EP group 只创建一个实例，并由所有 MoE 层复用；各层的专家权重仍由 layer
    weight 持有。每个 rank 将本地 token、路由结果发布到 symmetric memory，本 rank
    拉取所有发往本地专家的 token，完成两次 grouped GEMM，再把加权后的局部结果写回
    token 所在 rank。当前生产选择器只在单节点 SM90 Prefill 路径启用该类。
    """

    def __init__(
        self,
        group,
        num_experts: int,
        num_max_tokens_per_rank: int,
        topk: int,
        hidden_size: int,
        intermediate_size: int,
        alignment: Optional[int] = None,
    ):
        self.group = group
        self.rank = dist.get_rank(group)
        self.world = dist.get_world_size(group)
        assert num_experts % self.world == 0
        self.num_experts = num_experts
        self.local_experts = num_experts // self.world
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.topk = topk
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # 普通尾块使用 128 对齐；完整 num_max_tokens_per_rank 块若有单独调优配置，
        # 可以切到 256 对齐。显式传入 alignment 时则始终使用同一套配置。
        self.alignment = 128 if alignment is None else alignment
        assert hidden_size == 2 * intermediate_size
        assert hidden_size % 128 == 0 and intermediate_size % 128 == 0
        assert self.alignment in (128, 256)
        self.config = SM90FP8TritonEPMoEKernelConfig.try_to_get_best_config(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            local_experts=self.local_experts,
            topk=topk,
            world_size=self.world,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            alignment=self.alignment,
        )
        self.full_alignment = self.alignment
        self.full_config = self.config
        if alignment is None:
            full_config = SM90FP8TritonEPMoEKernelConfig.get_config_if_available(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                local_experts=self.local_experts,
                topk=topk,
                world_size=self.world,
                num_max_tokens_per_rank=num_max_tokens_per_rank,
                alignment=256,
            )
            if full_config is not None:
                self.full_alignment = 256
                self.full_config = full_config

        # 每个本地专家的接收段都需要独立补齐，最坏情况下额外占用
        # local_experts * (alignment - 1) 行。
        max_recv_rows = self.world * num_max_tokens_per_rank * topk
        workspace_alignment = max(self.alignment, self.full_alignment)
        self.workspace_rows = (
            triton.cdiv(max_recv_rows + self.local_experts * (workspace_alignment - 1), workspace_alignment)
            * workspace_alignment
        )
        self.handles = []
        self.pointers = []
        self.sources = []
        # sources/pointers 的下标是后续 Triton kernel 的固定协议：
        #   0: FP8 hidden states，1: 每 128 列一组的量化 scale
        #   2: top-k expert id，3: top-k weight，4: 本 rank 实际 token 数
        #   5: 各专家 owner 写回的 BF16 局部结果
        for shape, dtype in (
            ((num_max_tokens_per_rank, hidden_size), torch.float8_e4m3fn),
            ((num_max_tokens_per_rank, hidden_size // 128), torch.float32),
            ((num_max_tokens_per_rank, topk), torch.int64),
            ((num_max_tokens_per_rank, topk), torch.float32),
            ((1,), torch.int32),
            ((num_max_tokens_per_rank, topk, hidden_size), torch.bfloat16),
        ):
            tensor = symm.empty(*shape, dtype=dtype, device="cuda")
            handle = symm.rendezvous(tensor, group=group)
            assert all(pointer % 16 == 0 for pointer in handle.buffer_ptrs)
            self.sources.append(tensor)
            self.handles.append(handle)
            self.pointers.append(torch.tensor(handle.buffer_ptrs, device="cuda", dtype=torch.int64))

        # counts 记录每个本地专家收到的 route 数；ends 是对齐后的排他结束位置；
        # cursor 从各专家段起点开始原子递增；row_map 保存 (peer, token, top-k slot)
        # 到本地 expert-major workspace 行号的映射，供结果回传使用。
        self.counts = torch.empty(self.local_experts, device="cuda", dtype=torch.int32)
        self.ends = torch.empty_like(self.counts)
        self.cursor = torch.empty_like(self.counts)
        self.row_map = torch.empty(self.world * num_max_tokens_per_rank * topk, device="cuda", dtype=torch.int32)

        # x/x_scale 保存按本地专家分段后的 FP8 输入；workspace 依次承载 W1 和 W2
        # 的 BF16 输出。W1 完成后 x 已不再使用，且 H=2I，因此其前半空间可原地
        # 复用为量化后的激活输入，避免再分配一份 FP8 activation buffer。
        self.x = torch.empty(self.workspace_rows, hidden_size, device="cuda", dtype=torch.float8_e4m3fn)
        self.x_scale = torch.empty(hidden_size // 128, self.workspace_rows, device="cuda", dtype=torch.float32).T
        self.workspace = torch.empty(self.workspace_rows, hidden_size, device="cuda", dtype=torch.bfloat16)
        self.activation = self.x.view(self.workspace_rows * 2, intermediate_size)[: self.workspace_rows]
        self.activation_scale = self.x_scale[:, : intermediate_size // 128]

    def _get_runtime_alignment_and_config(self, rows: int) -> Tuple[int, dict]:
        """完整块使用 full-chunk 调优结果，尾块使用更稳妥的基础配置。"""
        if rows == self.num_max_tokens_per_rank:
            return self.full_alignment, self.full_config
        return self.alignment, self.config

    @property
    def allocated_bytes(self) -> int:
        """返回本 rank 主要通信与工作区 tensor 的字节数，供基准测试统计。"""
        tensors = self.sources + [
            self.counts,
            self.ends,
            self.cursor,
            self.row_map,
            self.x,
            self.x_scale,
            self.workspace,
        ]
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: Tuple[torch.Tensor, torch.Tensor],
        w2: Tuple[torch.Tensor, torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        clamp_limit: float,
        alloc_tensor_func: Callable = torch.empty,
    ) -> torch.Tensor:
        """执行一次 EP MoE，并返回与 hidden_states 同 shape、同 dtype 的结果。"""
        rows = hidden_states.shape[0]
        assert rows <= self.num_max_tokens_per_rank
        assert hidden_states.shape == (rows, self.hidden_size)
        assert hidden_states.dtype == torch.bfloat16 and hidden_states.is_contiguous()
        assert topk_ids.shape == topk_weights.shape == (rows, self.topk)
        assert topk_ids.dtype == torch.int64 and topk_ids.is_contiguous()
        assert topk_weights.dtype == torch.float32 and topk_weights.is_contiguous()
        assert w1[0].shape == (self.local_experts, 2 * self.intermediate_size, self.hidden_size)
        assert w2[0].shape == (self.local_experts, self.hidden_size, self.intermediate_size)

        alignment, config = self._get_runtime_alignment_and_config(rows)
        publish_rows = config["publish_rows_large"] if rows >= 16 else config["publish_rows_small"]

        # 1. 量化本地输入并发布输入、scale 和路由元数据。barrier 之后所有 rank
        # 才能安全读取彼此的 symmetric-memory source buffers。
        _publish[(max(1, triton.cdiv(rows, publish_rows)), self.hidden_size // 128)](
            hidden_states,
            topk_ids,
            topk_weights,
            *self.sources[:5],
            self.counts,
            rows,
            self.hidden_size,
            self.topk,
            self.local_experts,
            triton.next_power_of_2(self.local_experts),
            triton.next_power_of_2(self.topk),
            publish_rows,
            num_warps=config["publish_num_warps"],
        )
        self.handles[0].barrier(channel=0, timeout_ms=10000)

        # 2. 统计所有 peer 发往本地专家的 route，生成对齐后的 expert-major 分段，
        # 再从 peer buffer 拉取输入并记录每条 route 在 workspace 中的行号。
        histogram_block = config["histogram_block"]
        _histogram[(triton.cdiv(self.num_max_tokens_per_rank * self.topk, histogram_block), self.world)](
            self.pointers[2],
            self.pointers[4],
            self.counts,
            self.num_max_tokens_per_rank,
            self.topk,
            self.local_experts,
            self.rank,
            triton.next_power_of_2(self.local_experts),
            histogram_block,
            num_warps=config["histogram_num_warps"],
        )
        _prefix[(1,)](
            self.counts,
            self.ends,
            self.cursor,
            self.local_experts,
            triton.next_power_of_2(self.local_experts),
            alignment,
        )
        _pull[(self.num_max_tokens_per_rank * self.world,)](
            self.pointers[0],
            self.pointers[1],
            self.pointers[2],
            self.pointers[4],
            self.cursor,
            self.x,
            self.x_scale,
            self.row_map,
            self.num_max_tokens_per_rank,
            self.hidden_size,
            self.topk,
            self.local_experts,
            self.rank,
            self.workspace_rows,
            self.world,
            triton.next_power_of_2(self.topk),
            num_warps=config["pull_num_warps"],
        )

        # 3. 在本地 expert-major 布局上执行 W1 -> SwiGLU+FP8 quant -> W2。
        # DeepGEMM 的 contiguous-layout alignment 是进程级状态，调用后必须恢复。
        expected_m = triton.cdiv(self.num_max_tokens_per_rank * self.topk, self.local_experts)
        previous_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
        deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
        try:
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                (self.x, self.x_scale),
                w1,
                self.workspace,
                self.ends,
                use_psum_layout=True,
                expected_m_for_psum_layout=expected_m,
            )
            _activation[(config["activation_programs"],)](
                self.workspace,
                self.activation,
                self.activation_scale,
                self.ends,
                self.local_experts,
                self.workspace_rows,
                self.intermediate_size,
                clamp_limit,
                config["activation_rows"],
                num_warps=config["activation_num_warps"],
            )
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                (self.activation, self.activation_scale),
                w2,
                self.workspace,
                self.ends,
                use_psum_layout=True,
                expected_m_for_psum_layout=expected_m,
            )
        finally:
            deep_gemm.set_mk_alignment_for_contiguous_layout(previous_alignment)

        # 4. 每个专家 owner 按 row_map 找回源 token，在本地先合并并乘 route weight，
        # 然后向源 rank 写回一份局部结果。第二次 barrier 保证写回全部可见。
        _return[(self.num_max_tokens_per_rank * self.world,)](
            self.workspace,
            self.row_map,
            self.pointers[2],
            self.pointers[3],
            self.pointers[4],
            self.pointers[5],
            self.num_max_tokens_per_rank,
            self.hidden_size,
            self.topk,
            self.local_experts,
            self.rank,
            self.world,
            num_warps=config["return_num_warps"],
        )
        self.handles[5].barrier(channel=0, timeout_ms=10000)

        # 5. 一个 token 可能命中多个 owner rank；源 rank 对各 owner 的局部结果求和。
        output = alloc_tensor_func(hidden_states.shape, device=hidden_states.device, dtype=hidden_states.dtype)
        if rows:
            combine_rows = config["combine_rows_large"] if rows >= 16 else config["combine_rows_small"]
            combine_block = config["combine_block"]
            _combine[(triton.cdiv(rows, combine_rows), self.hidden_size // combine_block)](
                self.sources[5],
                self.sources[2],
                output,
                rows,
                self.hidden_size,
                self.topk,
                self.local_experts,
                combine_block,
                combine_rows,
                num_warps=config["combine_num_warps"],
            )
        return output
