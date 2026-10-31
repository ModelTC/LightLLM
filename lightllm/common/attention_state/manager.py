from __future__ import annotations

from typing import TYPE_CHECKING
from collections.abc import Mapping
from copy import copy
from dataclasses import replace
from functools import lru_cache
from typing import List, Literal

import torch

from lightllm.common.attention_state.types import (
    StateTransferLayout,
    PageKind,
)

from lightllm.utils.dist_utils import get_current_device_id
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger

from .allocators import StatePageAllocator, FullAttKVAllocator
from .full_att import FullAttentionState
from .bounded_att import BoundedAttentionState
from .utils import MODEL_ROLES

if TYPE_CHECKING:
    from easydict import EasyDict
    from lightllm.server.core.objs.start_args_type import StartArgs


logger = init_logger(__name__)


class AttentionStateManager:
    """管理模型组合的注意力资源，提供缓存复用及 CPU/DP/PD 传输所需的聚合操作。

    通过聚合查询操作各类 State；模型通过两类创建接口决定资源组合，main/draft 共享本管理器。
    未支持的传输路径必须明确报错。
    """

    # ========================================================================
    # 一、基础管理：共享分配器、资源初始化与聚合查询
    # ========================================================================

    # 共享分配器：按模型需要启用；调用方直接分配和归还索引。
    full_att_allocator: FullAttKVAllocator | None
    bounded_att_big_page_allocator: StatePageAllocator | None
    bounded_att_small_page_allocator: StatePageAllocator | None
    _state_registration_finished: bool = False

    def __init__(self, request_capacity: int):
        """记录共享请求容量；State 和 allocator 由后续初始化流程设置。"""
        self.full_att_allocator = None
        self.bounded_att_big_page_allocator = None
        self.bounded_att_small_page_allocator = None
        self._request_capacity = request_capacity

    def register_state(
        self, model_role: str, state: FullAttentionState | BoundedAttentionState
    ) -> None:
        """按模型角色和具体方案注册 State，保留原对象及其存储。

        成员名和 state_name 均为 {model_role}_{state.get_state_type()}_state。
        model_role 为 main 或 draft0..draft35；State 只返回中间的算法/存储方案标记，
        不自行添加角色或 _state。各 State 基类的 get_state_type 注释规定具体命名格式。
        同角色、同标记不能重复注册；成员名是资源身份，不代表完整布局兼容性。
        """
        assert not self._state_registration_finished, "State registration is already finished"
        assert model_role in MODEL_ROLES, f"unknown model role {model_role}"
        state_type = state.get_state_type()
        assert state_type.isidentifier() and state_type.islower(), f"invalid State type {state_type}"
        name = f"{model_role}_{state_type}_state"
        assert name not in self.__dict__, f"State {name} is already bound"
        assert all(state is not value for value in self.__dict__.values()), "a State cannot belong to multiple model roles"
        # config 只描述布局，资源身份仅保存在 State；绑定时保留原配置对象。
        state.state_name = name
        setattr(self, name, state)

    @lru_cache(maxsize=None)
    def get_state(
        self,
        model_role: str,
        state_type_filter: str,
        logical_layer_id: int,
        state_class: type[FullAttentionState | BoundedAttentionState] | None = None,
    ) -> FullAttentionState | BoundedAttentionState:
        """按模型角色、算法/方案字符串和逻辑层选择唯一 State，缓存查询结果。

        model_role 为 main 或 draft0..draft35，按完整角色前缀匹配，避免 draft1 匹配 draft10。
        state_type_filter 在 get_state_type() 返回的字符串中做包含匹配，如 gqa、gqa_int8、mla_packed。
        通用推理模板可传空字符串，并通过 state_class 指定资源类别；具体算法传自己的标记。
        同层存在多类资源时，state_class 可进一步区分；最终必须恰好匹配一个 State。
        全部模型 State 创建完成后才允许查询；后续原地分配存储不改变缓存对象身份。
        """
        assert self._state_registration_finished, "State registration must finish before querying"
        assert model_role in MODEL_ROLES, f"unknown model role {model_role}"
        matches = [
            state
            for name, state in self.__dict__.items()
            if name.startswith(model_role + "_")
            and isinstance(state, (FullAttentionState, BoundedAttentionState))
            and state_type_filter in state.get_state_type()
            and (state_class is None or isinstance(state, state_class))
            and state.contains_logical_layer(logical_layer_id)
        ]
        assert len(matches) == 1, (
            f"expected exactly one attention State: {model_role=}, {state_type_filter=}, {logical_layer_id=}, "
            f"matches={[state.state_name for state in matches]}"
        )
        return matches[0]

    def load_full_att_calibration(self, args: StartArgs | EasyDict) -> None:
        """全部 main/draft State 就绪后统一读取静态标定，并交给具体 State 绑定。

        文件沿用物理行顺序：main、draft0..draft35；同角色有多个 State 时按最小逻辑层号
        排序，State 内保留 logical_layer_groups 顺序。非静态 FP8 的 full State 同样占用
        对应物理行区间，共享逻辑层只占一行；CPU 布局创建不需要调用本流程。
        """
        import json
        from .full_att.static_fp8_state import StaticFP8FullAttentionState

        full_states = self.get_full_attention_states()
        calibration_states = [state for state in full_states if isinstance(state, StaticFP8FullAttentionState)]
        if not calibration_states:
            return

        # 阶段一：只读取一次文件；无文件时由 per_tensor State 绑定单位 scale。
        calibration = None
        path = args.kv_quant_calibration_config_path
        if path is not None:
            with open(path) as file:
                calibration = json.load(file)
            if any(state.config.scale_mode == "per_tensor" for state in calibration_states):
                from lightllm.utils.config_utils import get_model_architectures

                assert calibration["architectures"] == get_model_architectures(args.model_dir), (
                    "calibration architecture does not match the model"
                )

        # 阶段二：显式使用模型角色顺序，避免注册先后及 draft10 的字符串排序影响标定行。
        storage_layer_start = 0
        for role in MODEL_ROLES:
            states = sorted(
                self.get_full_attention_states(role),
                key=lambda state: (
                    min(layer for group in state.config.logical_layer_groups for layer in group), state.state_name,
                ),
            )
            for state in states:
                if isinstance(state, StaticFP8FullAttentionState):
                    state.bind_calibration(
                        calibration=calibration,
                        storage_layer_start=storage_layer_start,
                    )
                storage_layer_start += len(state.config.logical_layer_groups)
        logger.info("Full-attention calibration bound: states=%d, source=%s", len(calibration_states), path or "unit scales")

    def initialize_storage(
        self,
        args: StartArgs | EasyDict,
        token_capacity: int | None,
        mem_fraction: float,
        device: str | torch.device | int = "cuda",
    ) -> None:
        """全部模型完成资源准备后，规划共享容量并初始化缓存存储。

        调用前，所有模型的权重和绑定请求容量的运行态已创建，full-attention
        已完成创建标记和所需的静态标定绑定，大小页池尚未分配。调用方在本方法返回后，
        依次执行各模型的运行时初始化，由模型读取 allocator 中的最终容量。
        """
        assert self._state_registration_finished, "State registration must finish before resource initialization"

        # 阶段一：规划共享 token 容量，并向下对齐到完整 KV 页。
        # 未指定容量时，根据剩余显存评估；固定运行态的显存已经在前面占用。
        # PD 模式还需为传输 buffer 预留空间，布局按全部 full-attention State 计算。
        assert self.get_full_attention_states(), "the shared token index domain requires a full-attention State"
        pd_reserved_bytes = 0
        if args.run_mode in ("prefill", "decode"):
            pd_reserved_bytes = self.get_pd_page_bytes() * args.pd_kv_page_num
        if token_capacity is None:
            # 权重和固定运行态已分配，用当前设备剩余显存扣除保留比例及 PD buffer。
            torch.cuda.empty_cache()
            gpu_device = torch.device("cuda", get_current_device_id())
            free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_device)
            budget = int(free_bytes - total_bytes * (1 - mem_fraction)) - pd_reserved_bytes

            # 按每页开销直接计算容量，预留一个 HOLD 页；静态量化参数占比较小，不单独扣减。
            assert args.page_size > 0, "capacity planning requires a positive page size"
            page_bytes = self.get_full_att_token_bytes() * args.page_size
            if page_bytes > budget:
                raise MemoryError("attention storage budget cannot fit HOLD pages")
            token_capacity = (budget // page_bytes - 1) * args.page_size

            # 各 rank 的显存余量可能不同，统一采用最小容量。
            if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                capacity = torch.tensor(token_capacity, dtype=torch.int64, device=gpu_device)
                torch.distributed.all_reduce(capacity, op=torch.distributed.ReduceOp.MIN)
                token_capacity = capacity.item()
            logger.info(
                f"Attention storage budget: {budget / 1024**3:.3f} GiB, "
                f"transfer reserve: {pd_reserved_bytes / 1024**2:.3f} MiB, "
                f"profiled token capacity: {token_capacity} (mem_fraction={mem_fraction})"
            )
        token_capacity = token_capacity // args.page_size * args.page_size
        assert token_capacity > 0, "attention capacity must contain at least one complete page"

        # 阶段二：管理器为已创建的 State 分配实际存储与共享 allocator。
        # State 对象保持不变，查询缓存继续引用相同对象；固定运行态不重复分配。
        big_page_tokens = args.linear_att_hash_page_size * args.linear_att_page_block_num
        self.allocate_storage(
            token_capacity=token_capacity,
            page_size=args.page_size,
            big_page_tokens=big_page_tokens,
            small_page_capacity=args.linear_att_cache_size or 0,
            device=device,
        )

    # 创建完成只表示对象集合就绪；实际容量和存储在后续阶段初始化。
    def finish_state_registration(self) -> None:
        """确认全部模型的 State 对象已创建并装配完成，允许查询 meta 定义并开始容量规划。"""
        assert not self._state_registration_finished, "State registration is already finished"
        self._state_registration_finished = True

    def allocate_storage(
        self,
        token_capacity: int,
        page_size: int,
        big_page_tokens: int,
        small_page_capacity: int = 0,
        device: str | torch.device | int = "cuda",
    ) -> None:
        """在 State 对象创建完成后分配 full-attention 存储、共享 allocator 和 bounded 大小页。

        请求容量对应的 bounded 运行态（包括压缩窗口）已由各模型创建，这里仅补齐缓存存储。
        """
        assert self._state_registration_finished, "State registration must finish before storage initialization"
        assert self.full_att_allocator is None, "resources already initialized"
        full_states = self.get_full_attention_states()
        assert full_states, "the shared token index domain requires a full-attention State"
        for state in full_states:
            # 管理器统一设置实际页规格和设备；State 的分配接口只接收 token 容量。
            state.page_size = page_size
            state.device = torch.device(device)
            state.allocate_storage(token_capacity)
        self.full_att_allocator = FullAttKVAllocator(token_capacity, page_size)

        # bounded-attention 大页池末尾保留 CPU 加载/卸载暂存槽位，普通分配不使用它们。
        bounded_states = self.get_bounded_attention_states()
        if bounded_states:
            assert big_page_tokens > 0, "bounded attention pages require a positive token boundary"
            big_page_capacity = (token_capacity + big_page_tokens - 1) // big_page_tokens + 2
            self.bounded_att_big_page_allocator = StatePageAllocator(big_page_capacity, reserved_slots=2)
            self.bounded_att_small_page_allocator = (
                StatePageAllocator(small_page_capacity) if small_page_capacity else None
            )
            for state in bounded_states:
                # 无快照区域的 State 在合法缓存边界无需历史，不为它分配大小页 Tensor。
                # 共享槽位 allocator 仍供其他 State 和缓存流程使用，不随单个 State 重复创建。
                if not state.get_cpu_transfer_regions():
                    continue
                state.initialize_page_pools(
                    big_page_capacity=big_page_capacity, small_page_capacity=small_page_capacity,
                    pin_memory=torch.device(device).type == "cuda",
                )
            self.bounded_att_cpu_load_big_page_slot = big_page_capacity - 2
            self.bounded_att_cpu_offload_big_page_slot = big_page_capacity - 1

        # 存储分配可能改变物理页规格，清除 State 的 CPU/PD 区域和管理器的派生布局缓存。
        for state in (*full_states, *bounded_states):
            state.get_cpu_transfer_regions.cache_clear()
            state.get_pd_transfer_regions.cache_clear()
        self.get_cpu_page_layouts.cache_clear()
        self.get_cpu_page_bytes.cache_clear()
        self.get_pd_token_page_layouts.cache_clear()
        self.get_pd_runtime_page_layouts.cache_clear()
        self.get_pd_page_bytes.cache_clear()

    def create_pd_ipc_copy(self) -> AttentionStateManager:
        """创建 PD 导出副本，保留运行态 Tensor，排除 allocator 和全部大小页池。"""
        ipc_copy = copy(self)
        ipc_copy.full_att_allocator = None
        ipc_copy.bounded_att_big_page_allocator = None
        ipc_copy.bounded_att_small_page_allocator = None

        # PD 只传运行态；先复制 State 对象再移除页池，避免修改推理进程的资源。
        for state in self.get_bounded_attention_states():
            state_copy = copy(state)
            state_copy.big_pages_pool = None
            state_copy.small_pages_pool = None
            setattr(ipc_copy, state.state_name, state_copy)
        return ipc_copy

    def create_dp_ipc_copy(self) -> AttentionStateManager:
        """创建 DP shared 导出副本，保留运行态和共享大小页池，仅排除 allocator。"""
        # DP fetch 初始化已选择共享 CPU 页池；禁止导出私有页池时隐式分配、复制整块存储。
        for state in self.get_bounded_attention_states():
            for pool in (state.big_pages_pool, state.small_pages_pool):
                if pool is not None:
                    for tensor in vars(pool).values():
                        if isinstance(tensor, torch.Tensor):
                            assert tensor.device.type == "cpu" and tensor.is_shared(), "DP IPC requires shared CPU page pools"
        ipc_copy = copy(self)
        ipc_copy.full_att_allocator = None
        ipc_copy.bounded_att_big_page_allocator = None
        ipc_copy.bounded_att_small_page_allocator = None
        return ipc_copy

    def register_page_pools_pin_memory(self) -> None:
        """在当前 GPU 上统一注册所有 bounded State 的大小页池，供 DP 接收端调用。

        调用方先选择本进程设备；注册状态不随 IPC 传递，重复调用会跳过已注册存储。
        本接口只注册本地映射，不申请槽位或改变原进程的 allocator 所有权。
        """
        for state in self.get_bounded_attention_states():
            if state.big_pages_pool is not None:
                state.big_pages_pool.register_pin_memory()
            if state.small_pages_pool is not None:
                state.small_pages_pool.register_pin_memory()

    def get_full_attention_states(self, role: str | None = None) -> tuple[FullAttentionState, ...]:
        """返回全部或指定角色的 full-attention State；只缓存对象引用。"""
        return self._get_states(FullAttentionState, role)

    def get_bounded_attention_states(self, role: str | None = None) -> tuple[BoundedAttentionState, ...]:
        """返回全部或指定角色的 bounded-attention State。"""
        return self._get_states(BoundedAttentionState, role)

    def get_full_att_token_bytes(self) -> int:
        """累计当前 TP rank 全部 full-attention State 的单 token 存储字节数。"""
        total = 0
        for state in self.get_full_attention_states():
            total += state.get_token_bytes()
        return total

    @lru_cache(maxsize=None)
    def get_cpu_page_bytes(self) -> int:
        """从 CPU 联合页布局的最大终点计算物理页大小，包含字段及 State 间的对齐空间。"""
        page_bytes = 0
        for layout in self.get_cpu_page_layouts().values():
            page_bytes = max(page_bytes, layout.byte_end)
        # State 区间按 8 字节对齐，整个 CPU 物理页再按 16 字节对齐。
        return (page_bytes + 15) // 16 * 16

    @lru_cache(maxsize=None)
    def get_pd_page_bytes(self) -> int:
        """取 KV 页与完整运行态页占用的最大值，保证两类 PD 任务都能容纳。"""
        full_layouts = self.get_pd_token_page_layouts()
        runtime_layouts = self.get_pd_runtime_page_layouts()
        full_att_page_bytes = 0
        for layout in full_layouts.values():
            full_att_page_bytes = max(full_att_page_bytes, layout.byte_end)
        runtime_page_bytes = 0
        for layout in runtime_layouts.values():
            runtime_page_bytes = max(runtime_page_bytes, layout.byte_end)

        # KV 页不足时扩大物理传输页，不改变 KV 页的 token 数或各 State 的布局。
        if full_layouts and full_att_page_bytes < runtime_page_bytes:
            logger.warning(
                f"PD KV page size ({full_att_page_bytes} bytes) is smaller than the complete attention runtime "
                f"({runtime_page_bytes} bytes); using {runtime_page_bytes} bytes per transfer page"
            )
        # 纯 KV 或纯运行态取各自占用；没有任何参与资源时返回 0。
        return max(full_att_page_bytes, runtime_page_bytes)

    # ========================================================================
    # 二、Full attention：token 数据复制
    # ========================================================================

    # Full attention：在所有参与 State 内复制相同的物理 token 索引，不修改请求映射。
    def copy_full_att_tokens(self, src_indices: torch.Tensor, dst_indices: torch.Tensor) -> None:
        """复制 KV 及随 token 保存的量化参数、indexer 字段等完整 token 数据。"""
        full_states = self.get_full_attention_states()
        assert full_states, "no full-attention storage in this composition"
        for state in full_states:
            state.copy_full_att_tokens(src_indices, dst_indices)

    # ========================================================================
    # 三、Bounded attention：运行态重置、大小页保存与恢复、MTP 提交
    # ========================================================================

    # 空前缀初始化：只处理 bounded attention；无此类 State 时可为空操作。
    def reset_bounded_att_state(self, req_indices: List[int]) -> None:
        """将指定槽位的 bounded 运行态及控制信息恢复到长度 0，保留 KV 和状态页。"""
        for state in self.get_bounded_attention_states():
            state.reset_bounded_att_state(req_indices)

    # Bounded attention 状态页：供 radix/CPU cache 复用；未启用对应页池时操作报错。
    # 调用方提供已分配的大小页槽位；发布前完成保存，解除引用并结束使用后直接归还 allocator。
    def save_bounded_att_runtime_to_pages(
        self,
        req_indices: List[int],
        state_token_lens: List[int],
        page_kind: PageKind,
        page_slot_ids: List[int],
    ) -> None:
        """按各请求已执行的绝对 token 长度保存全部 bounded State；三个列表逐项对应。"""
        bounded_states = self.get_bounded_attention_states()
        assert isinstance(req_indices, list) and isinstance(page_slot_ids, list), "request and slot indices must be lists"
        assert page_kind in ("big", "small"), f"unknown page kind {page_kind}"
        allocator = self.bounded_att_big_page_allocator if page_kind == "big" else self.bounded_att_small_page_allocator
        assert allocator is not None, f"{page_kind} pages are disabled"
        assert len(page_slot_ids) == len(req_indices) == len(state_token_lens), "one length and slot are required per request"
        for state in bounded_states:
            state.save_bounded_att_runtime_to_pages(
                req_indices=req_indices,
                state_token_lens=state_token_lens,
                page_kind=page_kind,
                page_slot_ids=page_slot_ids,
            )

    def restore_bounded_att_runtime_from_pages(
        self,
        req_indices: List[int],
        state_token_lens: List[int],
        page_kind: PageKind,
        page_slot_ids: List[int],
    ) -> None:
        """按各快照对应的绝对 token 长度恢复运行态及 ring 位置等控制信息。"""
        bounded_states = self.get_bounded_attention_states()
        assert isinstance(req_indices, list) and isinstance(page_slot_ids, list), "request and slot indices must be lists"
        assert page_kind in ("big", "small"), f"unknown page kind {page_kind}"
        allocator = self.bounded_att_big_page_allocator if page_kind == "big" else self.bounded_att_small_page_allocator
        assert allocator is not None, f"{page_kind} pages are disabled"
        assert len(page_slot_ids) == len(req_indices) == len(state_token_lens), "one length and slot are required per request"
        assert not page_slot_ids or bounded_states, "state restore requires participating bounded states"
        for state in bounded_states:
            state.restore_bounded_att_runtime_from_pages(
                req_indices=req_indices,
                state_token_lens=state_token_lens,
                page_kind=page_kind,
                page_slot_ids=page_slot_ids,
            )

    # MTP 推测解码：只提交指定角色的已接受运行态，由具体 State 管理候选区和游标。
    def commit_bounded_att_state(
        self,
        req_indices: torch.Tensor,
        accepted_steps: torch.Tensor,
        role: str,
    ) -> None:
        """传入 CUDA 请求索引和 int32 接受步数，包含基准输入步；无参与 State 时为空操作。"""
        assert req_indices.is_cuda and accepted_steps.is_cuda, "commit inputs must be CUDA tensors"
        assert role in MODEL_ROLES, f"unknown model role {role}"
        for state in self.get_bounded_attention_states(role):
            state.commit_bounded_att_state(req_indices, accepted_steps)

    # ========================================================================
    # 四、CPU Cache 数据交换
    # ========================================================================

    @lru_cache(maxsize=None)
    def get_cpu_page_layouts(self) -> Mapping[str, StateTransferLayout]:
        """聚合各 State 的 CPU 字段区间，返回按 state_name 索引的联合页布局。

        使用全局 cpu_cache_token_page_size，按需缓存；不依赖页大小查询。
        """
        page_tokens = get_env_start_args().cpu_cache_token_page_size
        assert page_tokens > 0, "CPU pages require a positive token capacity"
        states = (*self.get_full_attention_states(), *self.get_bounded_attention_states())
        assert states, "a CPU page requires participating storage"

        # 按 full-attention、bounded 顺序组织全部角色；各类 State 内部按资源名排序。
        # 无重叠窗口在缓存组边界无需快照，由 State 返回空 CPU 区域。
        layouts, offset = {}, 0
        for state in states:
            regions = tuple(
                replace(region, byte_start=offset + region.byte_start, byte_end=offset + region.byte_end)
                for region in state.get_cpu_transfer_regions()
            )
            # 无跨步重叠的压缩器在组边界不需要快照，不占用 CPU 页。
            if not regions:
                continue
            # 以实际字段终点确定 State 占用，保留内部 padding，并对齐下一个 State 的起点。
            state_end = (max(region.byte_end for region in regions) + 7) // 8 * 8
            layouts[state.state_name] = StateTransferLayout(
                state_name=state.state_name,
                tokens_per_page=page_tokens,
                byte_start=offset,
                byte_end=state_end,
                regions=regions,
            )
            offset = state_end
        return layouts

    # CPU cache：按 layouts 组织联合页；混合模型的每页都包含全部参与 bounded State 的快照。
    # 纯 full-attention 模型只传 KV；具体实现负责校验格式、长度及各资源的区域映射。
    def write_req_to_cpu(
        self,
        token_indices: torch.Tensor,
        cpu_page_buffers: torch.Tensor,
        cpu_page_indices: List[int],
        layouts: Mapping[str, StateTransferLayout],
        cpu_page_ready_flags: List[bool] | None = None,
        big_page_slot_ids: List[int] | None = None,
        small_page_slot_ids: List[int] | None = None,
    ) -> None:
        """将单个请求本次指定前缀页面的 KV 和 bounded 快照写入 CPU cache。

        cpu_page_indices 前段对应 big_page_slot_ids，后段对应 small_page_slot_ids，
        各段按列表顺序一一对应；未参与的页池可传 None 或空列表。
        cpu_page_ready_flags 与 CPU 页一一对应，True 跳过整页；None 表示全部写出。
        """
        if cpu_page_ready_flags is not None:
            assert len(cpu_page_ready_flags) == len(cpu_page_indices), "CPU page and ready counts differ"
            if all(cpu_page_ready_flags):
                return
        full_states = self.get_full_attention_states()
        bounded_states = self.get_bounded_attention_states()
        if bounded_states:
            big_page_slot_ids = [] if big_page_slot_ids is None else big_page_slot_ids
            small_page_slot_ids = [] if small_page_slot_ids is None else small_page_slot_ids
            big_page_count = len(big_page_slot_ids)
            assert big_page_count + len(small_page_slot_ids) == len(cpu_page_indices), "one state slot is required per CPU page"
            big_cpu_page_indices = cpu_page_indices[:big_page_count]
            small_cpu_page_indices = cpu_page_indices[big_page_count:]
            big_page_readies = cpu_page_ready_flags[:big_page_count] if cpu_page_ready_flags is not None else None
            small_page_readies = cpu_page_ready_flags[big_page_count:] if cpu_page_ready_flags is not None else None

        # 阶段一：统一写出本批 full-attention token 数据。
        for state in full_states:
            state.write_to_cpu(
                token_indices=token_indices,
                cpu_page_buffers=cpu_page_buffers,
                cpu_page_indices=cpu_page_indices,
                layout=layouts[state.state_name],
                cpu_page_ready_flags=cpu_page_ready_flags,
            )

        # 阶段二：分别从大小页池取快照，写入对应 CPU 页面，不修改请求运行态。
        for state in bounded_states:
            layout = layouts.get(state.state_name)
            if layout is None:
                # 部分 bounded 状态在缓存边界没有需保留的历史（如 W128/S128 压缩器），不写入 CPU cache。
                # 仅允许未声明 CPU 传输区域的 State 跳过，避免必要快照因布局遗漏而丢失。
                assert not state.get_cpu_transfer_regions(), f"missing CPU layout for {state.state_name}"
                continue
            if big_page_slot_ids:
                state.write_state_pages_to_cpu(
                    page_kind="big",
                    page_slot_ids=big_page_slot_ids,
                    cpu_page_buffers=cpu_page_buffers,
                    cpu_page_indices=big_cpu_page_indices,
                    layout=layout,
                    cpu_page_ready_flags=big_page_readies,
                )
            if small_page_slot_ids:
                state.write_state_pages_to_cpu(
                    page_kind="small",
                    page_slot_ids=small_page_slot_ids,
                    cpu_page_buffers=cpu_page_buffers,
                    cpu_page_indices=small_cpu_page_indices,
                    layout=layout,
                    cpu_page_ready_flags=small_page_readies,
                )

    def read_req_from_cpu(
        self,
        cpu_page_buffers: torch.Tensor,
        cpu_page_indices: List[int],
        token_indices: torch.Tensor,
        layouts: Mapping[str, StateTransferLayout],
        big_page_slot_ids: List[int] | None = None,
        req_idx: int | None = None,
        state_token_len: int | None = None,
    ) -> None:
        """读回单个请求本次指定的有序前缀页面，并从末页恢复 req_idx 的 bounded 运行态。

        big_page_slot_ids 只包含完整 CPU 页对应的普通槽位；末尾碎页使用内部预留的 LOAD 大页。
        存在 bounded 时须提供 state_token_len，即恢复后的绝对前缀长度，不是本批 token 数。
        存在 bounded 时必须提供 req_idx；最终前缀须满足各 State 的快照或重置边界。
        """
        full_states = self.get_full_attention_states()
        bounded_states = self.get_bounded_attention_states()
        if bounded_states:
            assert req_idx is not None and cpu_page_indices, "state recovery requires a target request and prefix pages"
        if bounded_states:
            assert state_token_len is not None, "bounded state recovery requires its absolute prefix length"
            # CPU 页的 token 数与 bounded 大页边界在启动时统一，直接读取全局配置。
            page_tokens = get_env_start_args().cpu_cache_token_page_size
            # token 索引只用于判断本批末页是否完整，不读取 GPU 上的索引值。
            has_partial_page = token_indices.numel() % page_tokens != 0
            assert isinstance(big_page_slot_ids, list), "slot indices must be a list"
            assert len(big_page_slot_ids) + int(has_partial_page) == len(cpu_page_indices), "slots must cover complete CPU pages only"
            # 只扩展内部列表，预留槽位不写回调用方，也不作为普通大页发布。
            load_big_page_slot_ids = big_page_slot_ids.copy()
            if has_partial_page:
                load_big_page_slot_ids.append(self.bounded_att_cpu_load_big_page_slot)

        # 阶段一：恢复 full-attention token 数据；各 State 负责对应布局的搬运约束。
        for state in full_states:
            state.read_from_cpu(
                cpu_page_buffers=cpu_page_buffers,
                cpu_page_indices=cpu_page_indices,
                token_indices=token_indices,
                layout=layouts[state.state_name],
            )

        # 阶段二：完整页和末尾碎页统一读入大页池，碎页使用追加的 LOAD 槽位。
        for state in bounded_states:
            if state.state_name not in layouts:
                # 无跨步历史的 bounded 状态不占 CPU 页，因此无需读回快照；运行态恢复仍在下一阶段处理。
                # 有 CPU 传输区域的 State 必须具备布局，不能静默跳过。
                assert not state.get_cpu_transfer_regions(), f"missing CPU layout for {state.state_name}"
                continue
            state.read_state_pages_from_cpu(
                cpu_page_buffers=cpu_page_buffers,
                cpu_page_indices=cpu_page_indices,
                big_page_slot_ids=load_big_page_slot_ids,
                layout=layouts[state.state_name],
            )

        # 阶段三：从末页恢复请求运行态；纯碎页也走 LOAD 槽位。
        if bounded_states:
            self.restore_bounded_att_runtime_from_pages(
                req_indices=[req_idx],
                state_token_lens=[state_token_len],
                page_kind="big",
                page_slot_ids=[load_big_page_slot_ids[-1]],
            )

    # ========================================================================
    # 五、DP 迁移
    # ========================================================================

    def copy_from_dp(
        self,
        remote_att_manager: AttentionStateManager,
        src_token_indices: torch.Tensor,
        dst_token_indices: torch.Tensor,
        src_big_page_ids: List[int],
        dst_big_page_ids: List[int],
        src_req_idx: int,
        dst_req_idx: int,
    ) -> None:
        """复制其他 DP 同一 TP 分片的 token、大页和指定请求的 prefill 基准运行态。

        token 索引由调用方准备在源、目标各自的 CUDA 设备上，目标索引为 int64。
        不需要复制 token 或大页时传空 Tensor 或空列表；请求运行态仍按请求索引复制。
        调用方保证源 MTP 游标为 0、目标前缀满足各 State 的缓存边界，以及源页就绪和引用有效。
        """
        assert type(remote_att_manager) is type(self), "DP replicas must use the same model resource manager"
        assert src_token_indices.numel() == dst_token_indices.numel(), "source and destination token counts differ"
        assert len(src_big_page_ids) == len(dst_big_page_ids), "source and destination page counts differ"
        if src_token_indices.numel():
            for state in self.get_full_attention_states():
                state.copy_tokens_from_dp(
                    src_token_indices=src_token_indices,
                    dst_token_indices=dst_token_indices,
                    remote_dp_full_att_state=getattr(remote_att_manager, state.state_name),
                )
        if src_big_page_ids:
            for state in self.get_bounded_attention_states():
                state.copy_big_pages_from_dp(
                    src_page_ids=src_big_page_ids,
                    dst_page_ids=dst_big_page_ids,
                    remote_dp_bounded_state=getattr(remote_att_manager, state.state_name),
                )
        # 运行态由具体 State 复制或按合法缓存边界重置，不由大页复制隐式恢复。
        for state in self.get_bounded_attention_states():
            state.copy_runtime_state_from_dp(
                src_req_indices=[src_req_idx],
                dst_req_indices=[dst_req_idx],
                remote_dp_bounded_state=getattr(remote_att_manager, state.state_name),
            )

    # ========================================================================
    # 六、PD 分离传输
    # ========================================================================

    @lru_cache(maxsize=None)
    def get_pd_token_page_layouts(self) -> Mapping[str, StateTransferLayout]:
        """按实际字段区间生成 kv 布局，各 State 记录自身在页内的起点与终点。

        使用全局 pd_kv_page_size，按需缓存；不依赖页大小查询或运行态布局。
        """
        page_tokens = get_env_start_args().pd_kv_page_size
        assert page_tokens > 0, "PD pages require a positive token capacity"
        layouts, offset = {}, 0
        for state in self.get_full_attention_states():
            regions = tuple(
                replace(region, byte_start=offset + region.byte_start, byte_end=offset + region.byte_end)
                for region in state.get_pd_transfer_regions()
            )
            # 字段保留自身数据区间，State 终点包含留给下一个资源的对齐空间。
            state_end = (max(region.byte_end for region in regions) + 7) // 8 * 8
            layouts[state.state_name] = StateTransferLayout(
                state_name=state.state_name,
                tokens_per_page=page_tokens,
                byte_start=offset,
                byte_end=state_end,
                regions=regions,
            )
            offset = state_end
        return layouts

    @lru_cache(maxsize=None)
    def get_pd_runtime_page_layouts(self) -> Mapping[str, StateTransferLayout]:
        """独立生成 att_state 布局，记录各运行态 State 的完整字节区间。

        每个 State 保存一份运行态，tokens_per_page 为 None；无 State 时返回空字典。
        """
        layouts, offset = {}, 0
        for state in self.get_bounded_attention_states():
            regions = tuple(
                replace(region, byte_start=offset + region.byte_start, byte_end=offset + region.byte_end)
                for region in state.get_pd_transfer_regions()
            )
            state_end = (max(region.byte_end for region in regions) + 7) // 8 * 8
            layouts[state.state_name] = StateTransferLayout(
                state_name=state.state_name,
                tokens_per_page=None,
                byte_start=offset,
                byte_end=state_end,
                regions=regions,
            )
            offset = state_end
        return layouts

    # PD 分离：kind="kv" 传输 token 数据，kind="att_state" 传输 bounded 运行态。
    # pd_page_buffers 第一维为页，pd_page_index 选择物理页；网络收发与同步由传输流程组织。
    def write_to_pd_page(
        self,
        pd_page_buffers: torch.Tensor,
        pd_page_index: int,
        kind: Literal["kv", "att_state"],
        layouts: Mapping[str, StateTransferLayout],
        token_indices: torch.Tensor | None = None,
        req_idx: int | None = None,
        state_token_len: int | None = None,
    ) -> None:
        """按任务布局写出 PD 页；att_state 必须传入已执行的绝对前缀长度 state_token_len。"""
        match kind:
            case "kv":
                # KV 页只写出 full-attention 数据；各 State 负责自身布局与容量约束。
                assert token_indices is not None, "KV PD pages require token indices"
                for state in self.get_full_attention_states():
                    state.write_to_pd_page(
                        token_indices=token_indices,
                        pd_page_buffers=pd_page_buffers,
                        pd_page_index=pd_page_index,
                        layout=layouts[state.state_name],
                    )

            case "att_state":
                # 前缀长度由 PD 任务提供，各 State 据此定位自己的运行态窗口。
                assert req_idx is not None, "state PD pages require a request index"
                assert state_token_len is not None and state_token_len >= 0, "PD state requires its prefill prefix length"

                # 阶段一：写出 bounded 的 prefill 基准运行态。
                for state in self.get_bounded_attention_states():
                    state.write_to_pd_page(
                        req_idx=req_idx,
                        pd_page_buffers=pd_page_buffers,
                        pd_page_index=pd_page_index,
                        state_token_len=state_token_len,
                        layout=layouts[state.state_name],
                    )

            case _:
                raise AssertionError(f"unknown PD page kind {kind}")

    def read_from_pd_page(
        self,
        pd_page_buffers: torch.Tensor,
        pd_page_index: int,
        kind: Literal["kv", "att_state"],
        layouts: Mapping[str, StateTransferLayout],
        token_indices: torch.Tensor | None = None,
        req_idx: int | None = None,
        state_token_len: int | None = None,
    ) -> None:
        """按布局恢复 PD 页；att_state 使用绝对前缀长度 state_token_len 恢复运行态位置。"""
        match kind:
            case "kv":
                # KV 页只恢复 full-attention 数据；各 State 负责自身布局与容量约束。
                assert token_indices is not None, "KV PD pages require token indices"
                for state in self.get_full_attention_states():
                    state.read_from_pd_page(
                        pd_page_buffers=pd_page_buffers,
                        token_indices=token_indices,
                        pd_page_index=pd_page_index,
                        layout=layouts[state.state_name],
                    )

            case "att_state":
                # 前缀长度由 PD 任务提供，各 State 据此定位自己的运行态窗口。
                assert req_idx is not None, "state PD pages require a request index"
                assert state_token_len is not None and state_token_len >= 0, "PD state requires its prefill prefix length"

                # 阶段一：恢复 bounded 的 prefill 基准运行态。
                for state in self.get_bounded_attention_states():
                    state.read_from_pd_page(
                        pd_page_buffers=pd_page_buffers,
                        req_idx=req_idx,
                        pd_page_index=pd_page_index,
                        state_token_len=state_token_len,
                        layout=layouts[state.state_name],
                    )

            case _:
                raise AssertionError(f"unknown PD page kind {kind}")

    # ========================================================================
    # 七、私有辅助实现
    # ========================================================================

    # 私有实现：缓存创建完成后的成员集合，模型逐层计算直接访问具体 State 成员。
    @lru_cache(maxsize=None)
    def _get_states(
        self,
        state_class: type[FullAttentionState] | type[BoundedAttentionState],
        role: str | None,
    ):
        """按 State 类型和模型角色遍历规范成员，只缓存对象引用。"""
        assert self._state_registration_finished, "State registration must finish before querying"
        assert role is None or role in MODEL_ROLES, f"unknown model role {role}"
        states = (
            state
            for state in self.__dict__.values()
            if isinstance(state, state_class) and (role is None or state.state_name.startswith(role + "_"))
        )
        return tuple(sorted(states, key=lambda state: state.state_name))
