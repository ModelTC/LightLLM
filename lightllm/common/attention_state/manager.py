from __future__ import annotations

from lightllm.common.attention_state.query_cache import CachedQueries, cached_query

from typing import TYPE_CHECKING
from typing import List

import torch

from lightllm.common.attention_state.types import PageKind

from lightllm.utils.dist_utils import get_current_device_id
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger

from .allocators import StatePageAllocator, FullAttKVAllocator
from .cpu_cache_operator import CpuCacheOperator
from .dp_shared_operator import DPSharedOperator
from .full_att import FullAttentionState
from .bounded_att import BoundedAttentionState
from .pd_operator import PDOperator
from .utils import MODEL_ROLES
from .lifetime import DeferredReclaimer

if TYPE_CHECKING:
    from easydict import EasyDict
    from lightllm.server.core.objs.start_args_type import StartArgs


logger = init_logger(__name__)


class AttentionStateManager(CachedQueries):
    """管理模型组合的注意力资源，提供缓存复用及 CPU/DP/PD 传输所需的聚合操作。

    通过聚合查询操作各类 State；模型通过两类创建接口决定资源组合，main/draft 共享本管理器。
    未支持的传输路径必须明确报错。

    CPU cache、DP shared 和 PD 的实现分别由三个 Operator 持有；管理器只保留
    State 注册、查询和共享容量规划。Operator 只依赖本管理器提供的 State 查询
    和 allocator，不复制 State 的算法逻辑。
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
        self.reclaimer = DeferredReclaimer()
        self._cpu_cache_operator = CpuCacheOperator(self)
        self._dp_shared_operator = DPSharedOperator(self)
        self._pd_operator = PDOperator(self)

    def _reset_operators(self) -> None:
        """让浅拷贝后的 Operator 指向新的资源管理器副本。"""
        self.reclaimer = DeferredReclaimer()
        self._cpu_cache_operator = CpuCacheOperator(self)
        self._dp_shared_operator = DPSharedOperator(self)
        self._pd_operator = PDOperator(self)

    def __getstate__(self):
        state = super().__getstate__()
        for name in ("_cpu_cache_operator", "_dp_shared_operator", "_pd_operator"):
            state.pop(name, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._reset_operators()

    def get_cpu_cache_operator(self) -> CpuCacheOperator:
        """返回当前资源管理器唯一的 CPU cache 操作对象。"""
        return self._cpu_cache_operator

    def get_dp_shared_operator(self) -> DPSharedOperator:
        """返回当前资源管理器唯一的 DP shared 操作对象。"""
        return self._dp_shared_operator

    def get_pd_operator(self) -> PDOperator:
        """返回当前资源管理器唯一的 PD 操作对象。"""
        return self._pd_operator

    def retire(self, completion, *, token_indices=None, big_pages=(), small_pages=(), keepalive=None):
        """Return resources only after their last asynchronous access completes."""
        # Allocator index buffers are reused; retain an independent CPU snapshot.
        tokens = None if token_indices is None else token_indices.detach().cpu().clone()
        big_pages, small_pages = list(big_pages), list(small_pages)
        if tokens is not None and self.full_att_allocator is None:
            raise ValueError("token retirement requires a KV allocator")
        if big_pages and self.bounded_att_big_page_allocator is None:
            raise ValueError("big page retirement requires an allocator")
        if small_pages and self.bounded_att_small_page_allocator is None:
            raise ValueError("small page retirement requires an allocator")

        def release():
            if tokens is not None:
                self.full_att_allocator.free(tokens)
            if big_pages:
                self.bounded_att_big_page_allocator.free(big_pages)
            if small_pages:
                self.bounded_att_small_page_allocator.free(small_pages)

        self.reclaimer.defer(completion, release, keepalive)

    def register_state(
        self,
        model_role: str,
        state: FullAttentionState | BoundedAttentionState,
        resource_id: str | None = None,
    ) -> None:
        """按模型角色和具体方案注册 State，保留原对象及其存储。

        成员名和 state_name 均为 {model_role}_{state.get_state_type()}_state。
        model_role 为 main 或 draft0..draft35；State 只返回中间的算法/存储方案标记，
        不自行添加角色或 _state。各 State 基类的 get_state_type 注释规定具体命名格式。
        同角色、同方案的异构资源可提供 resource_id，追加在方案名后；默认保持原命名。
        同角色、同标记不能重复注册；成员名是资源身份，不代表完整布局兼容性。
        """
        assert not self._state_registration_finished, "State registration is already finished"
        assert model_role in MODEL_ROLES, f"unknown model role {model_role}"
        state_type = state.get_state_type()
        assert state_type.isidentifier() and state_type.islower(), f"invalid State type {state_type}"
        if resource_id is not None:
            assert resource_id.isidentifier() and resource_id.islower(), f"invalid resource ID {resource_id}"
            state_type = f"{state_type}_{resource_id}"
        name = f"{model_role}_{state_type}_state"
        assert name not in self.__dict__, f"State {name} is already bound"
        assert all(
            state is not value for value in self.__dict__.values()
        ), "a State cannot belong to multiple model roles"
        # config 只描述布局，资源身份仅保存在 State；绑定时保留原配置对象。
        state.state_name = name
        setattr(self, name, state)

    @cached_query
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

    def read_full_att_calibration(self, args: StartArgs | EasyDict):
        """Read one startup snapshot shared by format planning and State binding."""
        path = args.kv_quant_calibration_config_path
        if not hasattr(self, "_calibration_snapshot"):
            import json

            calibration = None
            if path is not None:
                with open(path) as file:
                    calibration = json.load(file)
            self._calibration_snapshot = (path, calibration)
        previous_path, calibration = self._calibration_snapshot
        if path != previous_path:
            raise ValueError("calibration source cannot change during startup")
        return calibration

    def load_full_att_calibration(self, args: StartArgs | EasyDict) -> None:
        """全部 main/draft State 就绪后统一读取静态标定，并交给具体 State 绑定。

        文件沿用物理行顺序：main、draft0..draft35；同角色有多个 State 时按最小逻辑层号
        排序，State 内保留 logical_layer_groups 顺序。非静态 FP8 的 full State 同样占用
        对应物理行区间，共享逻辑层只占一行；CPU 布局创建不需要调用本流程。
        """
        from .full_att.gqa.static_fp8_state import StaticFP8FullAttentionState

        full_states = self.get_full_attention_states()
        calibration_states = [state for state in full_states if isinstance(state, StaticFP8FullAttentionState)]
        if not calibration_states:
            return

        # 阶段一：只读取一次文件；无文件时由 per_tensor State 绑定单位 scale。
        calibration = self.read_full_att_calibration(args)
        path = args.kv_quant_calibration_config_path
        if path is not None:
            if any(state.config.scale_mode == "per_tensor" for state in calibration_states):
                from lightllm.utils.config_utils import get_model_architectures

                assert calibration["architectures"] == get_model_architectures(
                    args.model_dir
                ), "calibration architecture does not match the model"

        # 阶段二：显式使用模型角色顺序，避免注册先后及 draft10 的字符串排序影响标定行。
        storage_layer_start = 0
        for role in MODEL_ROLES:
            states = sorted(
                self.get_full_attention_states(role),
                key=lambda state: (
                    min(layer for group in state.config.logical_layer_groups for layer in group),
                    state.state_name,
                ),
            )
            for state in states:
                if isinstance(state, StaticFP8FullAttentionState):
                    state.bind_calibration(
                        calibration=calibration,
                        storage_layer_start=storage_layer_start,
                    )
                storage_layer_start += len(state.config.logical_layer_groups)
        logger.info(
            "Full-attention calibration bound: states=%d, source=%s",
            len(calibration_states),
            path or "unit scales",
        )

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
            pd_reserved_bytes = self.get_pd_operator().get_page_bytes() * args.pd_kv_page_num
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
                    big_page_capacity=big_page_capacity,
                    small_page_capacity=small_page_capacity,
                    pin_memory=torch.device(device).type == "cuda",
                )
            self.bounded_att_cpu_load_big_page_slot = big_page_capacity - 2
            self.bounded_att_cpu_offload_big_page_slot = big_page_capacity - 1

        # 存储分配可能改变物理页规格，清除 State 的 CPU/PD 区域和管理器的派生布局缓存。
        for state in (*full_states, *bounded_states):
            state.clear_query_cache()
        # Layout cache 归属于各 Operator；State 存储规格变化后统一失效。
        self.get_cpu_cache_operator().clear_layout_cache()
        self.get_pd_operator().clear_layout_cache()

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
        assert isinstance(req_indices, list) and isinstance(
            page_slot_ids, list
        ), "request and slot indices must be lists"
        assert page_kind in ("big", "small"), f"unknown page kind {page_kind}"
        allocator = self.bounded_att_big_page_allocator if page_kind == "big" else self.bounded_att_small_page_allocator
        assert allocator is not None, f"{page_kind} pages are disabled"
        assert (
            len(page_slot_ids) == len(req_indices) == len(state_token_lens)
        ), "one length and slot are required per request"
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
        assert isinstance(req_indices, list) and isinstance(
            page_slot_ids, list
        ), "request and slot indices must be lists"
        assert page_kind in ("big", "small"), f"unknown page kind {page_kind}"
        allocator = self.bounded_att_big_page_allocator if page_kind == "big" else self.bounded_att_small_page_allocator
        assert allocator is not None, f"{page_kind} pages are disabled"
        assert (
            len(page_slot_ids) == len(req_indices) == len(state_token_lens)
        ), "one length and slot are required per request"
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
    # 四、CPU/DP/PD 操作均由对应 Operator 提供。
    # ========================================================================

    # ========================================================================
    # 七、私有辅助实现
    # ========================================================================

    # 私有实现：缓存创建完成后的成员集合，模型逐层计算直接访问具体 State 成员。
    @cached_query
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
