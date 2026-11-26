"""按模型角色和逻辑层注册 attention 后端，并准备每个微批次的状态。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from .base_backend import AttentionBackend, PrefillAttBackend, DecodeAttBackend

if TYPE_CHECKING:
    from lightllm.common.basemodel.infer_struct import InferStateInfo


class AttentionBackendManager:
    """按模型角色和推理阶段注册后端，并准备每个微批次的状态。

    role 是本管理器内区分各模型后端的字符串键，取自 model.att_state_role。
    主模型的键为 "main"。草稿模型的键为 "draft{i}"，i 是从 0 开始的创建序号。
    第一个是 "draft0"，第二个是 "draft1"，第三个是 "draft2"，最多 36 个，末尾是 "draft35"。
    登记和查询使用同一个键，查询时精确匹配完整字符串。
    phase 的有效值为 "prefill" 和 "decode"，分别对应 PrefillAttBackend 和 DecodeAttBackend。
    同一角色和阶段内，完整配置相同的层可共用一个后端实例。
    每个角色在预热和捕获 CUDA Graph 之前完成注册。
    """

    # ========================================================================
    # 一、初始化与具名注册
    # ========================================================================

    def __init__(self):
        # 每个已登记的角色都有阶段清单，纯 FFN 模型的清单为空。
        self._backends_by_phase: dict[str, dict[str, tuple[AttentionBackend, ...]]] = {}
        # 推理时按角色、阶段和逻辑层号直接查找。
        self._backends_by_layer: dict[tuple[str, str, int], AttentionBackend] = {}

    def register_backends(self, role: str, backends: Iterable[AttentionBackend]) -> None:
        """一次登记指定角色的全部后端，每个后端保留为具名成员。

        参数
            role 是本次登记后端所用的键，取自 model.att_state_role。
            主模型用 "main"，草稿模型依次用 "draft0"、"draft1"、"draft2" 等，最大编号为 "draft35"。
            同一个键在本管理器中只能登记一次。
            backends 是模型已创建的 PrefillAttBackend 或 DecodeAttBackend 实例的可迭代对象。
            每个实例的 role 必须与入参一致，logical_layer_ids 至少包含一个全局逻辑层号。
            同一阶段内每个层号只能归属一个后端，两个阶段可以登记相同层号。
            纯 FFN 角色传空可迭代对象，也会完成该角色的注册。

        异常
            重复注册角色时抛出 RuntimeError。元素类型不符时抛出 TypeError。
            实例角色不一致、层号为空、同阶段层号重复或注册名已占用时抛出 ValueError。
        """
        if role in self._backends_by_phase:
            raise RuntimeError(f"attention backend registration is already finished for {role}")

        members = {}
        layers = {}
        phases = {"prefill": [], "decode": []}
        for backend in backends:
            if isinstance(backend, PrefillAttBackend):
                phase = "prefill"
            elif isinstance(backend, DecodeAttBackend):
                phase = "decode"
            else:
                raise TypeError("expected a PrefillAttBackend or DecodeAttBackend")
            implementation = backend.get_name()
            if backend.role != role:
                raise ValueError(f"cannot register {backend.role}:{implementation} for {role}")
            if not backend.logical_layer_ids:
                raise ValueError(f"{backend.role}:{implementation} serves no logical layers")
            for layer_id in backend.logical_layer_ids:
                key = (role, phase, layer_id)
                if key in layers:
                    raise ValueError(f"{role} layer {layer_id} already has a {phase} backend")
                layers[key] = backend
            name = backend.get_registration_name()
            if name in members or hasattr(self, name):
                raise ValueError(f"attention backend {name} is already registered")
            members[name] = backend
            phases[phase].append(backend)

        for name, backend in members.items():
            setattr(self, name, backend)
        self._backends_by_layer.update(layers)
        self._backends_by_phase[role] = {phase: tuple(impls) for phase, impls in phases.items()}

    # ========================================================================
    # 二、后端和能力查询
    # ========================================================================

    def get_backends(self, role: str, phase: str) -> tuple[AttentionBackend, ...]:
        """按注册顺序返回该角色在指定阶段的后端实例元组。

        参数
            role 是要查找的后端角色键，取自目标模型的 model.att_state_role。
            主模型用 "main"，草稿模型用 "draft0"、"draft1"、"draft2" 等，最大编号为 "draft35"。
            查询时须使用目标模型调用 register_backends 时传入的完整键。
            phase 为 "prefill" 或 "decode"，决定返回哪类后端。

        指定阶段没有后端或 phase 不在有效范围内时返回空元组。
        本管理器尚未通过 register_backends 登记该键的后端清单时抛出 RuntimeError。
        """
        phases = self._backends_by_phase.get(role)
        if phases is None:
            raise RuntimeError(f"attention backend registration must finish before querying backends for {role}")
        return phases.get(phase, ())

    def get_backend(self, role: str, phase: str, logical_layer_id: int) -> AttentionBackend | None:
        """返回服务指定逻辑层的后端实例。

        参数
            role 是登记目标模型后端时使用的角色键，即该模型的 att_state_role。
            phase 为 "prefill" 或 "decode"。
            logical_layer_id 是模型层的全局逻辑层号，对应 layer.layer_num_。
            可命中的值来自该角色、该阶段所登记的 logical_layer_ids，层号可从非零值开始或不连续。

        指定层在此阶段未登记后端或 phase 不在有效范围内时返回 None。
        本管理器尚未登记该键的后端清单时抛出 RuntimeError。
        """
        if role not in self._backends_by_phase:
            raise RuntimeError(f"attention backend registration must finish before querying backends for {role}")
        return self._backends_by_layer.get((role, phase, logical_layer_id))

    def has_attention(self, role: str) -> bool:
        """判断该角色是否登记了至少一个 prefill 或 decode 后端。

        role 是目标模型的后端角色键，取自 model.att_state_role。
        该键尚未登记或登记的后端清单为空时返回 False。
        """
        return any(self._backends_by_phase.get(role, {}).values())

    def requires_cpu_cache_sync(self, role: str) -> bool:
        """汇总指定模型角色的 CPU cache 拷贝同步需求。

        role 是登记目标模型后端时使用的角色键，取自 model.att_state_role。
        任一阶段的后端要求同步时返回 True，两个阶段均无同步需求时返回 False。
        本管理器尚未登记该键的后端清单时抛出 RuntimeError。
        """
        return any(
            impl.requires_cpu_cache_sync()
            for phase in ("prefill", "decode")
            for impl in self.get_backends(role=role, phase=phase)
        )

    # ========================================================================
    # 三、当前微批次的状态准备
    # ========================================================================

    def create_att_state(self, infer_state: InferStateInfo) -> dict[AttentionBackend, Any]:
        """为本次模型前向的当前微批次准备后端状态。

        infer_state 是模型已填充请求索引、序列长度和本步执行信息的 InferStateInfo。
        att_state_role 是本步模型登记后端时使用的角色键。is_prefill 为 True 时准备 prefill，
        为 False 时准备 decode。各后端按需读取 b_req_idx、b_seq_len、microbatch_index、
        graph_mode 和本阶段的其他请求字段。后端从自身运行时配置读取 Graph 容量和解码布局，
        在 create_att_state 中准备本步所需的派生参数。

        返回以本阶段的后端实例为键、对应的单步状态为值的字典。每个后端调用一次。
        当前阶段没有后端时返回空字典。本管理器尚未登记该键的后端清单时抛出 RuntimeError。
        """
        phase = "prefill" if infer_state.is_prefill else "decode"
        return {
            impl: impl.create_att_state(infer_state=infer_state)
            for impl in self.get_backends(role=infer_state.att_state_role, phase=phase)
        }
