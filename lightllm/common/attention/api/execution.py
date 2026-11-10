from dataclasses import dataclass

from .spec import AttentionPhase


@dataclass(frozen=True)
class ExecutionRequirements:
    """启动时确定的执行约束，与 LayerSpec 的计算语义分开保存。"""

    graph_phases: frozenset[AttentionPhase] = frozenset()
    microbatch_count: int = 1
    max_batch_size: int = 1
    max_seq_len: int = 1
    max_verify_tokens: int = 1
    tp_world_size: int = 1
    expert_parallel: bool = False
    transfer_paths: frozenset[str] = frozenset()

    @classmethod
    def from_model(cls, model):
        args = model.args
        phases = set()
        if not args.disable_cudagraph:
            phases.add(AttentionPhase.DECODE)
        if args.enable_prefill_cudagraph:
            phases.add(AttentionPhase.PREFILL)
        transfers = set()
        if args.enable_cpu_cache:
            transfers.add("cpu")
        if args.enable_dp_prompt_cache_fetch:
            transfers.add("dp")
        if args.run_mode in ("prefill", "decode"):
            transfers.add("pd")
        return cls(
            frozenset(phases),
            2 if args.enable_decode_microbatch_overlap or args.enable_prefill_microbatch_overlap else 1,
            getattr(model, "max_req_num", 1),
            max(getattr(model, "max_seq_length", 1), getattr(model, "graph_max_len_in_batch", 1)),
            model.mtp_manager.get_decode_draft_step(model.is_mtp_draft_model) + 1,
            model.tp_world_size_,
            args.enable_ep_moe,
            frozenset(transfers),
        )


@dataclass(frozen=True)
class BackendResources:
    """捕获前的资源上界；workspace 项为 (名称, 元素数, dtype)。

    同一执行槽位按名称取最大容量，并发槽位分别分配。主机长度快照只在需要
    它的实现中创建；持久存储的初始化由 State 执行，不能在 forward 中反复清零。
    """

    workspace: tuple = ()
    needs_host_lengths: bool = False
    zero_initialize_cache: bool = False
