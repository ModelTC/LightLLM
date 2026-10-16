"""Bind optional EP services at initialization, outside the inference loop."""

from functools import wraps

import torch.distributed as dist
from lightllm.common.eplb_utils import is_eplb_enabled

from .ep_balance_monitor import EPBalanceMonitor, should_enable_ep_balance_monitor


def install_ep_runtime(backend):
    """Install on the target model only, before either inference thread starts."""
    manager = None
    if is_eplb_enabled(backend.args):
        from .eplb_manager import EPLBManager

        manager = EPLBManager(backend.model)
        if not backend.args.enable_prefill_microbatch_overlap and not backend.args.enable_tpsp_mix_mode:
            # These MoE calls are globally ordered under the forward event.
            # Commit ready layers here, but finish the generation only at
            # the outer forward boundary.
            for state in manager._eplb_states:
                state.migration_poll = manager.poll_migration_at_layer_boundary

    monitor = EPBalanceMonitor(backend.model) if should_enable_ep_balance_monitor(backend.args) else None
    if monitor is not None and not monitor.enabled:
        monitor = None

    if manager is None and monitor is None:
        return

    def record_prefill(prefill):
        @wraps(prefill)
        def run(*args, **kwargs):
            if manager is not None:
                if backend.args.run_mode == "normal":
                    # Run control collectives and publish weights only while
                    # owning a prefill forward, never on decode/idle rounds.
                    manager.poll()
                manager.before_prefill()
            output = prefill(*args, **kwargs)
            if monitor is not None:
                monitor.record_prefill_round()
            if manager is not None:
                manager.step()
            return output

        return run

    # These are the two completed-prefill boundaries; decode and draft models
    # retain their original methods. Failed forwards never publish a sample.
    model = backend.model
    model._prefill = record_prefill(model._prefill)
    model._microbatch_overlap_prefill_cuda = record_prefill(model._microbatch_overlap_prefill_cuda)

    if manager is not None:
        dist.barrier()
        if backend.args.run_mode == "normal":
            return
        get_event_pack = backend.overlap_event_manager.get_overlap_event_pack

        @wraps(get_event_pack)
        def get_overlap_event_pack():
            event_pack = get_event_pack()
            # Capture the event rather than a bound pack method: storing the
            # latter back on the pack would create a cycle on every batch.
            wait_to_forward = event_pack.wait_forward_event.wait

            def wait_and_poll():
                wait_to_forward()
                # The existing event serializes both inference threads. Poll
                # here even on idle rounds, before request/DP collectives.
                manager.poll()

            event_pack.wait_to_forward = wait_and_poll
            return event_pack

        backend.overlap_event_manager.get_overlap_event_pack = get_overlap_event_pack
