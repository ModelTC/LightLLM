from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _run_control_rank(rank, world_size, rendezvous):
    from lightllm.server.router.model_infer.mode_backend import base_backend

    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world_size, timeout=timedelta(seconds=30)
    )
    try:
        base_backend.dist_group_manager.dp_control_group = dist.group.WORLD
        backend = base_backend.ModeBackend.__new__(base_backend.ModeBackend)
        backend.dp_control_tensor = torch.zeros(2, dtype=torch.int32)
        for owner, expected in [(0, (True, False)), (1, (False, True)), (-1, (False, False))]:
            prefill = [object()] if owner == 0 and rank == 0 else []
            decode = [object()] if owner == 1 and rank == 1 else []
            assert backend._dp_all_reduce_req_presence(prefill, decode) == expected

        # Simulate two nodes with distinct broadcast sources, then reconverge globally.
        groups = [dist.new_group([index], backend="gloo") for index in range(world_size)]
        backend.args = SimpleNamespace(node_rank=rank)
        backend.node_world_size = 1
        backend.node_gloo_group = groups[rank]
        backend.is_master_in_node = True
        backend.is_pd_mode = False
        backend.node_broadcast_tensor = torch.zeros(1, dtype=torch.int32)
        backend.shm_reqs_io_buffer = SimpleNamespace(is_ready=lambda: rank == 1)
        reads = []
        backend._read_reqs_buffer_and_init_reqs = lambda: reads.append(True)
        backend._try_read_new_reqs_normal()
        assert bool(reads) == (rank == 1)
        assert backend._dp_all_reduce_req_presence([], [object()] if rank == 1 else []) == (False, True)
    finally:
        dist.destroy_process_group()


def test_gloo_control_handles_empty_ranks_and_node_local_sources(tmp_path):
    mp.spawn(_run_control_rank, args=(2, str(tmp_path / "rendezvous")), nprocs=2, join=True)
