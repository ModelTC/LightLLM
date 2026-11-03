import threading
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _shutdown_rank(rank, rendezvous):
    from lightllm.server.router.model_infer.mode_backend import base_backend as base
    from lightllm.server.router.model_infer.mode_backend.dp_backend import impl
    from lightllm.server.router.model_infer.mode_backend.overlap_events import OverlapEventManager

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    base.all_reduce = dist.all_reduce
    base.dist_group_manager.dp_control_group = None
    impl.get_current_device_id = lambda: 0
    torch.cuda.set_device = lambda _: None
    backend = impl.DPChunkedPrefillBackend.__new__(impl.DPChunkedPrefillBackend)
    backend.shutdown_requested = threading.Event()
    backend.inference_stopped = threading.Event()
    backend.dp_control_tensor = torch.zeros(3, dtype=torch.int32)
    backend.overlap_event_manager = OverlapEventManager()
    backend.support_overlap = False
    backend.model = SimpleNamespace(eplb_manager=None)
    backend._try_read_new_reqs = lambda: None
    backend._get_classed_reqs = lambda **kwargs: ([], [], 0)
    backend.classed_req_no_decode = False
    backend.classed_req_strict_prefill = False
    backend.control_state_machine = SimpleNamespace(try_recover_paused_reqs=lambda: False)
    if rank == 0:
        backend.shutdown_requested.set()
    errors = []

    def loop():
        try:
            backend.infer_loop()
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=loop, daemon=True) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
    assert not errors, errors
    assert all(not thread.is_alive() for thread in threads)
    assert backend.inference_stopped.is_set()
    assert backend.dp_control_tensor.tolist() == [0, 0, 1]
    dist.destroy_process_group()


def test_shutdown_reaches_other_ranks_and_releases_both_inference_threads(tmp_path):
    mp.spawn(_shutdown_rank, args=(f"file://{tmp_path / 'gloo'}",), nprocs=2, join=True)
