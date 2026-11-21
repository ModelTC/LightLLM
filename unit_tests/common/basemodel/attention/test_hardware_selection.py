from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _select_on_rank(rank: int, rendezvous: str) -> None:
    from lightllm.common.basemodel.attention import get_prefill_att_backend_class
    from lightllm.server.core.objs.start_args_type import StartArgs
    from lightllm.utils import dist_utils
    from lightllm.utils.envs_utils import set_env_start_args

    torch.cuda.set_device(rank)
    dist_utils.set_current_device_id(rank)
    dist_utils.set_current_rank_in_dp(rank)
    dist_utils.set_dp_world_size(2)
    args = StartArgs()
    args.page_size = 16
    args.llm_kv_type = "None"
    args.llm_prefill_att_backend = ["triton"]
    set_env_start_args(args)
    model = SimpleNamespace(
        req_manager=SimpleNamespace(req_to_token_indexs=torch.empty(0, device=rank)),
        args=args,
        config={"num_attention_heads": 8},
        tp_world_size_=2,
        tp_k_head_num_=2,
        head_dim_=64,
        data_type=torch.float16,
    )
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2)
    try:
        assert get_prefill_att_backend_class(model=model).__name__ == "TritonAttBackend"
        # Rank 0 already cached success; rank 1 cannot run this head dimension.
        if rank == 1:
            model.head_dim_ = 15
        with pytest.raises(RuntimeError, match="gqa prefill.*triton"):
            get_prefill_att_backend_class(model=model)
        model.head_dim_ = 64
        assert get_prefill_att_backend_class(model=model).__name__ == "TritonAttBackend"
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
def test_select_on_every_rank(tmp_path):
    mp.spawn(_select_on_rank, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True)
