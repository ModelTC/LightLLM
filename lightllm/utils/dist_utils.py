import torch.distributed as dist
import os
import torch

# 规范 rank 的含义，在 llm 推理的相关代码中下述的 rank 的含义如下：
# global_rank 全局 rank 序列id， 如两节点 8卡，会存在 0 - 15 16个global_rank
# global_world_size 全局的 world size 大小， 如两节点 8 卡，该值为 16
# dp_size 如果部署形态是一个推理实列包含几个数据并行的推理实列，则 dp size 为整个系统中的 dp 并行数量
# dp_world_size 每一个dp 数据并行占用的卡数
# dp_rank 指每个dp 数据并行在整个推理实列中dp的rank号， 如果 16卡部署，4 dp size， 则存在 0 - 3 4个dp_rank
# 值，其中 0-3号卡为 dp_rank 0, 4-8 为 dp_rank 1， 9-12 为dp_rank 2, 13-15为dp_rank 3
# rank_in_dp 指在一个dp内的rank序号。
# node_world_size 指一个推理节点的使用的卡数，如两机 tp 推理，如果两机器8卡，则 node_world_size 为 8.
# rank_in_node 指在一个node内的rank序号，如两机8卡推理，每机上的rank序号都是0-8


def set_environ(environ_name, value):
    os.environ[environ_name] = str(value)


def get_environ(environ_name):
    value = os.getenv(environ_name, None)
    if value is None:
        raise RuntimeError(f"{environ_name} is not set")
    return value


def _init_vision_distributed_env(kvargs):
    world_size = kvargs["vit_tp"]
    set_global_rank(kvargs["tp_rank_id"])
    set_global_world_size(world_size)
    visual_gpu_ids = kvargs["visual_gpu_ids"]
    device_id = visual_gpu_ids[kvargs["vit_rank_id"]]
    set_current_device_id(device_id)
    torch.cuda.set_device(device_id)
    dist.init_process_group(
        "nccl",
        init_method=f'tcp://127.0.0.1:{kvargs["visual_nccl_port"]}',
        rank=kvargs["tp_rank_id"],
        world_size=world_size,
    )
    # warmup nccl communicator
    _a = torch.zeros([1]).to(f"cuda:{device_id}")
    dist.all_reduce(_a)
    del _a


def _init_distributed_env(kvargs):
    assert kvargs["world_size"] % kvargs["args"].nnodes == 0, "world_size should be divided by nnodes"
    node_world_size = kvargs["world_size"] // kvargs["args"].nnodes

    set_global_rank(kvargs["rank_id"])
    set_global_world_size(kvargs["world_size"])
    set_dp_size(kvargs.get("dp_size", 1))
    set_dp_world_size(get_global_world_size() // get_dp_size())
    set_current_dp_rank(get_global_rank() // get_dp_world_size())
    set_current_rank_in_dp(get_global_rank() % get_dp_world_size())
    set_current_rank_in_node(get_global_rank() % node_world_size)
    set_node_world_size(node_world_size)

    device_id = kvargs["rank_id"] % get_node_world_size()
    set_current_device_id(device_id)
    torch.cuda.set_device(device_id)
    dist.init_process_group(
        "nccl",
        init_method=f'tcp://{kvargs["nccl_host"]}:{kvargs["nccl_port"]}',
        rank=kvargs["rank_id"],
        world_size=kvargs["world_size"],
    )
    # warmup nccl communicator
    _a = torch.zeros([1]).to(f"cuda:{device_id}")
    dist.all_reduce(_a)
    del _a


def set_global_rank(global_rank: int):
    set_environ("LIGHTLLM_GLOBAL_RANK", global_rank)


def get_global_rank():
    return int(get_environ("LIGHTLLM_GLOBAL_RANK"))


def set_global_world_size(world_size: int):
    set_environ("LIGHTLLM_GLOBAL_WORLD_SIZE", world_size)


def get_global_world_size():
    return int(get_environ("LIGHTLLM_GLOBAL_WORLD_SIZE"))


def set_dp_size(dp_size: int):
    """
    total dp num
    """
    set_environ("LIGHTLLM_DP_SIZE", dp_size)


def get_dp_size():
    return int(get_environ("LIGHTLLM_DP_SIZE"))


def set_dp_world_size(world_size: int):
    set_environ("LIGHTLLM_DP_WORLD_SIZE", world_size)


def get_dp_world_size():
    return int(get_environ("LIGHTLLM_DP_WORLD_SIZE"))


def set_current_dp_rank(rank: int):
    set_environ("LIGHTLLM_CURRENT_DP_RANK", rank)


def get_current_dp_rank():
    return int(get_environ("LIGHTLLM_CURRENT_DP_RANK"))


def set_current_rank_in_dp(rank: int):
    set_environ("LIGHTLLM_CURRENT_RANK_IN_DP", rank)


def get_current_rank_in_dp():
    return int(get_environ("LIGHTLLM_CURRENT_RANK_IN_DP"))


def set_current_device_id(device_id: int):
    set_environ("LIGHTLLM_CURRENT_DEVICE_ID", device_id)


def get_current_device_id():
    return int(get_environ("LIGHTLLM_CURRENT_DEVICE_ID"))


def set_current_rank_in_node(rank: int):
    set_environ("LIGHTLLM_CURRENT_RANK_IN_NODE", rank)


def get_current_rank_in_node():
    return int(get_environ("LIGHTLLM_CURRENT_RANK_IN_NODE"))


def set_node_world_size(node_world_size: int):
    set_environ("LIGHTLLM_NODE_WORLD_SIZE", node_world_size)


def get_node_world_size():
    return int(get_environ("LIGHTLLM_NODE_WORLD_SIZE"))
