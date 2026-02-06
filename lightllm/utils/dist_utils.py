import torch.distributed as dist
import os
import torch
import requests

# 规范 rank 的含义，在 llm 推理的相关代码中下述的 rank 的含义如下：
# global_rank 全局 rank 序列id， 如两节点 8卡，会存在 0 - 15 16个global_rank
# global_world_size 全局的 world size 大小， 如两节点 8 卡，该值为 16
# dp_size 如果部署形态是一个推理实列包含几个数据并行的推理实列，则 dp size 为整个系统中的 dp 并行数量
# dp_world_size 每一个dp 数据并行占用的卡数
# global_dp_rank 指每个dp 数据并行在整个推理实列中dp的rank号， 如果 16卡部署，4 dp size， 则存在 0 - 3 4个global_dp_rank
# 值，其中 0-3号卡为 global_dp_rank 0, 4-8 为 global_dp_rank 1， 9-12 为global_dp_rank 2, 13-15为dglobal_dp_rank 3
# rank_in_dp 指在一个dp内的rank序号。
# node_world_size 指一个推理节点的使用的卡数，如两机 tp 推理，如果两机器8卡，则 node_world_size 为 8.
# rank_in_node 指在一个node内的rank序号，如两机8卡推理，每机上的rank序号都是0-8
# dp_rank_in_node 指的是在一个node节点中存在的dp_rank的序号，如果两机8卡，每4卡一个dp，则存在两个node，
# 其中 node 0 上，存在两个dp，其 dp_rank_in_node 分别为 0， 1 在 node 1 上， 也存在两个dp，其 dp_rank_in_node
# 也分别为 0， 1, 在一个node内的操作，几乎大部分都是使用 dp_rank_in_node 信息。

# 下面是一个 2 node， 16卡， dp_size 4 的 rank 信息结构图：
# +------+--------------+----------------+-------------+--------------+------------------+
# | card | global_rank  | global_dp_rank | rank_in_dp  | rank_in_node | dp_rank_in_node  |
# +------+--------------+----------------+-------------+--------------+------------------+
# |  0   |      0       |        0       |      0      |      0       |         0        |
# |  1   |      1       |        0       |      1      |      1       |         0        |
# |  2   |      2       |        0       |      2      |      2       |         0        |
# |  3   |      3       |        0       |      3      |      3       |         0        |
# +------+--------------+----------------+-------------+--------------+------------------+
# |  4   |      4       |        1       |      0      |      4       |         1        |
# |  5   |      5       |        1       |      1      |      5       |         1        |
# |  6   |      6       |        1       |      2      |      6       |         1        |
# |  7   |      7       |        1       |      3      |      7       |         1        |
# +------+--------------+----------------+-------------+--------------+------------------+
# |  8   |      8       |        2       |      0      |      0       |         0        |
# |  9   |      9       |        2       |      1      |      1       |         0        |
# | 10   |     10       |        2       |      2      |      2       |         0        |
# | 11   |     11       |        2       |      3      |      3       |         0        |
# +------+--------------+----------------+-------------+--------------+------------------+
# | 12   |     12       |        3       |      0      |      4       |         1        |
# | 13   |     13       |        3       |      1      |      5       |         1        |
# | 14   |     14       |        3       |      2      |      6       |         1        |
# | 15   |     15       |        3       |      3      |      7       |         1        |
# +------+--------------+----------------+-------------+--------------+------------------+


def set_environ(environ_name, value):
    os.environ[environ_name] = str(value)


def get_environ(environ_name):
    value = os.getenv(environ_name, None)
    if value is None:
        raise RuntimeError(f"{environ_name} is not set")
    return value


def init_vision_distributed_env(kvargs):
    tp_world_size = kvargs["vit_tp"]
    dp_size = 1
    tp_rank_id = kvargs["tp_rank_id"]
    set_dp_size(dp_size)
    set_dp_world_size(tp_world_size)
    set_current_rank_in_dp(tp_rank_id)
    visual_gpu_ids = kvargs["visual_gpu_ids"]
    device_id = visual_gpu_ids[kvargs["vit_rank_id"]]
    set_current_device_id(device_id)
    torch.cuda.set_device(device_id)
    # 不要在init_process_group时，显示的传入device_id
    # 这会触发torch的device-bound split优化，会默认后面想加入新进程组的rank
    # 都已经存在于默认组，这样RL更新weight的init_group时，外部想加入的组，在执行
    # 通信原语时例如all_reduce，会永远等不到LightLLM默认组里的回复，从而导致错误结果。
    dist.init_process_group(
        "nccl",
        init_method=f'tcp://127.0.0.1:{kvargs["visual_nccl_port"]}',
        rank=kvargs["tp_rank_id"],
        world_size=tp_world_size,
    )
    # warmup nccl communicator
    _a = torch.zeros([1]).to(f"cuda:{device_id}")
    dist.all_reduce(_a)
    del _a


def init_distributed_env(kvargs):
    assert kvargs["world_size"] % kvargs["args"].nnodes == 0, "world_size should be divided by nnodes"
    node_world_size = kvargs["world_size"] // kvargs["args"].nnodes

    set_global_rank(kvargs["rank_id"])
    set_global_world_size(kvargs["world_size"])
    set_dp_size(kvargs.get("dp_size", 1))
    set_dp_world_size(get_global_world_size() // get_dp_size())
    set_global_dp_rank(get_global_rank() // get_dp_world_size())
    set_current_rank_in_dp(get_global_rank() % get_dp_world_size())
    set_current_rank_in_node(get_global_rank() % node_world_size)
    set_node_world_size(node_world_size)

    nnodes = kvargs["args"].nnodes
    dp_size_in_node = max(1, get_dp_size() // nnodes)
    set_dp_rank_in_node(get_global_dp_rank() % dp_size_in_node)

    _init_nccl_env()
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


def set_global_dp_rank(rank: int):
    set_environ("LIGHTLLM_GLOBAL_DP_RANK", rank)


def get_global_dp_rank():
    return int(get_environ("LIGHTLLM_GLOBAL_DP_RANK"))


def set_dp_rank_in_node(rank: int):
    set_environ("LIGHTLLM_DP_RANK_IN_NODE", rank)


def get_dp_rank_in_node():
    return int(get_environ("LIGHTLLM_DP_RANK_IN_NODE"))


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


def create_new_group_for_current_dp(backend):
    ans_group = None
    for iter_dp_rank in range(get_dp_size()):
        ranks = list(i + iter_dp_rank * get_dp_world_size() for i in range(get_dp_world_size()))
        device_group = dist.new_group(ranks, backend=backend)
        if get_global_dp_rank() == iter_dp_rank:
            ans_group = device_group
    return ans_group


def create_new_group_for_current_node(backend):
    from lightllm.utils.envs_utils import get_env_start_args

    args = get_env_start_args()
    ans_group = None
    nnodes = args.nnodes
    node_world_size = args.tp // nnodes
    for iter_node_rank in range(nnodes):
        ranks = list(i + iter_node_rank * node_world_size for i in range(node_world_size))
        device_group = dist.new_group(ranks, backend=backend)
        if args.node_rank == iter_node_rank:
            ans_group = device_group
    return ans_group


def create_dp_special_inter_group(backend):
    """
    创建一种特殊的通信组
    假设全局通信组为 [0, 1, 2, 3, 4, 5, 6, 7], 其中
    0,1,2,3 为一个dp, 4,5,6,7 为另一个 dp, 则在[0,4],
    [1,5], [2,6], [3,7] 间建立通信组
    """
    from lightllm.utils.envs_utils import get_env_start_args

    args = get_env_start_args()
    ans_group = None
    dp_size = args.dp
    dp_world_size = get_dp_world_size()
    rank = get_global_rank()
    for iter_tp_rank in range(dp_world_size):
        ranks = list(iter_tp_rank + i * dp_world_size for i in range(dp_size))
        device_group = dist.new_group(ranks, backend=backend)
        if rank in ranks:
            ans_group = device_group
    return ans_group


def _init_nccl_env():
    from lightllm.utils.envs_utils import get_env_start_args

    args = get_env_start_args()

    # 配置使用外部的 tcp store server 来创建 nccl 连接
    if args.use_config_server_to_init_nccl:
        os.environ["TORCHELASTIC_USE_AGENT_STORE"] = "True"
        rank_id = get_global_rank()
        world_size = get_global_world_size()
        ip_port = f"{args.config_server_host}:{args.config_server_port}"
        params = f"tcp_store_port={args.nccl_port}&&rank_id={rank_id}&&world_size={world_size}"

        if rank_id == 0:
            # 当使用外部config server 启动的tcpStore来初始化nccl时，需要保证配置了config_server_host.
            # 同时也需要保证config_server_host和nccl_host是同一个ip, 这个时候 rank 0 推理进程会先调用
            # config server的http接口来启动tcp store server, 然后再调用nccl init方法来初始化nccl.
            assert args.config_server_host == args.nccl_host
            url = f"http://{ip_port}/start_tcp_store_server?{params}"
            response = requests.get(url, timeout=60 * 3)
            assert response.status_code == 200, f"Failed to init config server nccl tcp store: {response.status_code}"
        else:
            assert args.config_server_host == args.nccl_host
            url = f"http://{ip_port}/start_tcp_store_server?{params}"
            response = requests.get(url, timeout=60 * 3)
            assert response.status_code == 200, f"Failed to init config server nccl tcp store: {response.status_code}"

    return


# copy from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/utils/common.py#L1675
def init_custom_process_group(
    backend=None,
    init_method=None,
    timeout=None,
    world_size=-1,
    rank=-1,
    store=None,
    group_name=None,
    pg_options=None,
    device_id=None,
):
    from torch.distributed.distributed_c10d import (
        Backend,
        PrefixStore,
        _new_process_group_helper,
        _world,
        default_pg_timeout,
        rendezvous,
    )

    assert (store is None) or (init_method is None), "Cannot specify both init_method and store."

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("undefined")

    if timeout is None:
        timeout = default_pg_timeout

    # backward compatible API
    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)

        # Use a PrefixStore to avoid accidental overrides of keys used by
        # different systems (e.g. RPC) in case the store is multi-tenant.
        store = PrefixStore(group_name, store)

    # NOTE: The pg_options parameter was renamed into backend_options in PyTorch 2.6.0
    # https://github.com/pytorch/pytorch/commit/a0c7029a75628cd5fa8df83c0de0ea98ee7fd844
    # We need to determine the appropriate parameter name based on PyTorch version
    pg_options_param_name = "backend_options" if str(torch.__version__) >= "2.6" else "pg_options"
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        **{pg_options_param_name: pg_options},
        timeout=timeout,
        device_id=device_id,
    )

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}

    return pg
