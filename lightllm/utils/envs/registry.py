"""
LightLLM 环境变量注册表

集中定义所有环境变量，按功能分组管理。

使用方式：
    from lightllm.utils.envs import env

    # 获取值
    log_level = env.logging.LOG_LEVEL.get()

    # 设置值
    env.logging.LOG_LEVEL.set("debug")

    # 验证
    errors = env.manager.validate_all()
"""

from lightllm.utils.envs.core import EnvVar, EnvGroup, EnvManager, EnvVarType


# ============================================================================
# 创建全局管理器
# ============================================================================
manager = EnvManager.get_instance()


# ============================================================================
# 日志组
# ============================================================================
logging = EnvGroup("logging", "日志和调试相关配置")

logging.LOG_LEVEL = logging.add(
    EnvVar(
        name="LIGHTLLM_LOG_LEVEL",
        var_type=EnvVarType.STRING,
        default="info",
        description="日志级别",
        choices=["debug", "info", "warning", "error", "critical"],
    )
)

logging.LOG_DIR = logging.add(
    EnvVar(
        name="LIGHTLLM_LOG_DIR",
        var_type=EnvVarType.STRING,
        default=None,
        description="日志输出目录，None 表示输出到控制台",
    )
)

manager.register_group(logging)


# ============================================================================
# 服务配置组
# ============================================================================
server = EnvGroup("server", "HTTP服务器和网络相关配置")

server.UNIQUE_SERVICE_NAME_ID = server.add(
    EnvVar(
        name="LIGHTLLM_UNIQUE_SERVICE_NAME_ID",
        var_type=EnvVarType.STRING,
        default=None,
        description="服务唯一标识符",
    )
)

server.GUNICORN_TIMEOUT = server.add(
    EnvVar(
        name="LIGHTLLM_GUNICORN_TIMEOUT",
        var_type=EnvVarType.INT,
        default=180,
        description="Gunicorn 请求超时时间",
        unit="秒",
        min_value=1,
        # 向后兼容旧的拼写错误
        aliases=["LIGHTLMM_GUNICORN_TIME_OUT"],
    )
)

server.GUNICORN_KEEP_ALIVE = server.add(
    EnvVar(
        name="LIGHTLLM_GUNICORN_KEEP_ALIVE",
        var_type=EnvVarType.INT,
        default=10,
        description="Gunicorn 保活时间",
        unit="秒",
        min_value=1,
        aliases=["LIGHTLMM_GUNICORN_KEEP_ALIVE"],
    )
)

server.WEBSOCKET_MAX_SIZE = server.add(
    EnvVar(
        name="LIGHTLLM_WEBSOCKET_MAX_SIZE",
        var_type=EnvVarType.INT,
        default=16 * 1024 * 1024,
        description="WebSocket 最大消息大小",
        unit="字节",
        min_value=1024,
    )
)

server.REQUEST_TIMEOUT = server.add(
    EnvVar(
        name="LIGHTLLM_REQUEST_TIMEOUT",
        var_type=EnvVarType.INT,
        default=5,
        description="请求超时时间",
        unit="秒",
        aliases=["REQUEST_TIMEOUT"],
    )
)

server.REQUEST_PROXY = server.add(
    EnvVar(
        name="LIGHTLLM_REQUEST_PROXY",
        var_type=EnvVarType.STRING,
        default=None,
        description="HTTP 请求代理地址",
        aliases=["REQUEST_PROXY"],
    )
)

manager.register_group(server)


# ============================================================================
# 内存和缓冲区组
# ============================================================================
memory = EnvGroup("memory", "内存管理和缓冲区配置")

memory.REQS_BUFFER_BYTE_SIZE = memory.add(
    EnvVar(
        name="LIGHTLLM_REQS_BUFFER_BYTE_SIZE",
        var_type=EnvVarType.INT,
        default=64 * 1024 * 1024,
        description="请求缓冲区大小",
        unit="字节",
        min_value=1024 * 1024,
    )
)

memory.RPC_BYTE_SIZE = memory.add(
    EnvVar(
        name="LIGHTLLM_RPC_BYTE_SIZE",
        var_type=EnvVarType.INT,
        default=16 * 1024 * 1024,
        description="RPC 缓冲区大小",
        unit="字节",
        min_value=1024 * 1024,
    )
)

memory.RPC_RESULT_BYTE_SIZE = memory.add(
    EnvVar(
        name="LIGHTLLM_RPC_RESULT_BYTE_SIZE",
        var_type=EnvVarType.INT,
        default=1 * 1024 * 1024,
        description="RPC 结果缓冲区大小",
        unit="字节",
        min_value=1024,
    )
)

memory.OUT_TOKEN_QUEUE_SIZE = memory.add(
    EnvVar(
        name="LIGHTLLM_OUT_TOKEN_QUEUE_SIZE",
        var_type=EnvVarType.INT,
        default=8,
        description="输出 Token 队列大小",
        min_value=1,
    )
)

memory.TOKEN_HASH_LIST_SIZE = memory.add(
    EnvVar(
        name="LIGHTLLM_TOKEN_HASH_LIST_SIZE",
        var_type=EnvVarType.INT,
        default=2048,
        description="Token 哈希列表大小",
        min_value=128,
    )
)

memory.TOKEN_MAX_BYTES = memory.add(
    EnvVar(
        name="LIGHTLLM_TOKEN_MAX_BYTES",
        var_type=EnvVarType.INT,
        default=1280,
        description="单个 Token 最大字节数",
        unit="字节",
        min_value=128,
    )
)

memory.NIXL_PARAM_OBJ_MAX_BYTES = memory.add(
    EnvVar(
        name="LIGHTLLM_NIXL_PARAM_OBJ_MAX_BYTES",
        var_type=EnvVarType.INT,
        default=8 * 1024,
        description="NIXL 参数对象最大字节数",
        unit="字节",
        min_value=1024,
    )
)

memory.DECODE_PREFIX_LENGTH = memory.add(
    EnvVar(
        name="LIGHTLLM_DECODE_PREFIX_LENGTH",
        var_type=EnvVarType.INT,
        default=5,
        description="解码前缀长度",
        min_value=1,
    )
)

memory.HUGE_PAGE_ENABLE = memory.add(
    EnvVar(
        name="LIGHTLLM_HUGE_PAGE_ENABLE",
        var_type=EnvVarType.BOOL,
        default=False,
        description="启用大页内存模式，可大幅缩短 CPU KV cache 加载时间",
    )
)

manager.register_group(memory)


# ============================================================================
# 采样参数组
# ============================================================================
sampling = EnvGroup("sampling", "采样参数限制和配置")

sampling.STOP_SEQUENCE_MAX_LENGTH = sampling.add(
    EnvVar(
        name="LIGHTLLM_STOP_SEQUENCE_MAX_LENGTH",
        var_type=EnvVarType.INT,
        default=256,
        description="停止序列最大长度",
        min_value=1,
    )
)

sampling.STOP_SEQUENCE_STR_MAX_LENGTH = sampling.add(
    EnvVar(
        name="LIGHTLLM_STOP_SEQUENCE_STR_MAX_LENGTH",
        var_type=EnvVarType.INT,
        default=256,
        description="停止序列字符串最大长度",
        min_value=1,
    )
)

sampling.ALLOWED_TOKEN_IDS_MAX_LENGTH = sampling.add(
    EnvVar(
        name="LIGHTLLM_ALLOWED_TOKEN_IDS_MAX_LENGTH",
        var_type=EnvVarType.INT,
        default=256,
        description="允许的 Token IDs 最大长度",
        min_value=1,
    )
)

sampling.MAX_STOP_SEQUENCES = sampling.add(
    EnvVar(
        name="LIGHTLLM_MAX_STOP_SEQUENCES",
        var_type=EnvVarType.INT,
        default=10,
        description="最大停止序列数",
        min_value=1,
    )
)

sampling.REGULAR_CONSTRAINT_MAX_LENGTH = sampling.add(
    EnvVar(
        name="LIGHTLLM_REGULAR_CONSTRAINT_MAX_LENGTH",
        var_type=EnvVarType.INT,
        default=2048,
        description="正则表达式约束最大长度",
        min_value=1,
    )
)

sampling.GRAMMAR_CONSTRAINT_MAX_LENGTH = sampling.add(
    EnvVar(
        name="LIGHTLLM_GRAMMAR_CONSTRAINT_MAX_LENGTH",
        var_type=EnvVarType.INT,
        default=2048,
        description="语法约束最大长度",
        min_value=1,
    )
)

sampling.JSON_SCHEMA_MAX_LENGTH = sampling.add(
    EnvVar(
        name="LIGHTLLM_JSON_SCHEMA_MAX_LENGTH",
        var_type=EnvVarType.INT,
        default=2048,
        description="JSON Schema 最大长度",
        min_value=1,
    )
)

sampling.INPUT_PENALTY = sampling.add(
    EnvVar(
        name="LIGHTLLM_INPUT_PENALTY",
        var_type=EnvVarType.BOOL,
        default=False,
        description="是否启用输入惩罚",
        aliases=["INPUT_PENALTY"],
    )
)

sampling.SKIP_SPECIAL_TOKENS = sampling.add(
    EnvVar(
        name="LIGHTLLM_SKIP_SPECIAL_TOKENS",
        var_type=EnvVarType.BOOL,
        default=True,
        description="是否跳过特殊 Token",
        aliases=["SKIP_SPECIAL_TOKENS"],
    )
)

manager.register_group(sampling)


# ============================================================================
# KV 缓存和量化组
# ============================================================================
kv_cache = EnvGroup("kv_cache", "KV 缓存和量化相关配置")

kv_cache.QUANT_CALIBRATION_WARMUP_COUNT = kv_cache.add(
    EnvVar(
        name="LIGHTLLM_KV_QUANT_CALIBRATION_WARMUP_COUNT",
        var_type=EnvVarType.INT,
        default=0,
        description="KV 量化校准预热次数，服务启动后前 N 次推理不计入校准统计",
        min_value=0,
        # 向后兼容拼写错误
        aliases=["LIGHTLLM_KV_QUANT_CALIBRARTION_WARMUP_COUNT"],
    )
)

kv_cache.QUANT_CALIBRATION_INFERENCE_COUNT = kv_cache.add(
    EnvVar(
        name="LIGHTLLM_KV_QUANT_CALIBRATION_INFERENCE_COUNT",
        var_type=EnvVarType.INT,
        default=4000,
        description="KV 量化校准推理次数，达到此次数后输出统计校准结果",
        min_value=1,
        aliases=["LIGHTLLM_KV_QUANT_CALIBRARTION_INFERENCE_COUNT"],
    )
)

kv_cache.DISK_CACHE_PROMPT_LIMIT_LENGTH = kv_cache.add(
    EnvVar(
        name="LIGHTLLM_DISK_CACHE_PROMPT_LIMIT_LENGTH",
        var_type=EnvVarType.INT,
        default=2048,
        description="磁盘缓存提示长度限制",
        min_value=1,
    )
)

manager.register_group(kv_cache)


# ============================================================================
# 分布式和并行组
# ============================================================================
distributed = EnvGroup("distributed", "分布式通信和并行配置")

distributed.MOE_MODE = distributed.add(
    EnvVar(
        name="LIGHTLLM_MOE_MODE",
        var_type=EnvVarType.STRING,
        default="TP",
        description="MoE 并行模式",
        choices=["TP", "EP", "TP_EP"],
        aliases=["MOE_MODE"],
    )
)

distributed.DEEPEP_SMS = distributed.add(
    EnvVar(
        name="LIGHTLLM_DEEPEP_SMS",
        var_type=EnvVarType.INT,
        default=None,
        description="DeepEP SMS 数量",
        aliases=["DEEPEP_SMS"],
    )
)

distributed.NUM_MAX_DISPATCH_TOKENS_PER_RANK = distributed.add(
    EnvVar(
        name="LIGHTLLM_NUM_MAX_DISPATCH_TOKENS_PER_RANK",
        var_type=EnvVarType.INT,
        default=256,
        description="每个 rank 最大派发 Token 数，需大于单卡最大 batch size 且是 8 的倍数",
        min_value=8,
        aliases=["NUM_MAX_DISPATCH_TOKENS_PER_RANK"],
    )
)

distributed.USE_VLLM_CUSTOM_ALLREDUCE = distributed.add(
    EnvVar(
        name="LIGHTLLM_USE_VLLM_CUSTOM_ALLREDUCE",
        var_type=EnvVarType.BOOL,
        default=False,
        description="是否使用 vLLM 自定义 AllReduce",
    )
)

distributed.DISABLE_KV_TRANS_USE_P2P = distributed.add(
    EnvVar(
        name="LIGHTLLM_DISABLE_KV_TRANS_USE_P2P",
        var_type=EnvVarType.BOOL,
        default=False,
        description="禁用 KV 传输使用 P2P",
        aliases=["DISABLE_KV_TRANS_USE_P2P"],
    )
)

distributed.DISABLE_CC_METHOD = distributed.add(
    EnvVar(
        name="LIGHTLLM_DISABLE_CC_METHOD",
        var_type=EnvVarType.BOOL,
        default=False,
        description="禁用 CC 方法",
        aliases=["DISABLE_CC_METHOD"],
    )
)

manager.register_group(distributed)


# ============================================================================
# NCCL 配置组
# ============================================================================
nccl = EnvGroup("nccl", "NCCL 通信配置")

nccl.MAX_NCHANNELS = nccl.add(
    EnvVar(
        name="LIGHTLLM_NCCL_MAX_NCHANNELS",
        var_type=EnvVarType.STRING,
        default="2",
        description="NCCL 最大通道数",
        aliases=["NCCL_MAX_NCHANNELS"],
    )
)

nccl.NSOCKS_PER_CHANNEL = nccl.add(
    EnvVar(
        name="LIGHTLLM_NCCL_NSOCKS_PER_CHANNEL",
        var_type=EnvVarType.STRING,
        default="1",
        description="NCCL 每通道 Socket 数",
        aliases=["NCCL_NSOCKS_PER_CHANNEL"],
    )
)

nccl.SOCKET_NTHREADS = nccl.add(
    EnvVar(
        name="LIGHTLLM_NCCL_SOCKET_NTHREADS",
        var_type=EnvVarType.STRING,
        default="1",
        description="NCCL Socket 线程数",
        aliases=["NCCL_SOCKET_NTHREADS"],
    )
)

nccl.DEBUG = nccl.add(
    EnvVar(
        name="LIGHTLLM_NCCL_DEBUG",
        var_type=EnvVarType.STRING,
        default=None,
        description="NCCL 调试级别",
        aliases=["NCCL_DEBUG"],
    )
)

manager.register_group(nccl)


# ============================================================================
# GPU 和 CUDA 配置组
# ============================================================================
gpu = EnvGroup("gpu", "GPU 和 CUDA 相关配置")

gpu.CUDA_GROUPED_TOPK = gpu.add(
    EnvVar(
        name="LIGHTLLM_CUDA_GROUPED_TOPK",
        var_type=EnvVarType.BOOL,
        default=False,
        description="是否使用 CUDA 分组 TopK",
    )
)

gpu.USE_TRITON_FP8_SCALED_MM = gpu.add(
    EnvVar(
        name="LIGHTLLM_USE_TRITON_FP8_SCALED_MM",
        var_type=EnvVarType.BOOL,
        default=False,
        description="是否使用 Triton FP8 缩放矩阵乘法",
    )
)

gpu.TRITON_AUTOTUNE_LEVEL = gpu.add(
    EnvVar(
        name="LIGHTLLM_TRITON_AUTOTUNE_LEVEL",
        var_type=EnvVarType.INT,
        default=0,
        description="Triton 自动调优级别",
        min_value=0,
        max_value=3,
    )
)

gpu.DISABLE_GPU_TENSOR_CACHE = gpu.add(
    EnvVar(
        name="LIGHTLLM_DISABLE_GPU_TENSOR_CACHE",
        var_type=EnvVarType.BOOL,
        default=False,
        description="禁用 GPU 张量缓存",
        aliases=["DISABLE_GPU_TENSOR_CACHE"],
    )
)

gpu.RMSNORM_WARPS = gpu.add(
    EnvVar(
        name="LIGHTLLM_RMSNORM_WARPS",
        var_type=EnvVarType.INT,
        default=8,
        description="RMSNorm Warp 数量",
        min_value=1,
        aliases=["RMSNORM_WARPS"],
    )
)

manager.register_group(gpu)


# ============================================================================
# 专家冗余组
# ============================================================================
redundancy = EnvGroup("redundancy", "专家冗余相关配置")

redundancy.UPDATE_INTERVAL = redundancy.add(
    EnvVar(
        name="LIGHTLLM_REDUNDANCY_EXPERT_UPDATE_INTERVAL",
        var_type=EnvVarType.INT,
        default=30 * 60,
        description="冗余专家更新间隔",
        unit="秒",
        min_value=1,
    )
)

redundancy.UPDATE_MAX_LOAD_COUNT = redundancy.add(
    EnvVar(
        name="LIGHTLLM_REDUNDANCY_EXPERT_UPDATE_MAX_LOAD_COUNT",
        var_type=EnvVarType.INT,
        default=1,
        description="冗余专家更新最大加载次数",
        min_value=1,
    )
)

manager.register_group(redundancy)


# ============================================================================
# Radix Tree 组
# ============================================================================
radix_tree = EnvGroup("radix_tree", "Radix Tree 索引配置")

radix_tree.MERGE_ENABLE = radix_tree.add(
    EnvVar(
        name="LIGHTLLM_RADIX_TREE_MERGE_ENABLE",
        var_type=EnvVarType.BOOL,
        default=False,
        description="启用定期合并 radix tree 叶节点，防止插入查询性能下降",
    )
)

radix_tree.MERGE_DELTA = radix_tree.add(
    EnvVar(
        name="LIGHTLLM_RADIX_TREE_MERGE_DELTA",
        var_type=EnvVarType.INT,
        default=6000,
        description="Radix tree 合并增量",
        min_value=1,
    )
)

manager.register_group(radix_tree)


# ============================================================================
# 批处理和调度组
# ============================================================================
scheduling = EnvGroup("scheduling", "批处理和调度相关配置")

scheduling.MAX_BATCH_SHARED_GROUP_SIZE = scheduling.add(
    EnvVar(
        name="LIGHTLLM_MAX_BATCH_SHARED_GROUP_SIZE",
        var_type=EnvVarType.INT,
        default=4,
        description="最大批处理共享组大小",
        min_value=1,
    )
)

scheduling.DISABLE_CHECK_MAX_LEN_INFER = scheduling.add(
    EnvVar(
        name="LIGHTLLM_DISABLE_CHECK_MAX_LEN_INFER",
        var_type=EnvVarType.BOOL,
        default=False,
        description="禁用最大长度推理检查",
        aliases=["DISABLE_CHECK_MAX_LEN_INFER"],
    )
)

manager.register_group(scheduling)


# ============================================================================
# 健康检查组
# ============================================================================
health = EnvGroup("health", "健康检查相关配置")

health.FAILURE_THRESHOLD = health.add(
    EnvVar(
        name="LIGHTLLM_HEALTH_FAILURE_THRESHOLD",
        var_type=EnvVarType.INT,
        default=3,
        description="健康检查失败阈值",
        min_value=1,
        aliases=["HEALTH_FAILURE_THRESHOLD"],
    )
)

health.TIMEOUT = health.add(
    EnvVar(
        name="LIGHTLLM_HEALTH_TIMEOUT",
        var_type=EnvVarType.INT,
        default=100,
        description="健康检查超时时间",
        unit="秒",
        min_value=1,
        aliases=["HEALTH_TIMEOUT"],
    )
)

health.CHECK_INTERVAL_SECONDS = health.add(
    EnvVar(
        name="LIGHTLLM_HEALTH_CHECK_INTERVAL_SECONDS",
        var_type=EnvVarType.INT,
        default=88,
        description="健康检查间隔",
        unit="秒",
        min_value=1,
        aliases=["HEALTH_CHECK_INTERVAL_SECONDS"],
    )
)

health.DEBUG_RETURN_FAIL = health.add(
    EnvVar(
        name="LIGHTLLM_DEBUG_HEALTHCHECK_RETURN_FAIL",
        var_type=EnvVarType.BOOL,
        default=False,
        description="调试：健康检查返回失败",
        aliases=["DEBUG_HEALTHCHECK_RETURN_FAIL"],
    )
)

manager.register_group(health)


# ============================================================================
# 模型特定配置组
# ============================================================================
model = EnvGroup("model", "模型特定配置")

model.NTK_ALPHA = model.add(
    EnvVar(
        name="LIGHTLLM_NTK_ALPHA",
        var_type=EnvVarType.FLOAT,
        default=1.0,
        description="LLaMA NTK Alpha 参数",
        min_value=0.0,
    )
)

model.INTERNVL_IMAGE_LENGTH = model.add(
    EnvVar(
        name="LIGHTLLM_INTERNVL_IMAGE_LENGTH",
        var_type=EnvVarType.INT,
        default=256,
        description="InternVL 图像序列长度",
        min_value=1,
        aliases=["INTERNVL_IMAGE_LENGTH"],
    )
)

model.IMAGE_H = model.add(
    EnvVar(
        name="LIGHTLLM_IMAGE_H",
        var_type=EnvVarType.INT,
        default=448,
        description="图像高度",
        min_value=1,
        aliases=["IMAGE_H"],
    )
)

model.IMAGE_W = model.add(
    EnvVar(
        name="LIGHTLLM_IMAGE_W",
        var_type=EnvVarType.INT,
        default=448,
        description="图像宽度",
        min_value=1,
        aliases=["IMAGE_W"],
    )
)

model.MAX_PATH_NUM = model.add(
    EnvVar(
        name="LIGHTLLM_MAX_PATH_NUM",
        var_type=EnvVarType.INT,
        default=13,
        description="最大路径数",
        min_value=1,
        aliases=["MAX_PATH_NUM"],
    )
)

model.LOADWORKER = model.add(
    EnvVar(
        name="LIGHTLLM_LOADWORKER",
        var_type=EnvVarType.INT,
        default=1,
        description="模型加载工作进程数",
        min_value=1,
        aliases=["LOADWORKER"],
    )
)

model.USE_WHISPER_SDPA_ATTENTION = model.add(
    EnvVar(
        name="LIGHTLLM_USE_WHISPER_SDPA_ATTENTION",
        var_type=EnvVarType.BOOL,
        default=False,
        description="Whisper 重训后使用特定的 SDPA 实现以提升精度",
    )
)

manager.register_group(model)


# ============================================================================
# 外部服务组
# ============================================================================
external = EnvGroup("external", "外部服务和 API 配置")

external.PETREL_PATH = external.add(
    EnvVar(
        name="LIGHTLLM_PETREL_PATH",
        var_type=EnvVarType.STRING,
        default="~/petreloss.conf",
        description="Petrel 配置文件路径",
        aliases=["PETRELPATH"],
    )
)

external.RETURN_LIST = external.add(
    EnvVar(
        name="LIGHTLLM_RETURN_LIST",
        var_type=EnvVarType.BOOL,
        default=False,
        description="TGI 兼容：是否返回列表格式",
        aliases=["RETURN_LIST"],
    )
)

manager.register_group(external)


# ============================================================================
# 便捷访问接口
# ============================================================================
class EnvRegistry:
    """
    环境变量注册表的便捷访问接口

    用法：
        from lightllm.utils.envs import env

        # 通过分组访问
        env.logging.LOG_LEVEL.get()  # -> "info"

        # 直接通过管理器访问
        env.manager.get_value("LIGHTLLM_LOG_LEVEL")

        # 验证所有变量
        errors = env.manager.validate_all()

        # 生成文档
        print(env.manager.generate_docs())
    """

    def __init__(self):
        self.manager = manager
        self.logging = logging
        self.server = server
        self.memory = memory
        self.sampling = sampling
        self.kv_cache = kv_cache
        self.distributed = distributed
        self.nccl = nccl
        self.gpu = gpu
        self.redundancy = redundancy
        self.radix_tree = radix_tree
        self.scheduling = scheduling
        self.health = health
        self.model = model
        self.external = external


# 全局单例
env = EnvRegistry()
