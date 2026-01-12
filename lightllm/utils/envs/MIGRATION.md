# 环境变量管理框架迁移指南

本文档指导如何从旧的分散式环境变量管理迁移到新的集中式框架。

## 快速入门

### 新框架用法

```python
from lightllm.utils.envs import env

# 获取值（推荐方式）
log_level = env.logging.LOG_LEVEL.get()
timeout = env.server.GUNICORN_TIMEOUT.get()
moe_mode = env.distributed.MOE_MODE.get()

# 设置值
env.logging.LOG_LEVEL.set("debug")

# 检查是否已设置
if env.server.UNIQUE_SERVICE_NAME_ID.is_set():
    name = env.server.UNIQUE_SERVICE_NAME_ID.get()

# 验证所有环境变量
errors = env.manager.validate_all()
if errors:
    print("配置错误:", errors)

# 生成文档
print(env.manager.generate_docs())
```

## 迁移映射表

### 服务配置

| 旧代码 | 新代码 |
|--------|--------|
| `os.getenv("LIGHTLMM_GUNICORN_TIME_OUT", 180)` | `env.server.GUNICORN_TIMEOUT.get()` |
| `os.getenv("LIGHTLMM_GUNICORN_KEEP_ALIVE", 10)` | `env.server.GUNICORN_KEEP_ALIVE.get()` |
| `os.getenv("LIGHTLLM_WEBSOCKET_MAX_SIZE", ...)` | `env.server.WEBSOCKET_MAX_SIZE.get()` |

### 内存配置

| 旧代码 | 新代码 |
|--------|--------|
| `os.getenv("LIGHTLLM_REQS_BUFFER_BYTE_SIZE", ...)` | `env.memory.REQS_BUFFER_BYTE_SIZE.get()` |
| `os.getenv("LIGHTLLM_RPC_BYTE_SIZE", ...)` | `env.memory.RPC_BYTE_SIZE.get()` |
| `enable_env_vars("LIGHTLLM_HUGE_PAGE_ENABLE")` | `env.memory.HUGE_PAGE_ENABLE.get()` |

### 分布式配置

| 旧代码 | 新代码 |
|--------|--------|
| `os.getenv("MOE_MODE", "TP")` | `env.distributed.MOE_MODE.get()` |
| `os.getenv("NUM_MAX_DISPATCH_TOKENS_PER_RANK", 256)` | `env.distributed.NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()` |

### KV 缓存配置

| 旧代码 | 新代码 |
|--------|--------|
| `os.getenv("LIGHTLLM_KV_QUANT_CALIBRARTION_WARMUP_COUNT", 0)` | `env.kv_cache.QUANT_CALIBRATION_WARMUP_COUNT.get()` |
| `os.getenv("LIGHTLLM_KV_QUANT_CALIBRARTION_INFERENCE_COUNT", 4000)` | `env.kv_cache.QUANT_CALIBRATION_INFERENCE_COUNT.get()` |

### 采样参数

| 旧代码 | 新代码 |
|--------|--------|
| `os.getenv("INPUT_PENALTY", "False").upper() in [...]` | `env.sampling.INPUT_PENALTY.get()` |
| `os.getenv("SKIP_SPECIAL_TOKENS", "True").upper() in [...]` | `env.sampling.SKIP_SPECIAL_TOKENS.get()` |

## 向后兼容

### 使用兼容模块

如果需要逐步迁移，可以使用兼容模块：

```python
# 兼容模块提供与旧接口相同的函数
from lightllm.utils.envs.compat import (
    get_lightllm_gunicorn_time_out_seconds,
    get_lightllm_gunicorn_keep_alive,
    enable_env_vars,
)

# 这些函数内部会使用新框架，并发出废弃警告
timeout = get_lightllm_gunicorn_time_out_seconds()
```

### 别名支持

新框架支持旧的环境变量名称作为别名：

```python
# 以下两种方式等效（但旧名称会触发警告）
os.environ["LIGHTLMM_GUNICORN_TIME_OUT"] = "300"  # 旧名称（有拼写错误）
os.environ["LIGHTLLM_GUNICORN_TIMEOUT"] = "300"    # 新名称（推荐）

# 获取值时都能正常工作
timeout = env.server.GUNICORN_TIMEOUT.get()  # -> 300
```

## 拼写错误修正

以下环境变量名称有拼写错误，已在新框架中修正：

| 旧名称（错误） | 新名称（正确） |
|---------------|---------------|
| `LIGHTLMM_GUNICORN_TIME_OUT` | `LIGHTLLM_GUNICORN_TIMEOUT` |
| `LIGHTLMM_GUNICORN_KEEP_ALIVE` | `LIGHTLLM_GUNICORN_KEEP_ALIVE` |
| `LIGHTLLM_KV_QUANT_CALIBRARTION_WARMUP_COUNT` | `LIGHTLLM_KV_QUANT_CALIBRATION_WARMUP_COUNT` |
| `LIGHTLLM_KV_QUANT_CALIBRARTION_INFERENCE_COUNT` | `LIGHTLLM_KV_QUANT_CALIBRATION_INFERENCE_COUNT` |

旧名称作为别名继续支持，但会触发废弃警告。

## 添加新环境变量

如需添加新的环境变量，请在 `registry.py` 中对应的组添加：

```python
# 在对应组中添加
my_group.MY_NEW_VAR = my_group.add(
    EnvVar(
        name="LIGHTLLM_MY_NEW_VAR",
        var_type=EnvVarType.INT,
        default=100,
        description="描述这个变量的用途",
        min_value=1,
        max_value=1000,
        unit="毫秒",
    )
)
```

## 类型转换

新框架自动处理类型转换：

| EnvVarType | 输入示例 | 输出类型 |
|------------|---------|---------|
| `STRING` | `"hello"` | `str` |
| `INT` | `"42"` | `int` |
| `FLOAT` | `"3.14"` | `float` |
| `BOOL` | `"true"`, `"1"`, `"on"` | `bool` |
| `LIST_INT` | `"1,2,3"` | `List[int]` |
| `LIST_STR` | `"a,b,c"` | `List[str]` |

## 验证

新框架支持多种验证方式：

```python
# 在定义时指定
EnvVar(
    name="LIGHTLLM_MY_VAR",
    var_type=EnvVarType.INT,
    default=10,
    choices=[1, 5, 10, 20],      # 限制可选值
    min_value=1,                  # 最小值
    max_value=100,                # 最大值
    required=True,                # 必需变量
    validator=lambda x: x % 2 == 0,  # 自定义验证
)
```

## 最佳实践

1. **使用新框架定义变量**：所有新变量应在 `registry.py` 中定义
2. **使用 `LIGHTLLM_` 前缀**：保持命名一致性
3. **提供描述和单位**：便于生成文档
4. **设置合理的默认值**：避免必需变量为空
5. **使用类型提示**：指定正确的 `EnvVarType`
6. **分组管理**：将相关变量放在同一组

## 调试

```python
# 打印所有环境变量摘要
env.manager.print_summary()

# 导出为字典
config = env.manager.to_dict()
print(config)

# 查找特定变量
vars = env.manager.find_vars("TIMEOUT")
for v in vars:
    print(f"{v.name}: {v.get()}")
```
