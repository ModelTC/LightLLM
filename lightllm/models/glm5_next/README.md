# GLM-5.3 Flash：首版推理支持

基于 main `cd2edb90`，参考 [PR #1525](https://github.com/ModelTC/LightLLM/pull/1525)
移植必要的模型逻辑。使用现有 `bsh_dsv4` 镜像，无需升级依赖或修改模型文件。

## 支持范围

- 文本生成、原生 FP8 权重、BF16 激活、普通 TP、chunked prefill、decode CUDA graph。
- KDA、NoPE sparse MLA、4-token K-pool、mHC、GLM 的 sigmoid gated RMSNorm 和 clamped SwiGLU。
- main 的大小页前缀缓存与请求状态管理；首版验证范围为单机 H200、TP4、8K 内上下文。
- 暂不支持 MTP、EP、PD 分离、TP/SP 混合、微批重叠和 prefill CUDA graph；模型入口会显式拒绝这些组合。

## 缓存布局与流程

KDA 的卷积与 FP32 recurrent state 直接使用 `ReqManagerForMamba` 和
`LinearAttCacheManager`。大小页保存、命中恢复、请求槽位回收、CPU 页搬运均沿用 main。

每个稀疏注意力层、每个 token 使用一个 904 元素的 BF16 容器：

| 区域 | 内容 |
| --- | --- |
| 0–511 | MLA latent KV |
| 512–575 | 零填充，适配镜像已有的 576 维 FlashMLA/FA3 接口 |
| 576–703 | K-pool 原始 index key |
| 704–831 | K-pool 压缩 gate |
| 末尾 132 字节 | 128 维 FP8 压缩 key 与 FP32 scale，仅完整 pool 的末 token 有效 |

一次 KV 搬运就能带走全部 K-pool 历史。跨 chunk、跨缓存命中的不足 4-token 尾部从原始
token KV 重建，不引入独立的池化尾状态，也不改 scheduler/radix cache 的生命周期。
代价是每个注意力层每 token 1808 字节的 KV；11 层合计 19,888 字节，TP 各 rank 复制。
完整 pool 选择 512 组后展开为最多 2048 个 token，另保留当前未完成 pool 的尾部。

共享算子的扩展参数保持原默认值；GLM 显式启用 sigmoid gate、无 `up + 1` 的 clamp、
KDA 的 exp2 gate。模型特有的权重、attention、索引和 tokenizer 适配放在本目录。
MoE 在推理调用处显式传入 `alpha=1.0`、`limit=10.0`、`clamp_up_add_one=False`，
沿 `experts → __call__ → _fused_experts` 传给激活算子，不在通用 MoE 对象上保存激活配置。
GPT-OSS 的专用 experts 调用仍显式选择 `up + 1`；EP/Marlin 尚未实现的 clamped SwiGLU
组合会在执行时拒绝。

## 启动与测速

在工作区根目录，使用容器中现有环境（GPU 编号按实际空闲情况调整）：

```bash
docker exec -d -w /mtc/baishihao/LightLLM \
  -e PYTHONPATH=/mtc/baishihao/LightLLM \
  -e CUDA_VISIBLE_DEVICES=2,3,4,6 -e LOADWORKER=4 \
  bsh_dsv4 bash -lc 'exec python -m lightllm.server.api_server \
    --model_dir /mtc/models/GLM-5.3-Flash --model_name glm53 \
    --tp 4 --host 127.0.0.1 --port 18153 --nccl_port 28153 \
    --max_total_token_num 32768 --max_req_total_len 8192 \
    --batch_max_tokens 2048 --chunked_prefill_size 1024 \
    --running_max_req_size 16 --graph_max_batch_size 8 --graph_max_len_in_batch 8192 \
    --linear_att_hash_page_size 128 --linear_att_page_block_num 4 \
    --linear_att_cache_size 32 --disable_vision --disable_audio \
    --enable_fused_shared_experts > /tmp/glm53.log 2>&1'
```

这组参数用 128-token 小页、512-token 大页，方便同时覆盖两种状态恢复路径。
`linear_att_cache_size` 控制小页 checkpoint 的数量，运行中 KDA state 按请求槽位分配。
CPU cache 开启时，`cpu_cache_token_page_size` 必须等于大页 token 数。
本次服务测试未开启完整 CPU KV offload；CPU 页的数据布局与 TP 分片通过下述往返测试验证。

```bash
docker exec -w /mtc/baishihao/LightLLM \
  -e PYTHONPATH=/mtc/baishihao/LightLLM bsh_dsv4 \
  python test/benchmark/service/benchmark_glm53_flash.py \
    --input-tokens 1024 4096 --concurrency 1 4 8 \
    --output-tokens 128 --repeats 3 --output /tmp/glm53-benchmark.json
```

脚本每组预热，固定输入/输出 token 数，随机化首段 token 避免前缀命中影响 TTFT。
分别记录首 token 延迟、decode TPOT、每请求 decode tok/s，以及包含 prefill 的输出吞吐。
权重加载、编译和 graph 捕获不计入请求耗时。测速使用 `ignore_eos=true`，因此不是质量评估。

## 实测结果（2026-09-10）

4 × H200，TP4，镜像 ID `c3f03de5e8dc`；PyTorch `2.11.0+cu130`、Triton `3.6.0`、
transformers `4.57.1`。启动参数如上，保留 prefix cache，但各测速请求的首段 token 不同。
先完成各形状的 JIT，再每组预热两轮、正式测三轮；输出固定 128 token。下表取轮次中位数。

| 输入 token | 并发 | TTFT（ms） | 每请求 decode（tok/s） | 总输出吞吐，含 prefill（tok/s） |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 1 | 431 | 99.8 | 75.1 |
| 1024 | 4 | 482 | 85.1 | 257.6 |
| 1024 | 8 | 747 | 68.5 | 391.6 |
| 4096 | 1 | 1160 | 99.6 | 52.6 |
| 4096 | 4 | 1461 | 85.7 | 173.6 |
| 4096 | 8* | 2154 | 64.1 | 150.5 |

`*` 4096 × 8 再加输出超过本次 32768-token KV 容量，出现排队/分批；该行是容量压力测试，
不能当作完整八路同时 decode 的性能。提高 `max_total_token_num` 后应重新测量。

单请求约 10 ms/token 可以作为首版基线，尚未到硬件上限。32 次 decode graph 的独立 profile
中，每 rank 每 token 有 1723 个 GPU kernel，graph 跨度中位数约 9.9 ms。
按 kernel duration 合计分类：MoE GEMM 约 33%，其他 GEMM/BMM 20%，mHC 13%，
MoE 路由/归并 10%，K-pool 索引 7%，TP 通信 5%，KDA recurrence/conv/gated norm 3%。
这些百分比用于定位算子耗时，不包含 CPU 调度，也不把 prefill 的通信等待计入 decode。
18B 激活参数之外，45 层的小矩阵运算、路由和 mHC 都会产生开销；后续应先优化 MoE/GEMM
与算子融合，而不是重写大小页管理流程。

首次遇到新形状可能触发十几秒的 JIT，表中是预热后速度。本版没有实现统一的 prefill 形状
分桶，也没有跑完整精度基准或模型宣称的超长上下文测试。

## 验证

```bash
docker exec -w /mtc/baishihao/LightLLM \
  -e PYTHONPATH=/mtc/baishihao/LightLLM -e CUDA_VISIBLE_DEVICES=0 bsh_dsv4 \
  python -m pytest -q unit_tests/models/glm5_next
```

新增 19 项测试覆盖：KDA chunk/decode 对照 FP32 recurrence；跨 chunk、非连续 KV 和长序列
K-pool；当前镜像的 NoPE attention；mHC 数值及解码调优；大小页 checkpoint、完整 CPU 页往返；
TP4 的 replicated MLA/index KV 与各 rank 独立 KDA state 区域；transformers 5 tokenizer 文件
在现有 transformers 4 镜像中的兼容。

实际服务验证了中文自我介绍、`17 × 23 = 391`、Python 列表求和函数，并正常遇 EOS 结束。
为了检查最终答案，使用 checkpoint 的 `reasoning_effort="low"` 模板并追加 `</think>`；
原始默认 Max reasoning 模板也能生成连贯推理，但 256-token 输出上限可能在思考期间截断。
2179-token 请求的冷/热输出 token ID 完全一致，热请求命中 2176-token 小页；扩展到
3075 token 的请求命中 2048-token 大页前缀。四并发重复请求、流式中途断开后继续请求均通过，
日志确认请求槽位全部释放。

共享流程回归测试 `test_config_utils.py` 与 radix cache 的其余 14 项通过。
`test_radix_cache.py::test_case10` 在原始 main 上同样失败：没有传入 mem_manager 的实例调用
`flush_cache()` 触发断言。该既有问题不在本次模型支持中修改。
