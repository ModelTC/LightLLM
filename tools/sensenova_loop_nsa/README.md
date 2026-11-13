# SenseNova Loop NSA

这是基于 `9640f76dfb61e36add0a75479edcecdcb452b6e2` 的收敛实现。默认路径只保留可验证的
原生 FA3、FlashInfer/vLLM TopK、精确 radix fallback 和 TMA score；实验性的 cluster TopK
与自定义 fused GQA 已移除，避免 review 时混入未采用的分支。

## 运行路径

- 训练得到的 8×128 Indexer Q/K/W 保持 BF16/FP32 语义，Indexer score 使用 FP32 归约。
- P 阶段先做 causal Indexer score，再用 vLLM prefill TopK（可用时）或 FlashInfer
  page-table TopK；vLLM 的同分 cutoff 行批量交给 FlashInfer 修复，保证 baseline 的
  FP32、同分小下标优先语义。
- D 阶段默认保留 exact packed rank key，最后统一交给原生 FA3 selected GQA；关闭
  `DSA_DECODE_PACKED=0` 后，较大 batch 才会选择 radix/FlashInfer。page table 映射在 TopK
  路径中完成，避免额外的 K/V gather。
- 超过 262K 的单请求长 P 会把 Indexer-K 按 page table 紧凑一次，再用 TMA score kernel；
  不满足条件时回退到指针版 Triton。当前启动脚本的 `DSA_PREFILL_POLICY=auto` 在 TP2/TP4
  的每卡 query head ≤8 时让 P 走原生 Full Attention，以避免已测出的长 P 回退；需要实际启用
  NSA P 时显式设置 `DSA_PREFILL_POLICY=dsa`。D 的短上下文仍由
  `DSA_SHORT_DENSE_FA3=auto` 路由到原生 Full Attention。

## 关键开关

`tools/sensenova_loop_nsa/launch.sh` 已固定 baseline/candidate 的启动参数。候选默认：

- `DSA_TOPK_IMPL=auto`
- `DSA_PREFILL_POLICY=auto`
- `DSA_SCORE_QUERY_TILE=16`，`DSA_SCORE_KEY_TILE=128`
- H200 使用 4096 MiB score workspace 和 FA3 自动 split；其他 GPU 使用 1024 MiB
- `DSA_SHORT_DENSE_FA3=auto`

需要复现 baseline 时使用 `MODE=baseline`；需要定位路径时可显式设置
`DSA_TOPK_IMPL=packed|radix|flashinfer|vllm`。所有精确性相关路径默认开启
`DSA_PREFILL_TOPK_TIE_REPAIR=1`。

## 已验证范围

- TMA score 与指针版在随机 page table、causal mask、FP32 输出上逐元素一致。
- D 的 synthetic H200 测试在 batch 8/128、131K/524K 上相对 dense FA3 约有 3–15×
  的 attention kernel 优势；batch 1 或短上下文仍由 dense 路径获胜。
- 长 P 的主要成本仍是 Indexer score；TP4、8 个 query heads、524K 上，当前 NSA
  约 2.1 s，dense FA3 约 1.74 s，因此 `auto` 保留短 P dense 路由，不能把这项
  kernel 数据直接当成整模型端到端加速承诺。
- 80 样本整模型精度/P-D A/B 需要 step519 与 MTP 权重在测试节点可见后运行。

## 检查命令

```bash
PYTHONPATH="$PWD" python -m pytest -q unit_tests/models/sensenova_loop

MODE=baseline DRY_RUN=1 bash tools/sensenova_loop_nsa/launch.sh
MODE=candidate DRY_RUN=1 bash tools/sensenova_loop_nsa/launch.sh

PYTHONPATH="$PWD" python -u tools/sensenova_loop_nsa/benchmark.py \
  --baseline-code ../work/original_runtime --query-tile 16 \
  --output ../work/results/kernels-final.json
```

`run_ab.py` 和 `verify_checkpoint.py` 用于权重可见节点上的 80 样本精度与服务
对照；`baseline_manifest.json` 固定原始启动链和覆盖层哈希，便于 review 与复现。
