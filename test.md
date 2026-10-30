# V4.1 移植到 support_ds4 的验证记录

## 2026-10-30：基线与迁移原则

- 目标分支 `support_ds4_v41`，worktree `/mtc/wanzihao/lightllm-ds4-v41`；基于远端已核对的 `support_ds4:6dc33df85cb92c2dfefb2206a92dff6b28375ba9`。
- 功能参考为旧 V4.1 `23d53ba5e`，额外迁入已经验证的 tokenizer 别名 medium→low、xhigh→high。原两个工作区未修改。
- 保留新基线的 DP-shared checkpoint、异步 CPU cache、紧凑 PD 协议、WRITE 生命周期、FlashMLA out/split workspace、融合 SwiGLU 和请求 presence all-reduce。未迁入旧 8 步调度、token-budget/underfill 或分组 FP8 量化实验。
- DeepEP V2 布局转换参考当前 support_ds4 的未提交适配，但显式使用旧 V4.1 已验证的 compute-stream ownership（dispatch/combine 的 async_with_compute_stream=False、allocate_on_comm_stream=False），避免曾复现的多层 graph allocator 膨胀。没有整文件覆盖新 MoE/EPLB 框架。
- 四台预检 .17.62/.17.55/.23.225 空闲，.23.201 已有新的 GPU 任务，不关闭未知进程。本轮先在前两台单卡测试，不启动 LLM Bench 压测。

## 测试环境

远端 `/opt/conda/bin/python` 3.10.20，Torch2.11.0+cu130；实际 FlashMLA 来自 `vllm/v1/attention/ops/flashmla.py`，支持 out，且 `_flashmla_C.sparse_decode_fwd_with_workspace` 已注册。测试使用 GPU7、现有本地 kernel cache；nofile1048576。未安装或替换推理依赖。远端没有 pytest，复用本机已有容器的纯 Python pytest 模块至独立 `test_deps/`，仅测试 PYTHONPATH 引用，未写 site-packages。首次本机 Python3.8 compileall 无法解析基线既有 match 语法，后续使用远端 Python3.10。

原始日志：`/mnt/vqa/wanzihao/dsv41_port_support_ds4_20261030/tests/`。初始失败轮次保留，不作为通过记录。

- initial_v41_vendor：128 passed / 3 failed；发现 Marlin fixture 的设备来源及移植中 DSpark 共享 stream 初始化不完整，随后修复。
- v41_second：146 passed / 5 failed / 7 skipped；8组真实小模型 FP4/FP8/视觉/DSpark 分块推理均已通过，失败来自旧视觉测试引用新基线已不存在的 block_start_idx/block_end_idx。按 start_idx/token_num 当前契约更新测试。
- v41_third：229 passed / 1 failed / 1 skipped；真实 checkpoint 图像处理参考已启用。剩余 V2 初始化测试引用旧 workspace-size helper，改为新基线的既有函数。
- base_regression/base_second：测试夹具缺少 dp/nnodes、prefill workspace、C128 scratch，导致旧基线接口测试失败。只补齐实际框架要求的测试输入，不在生产代码增加 fallback。

尚未完成整模型准确率、性能或四机 PD 验收；不将这些内核/小模型结果写成整机加速收益。

## 迁移回归通过

- `v41_all_final.log`：231 passed，30.84s，无跳过；包含 checkpoint 原生图像处理及 prompt 展开参考、FP4/FP8 小模型、CED、DSpark、parser、V2 ownership 与 autotuner 回归。
- `base_final.log`：644 passed，37.59s；覆盖 V4 推理、mHC、FlashMLA、shared expert、CUDA Graph pool、多卡缓存搬运、CPU cache 事件与 PD checkpoint。
- 改动文件 flake8、`git diff --check` 通过。测试使用已有依赖，无推理包安装或替换。
- 四机 PD 实测按用户要求推迟，等待通知；本轮仅启动 .17.62/.17.55 的双机 DPEP16 独立服务，不占用 .23 两台。
