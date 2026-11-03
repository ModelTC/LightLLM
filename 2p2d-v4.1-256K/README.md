# V4.1 FP8：双机 P + 双机 D，256K

基于 `support_ds4_v41`。本目录 launcher/config 为部署命令的唯一来源，每个参数一行；部署时复制原文件并核对 SHA256，不另拼启动命令。

| 主机 | 角色 | rank | HTTP |
| --- | --- | --- | --- |
| 10.119.30.78 | P + PD master | 0 | 8019 / 60011 |
| 10.119.30.56 | P | 1 | 8019 |
| 10.119.29.251 | D | 0 | 8121 |
| 10.119.29.227 | D | 1 | 8121 |

每个 P/D 组为 DPEP16、nnodes2；每台注册 master，master mode=2p2d。P 使用 DeepEP V2，D 使用 legacy/IBGDA。主/草稿权重均为 `/mnt/vqa/models/DeepSeek-V4.1-Flash-1017-FP8`，DSpark5，max_req_total_len262144，mem_fraction0.75，KV自动profile。P batch/chunk8192、CPU cache1024GiB/主机、disk cache5120GiB/主机；D batch1024、graph batch64/rank。内核和磁盘缓存保留节点本地目录。

运行目录见 config.txt。冻结源码放 `$RUN_DIR/src`，本目录脚本放 `$RUN_DIR/deploy`。先核对 GPU/旧进程/SysV/端口空闲、模型分片可读，再启动 master，随后四台并发启动；每个 launcher 在相同 shell 中设置 PATH/PYTHONPATH/nofile1048576。完整验收需要四节点健康、master2P2D就绪、实际文本/图片请求、KV/state传输和所有长期进程FD检查。

各机器执行：

```bash
bash "$RUN_DIR/deploy/launch_pd_master.sh"  # 仅 .30.78，先启动
bash "$RUN_DIR/deploy/launch_prefill.sh" 0  # .30.78
bash "$RUN_DIR/deploy/launch_prefill.sh" 1  # .30.56
bash "$RUN_DIR/deploy/launch_decode.sh" 0   # .29.251
bash "$RUN_DIR/deploy/launch_decode.sh" 1   # .29.227
```

本轮部署到上述四台节点；启动与验收进度追加到仓库根目录 test.md。

D节点使用隔离目录 `$RUN_DIR/deps/deepep-pr696` 中的历史自编译DeepEP PR696 wheel，不修改全局安装；wheel SHA256=e4981f5645c23b3d08d61eb6409503ec6f3a0c20a5eb3005eec1dd45f729ea9a。当前全局2.1.0+60d4403在D双机首次legacy dispatch复现illegal memory access；隔离依赖下16rank dispatch/combine、数值及graph检查通过。P仍使用全局DeepEP V2。

本轮外部入口为 `http://10.210.6.10:11046`，映射H100-msw的8020，再由本目录haproxy.cfg转发至P0的60011；10418/10066不修改。

P 开启 `--enable_dp_prompt_cache_fetch`，在同一节点的 DP rank 间复用前缀缓存；D 开启 `--mtp_dynamic_verify`，按置信度动态选择主模型验证行，draft 最大步数仍为5。P/D 的 `--mtp_step` 暂均保持5：P 仅提交 draft KV，但当前 P/D 原样传输的 compressor ring 尺寸依赖该值，不能单独把 P 改成1。
