# V4.1 FP8：双机 P + 双机 D，256K

基于 `support_ds4_v41`。本目录 launcher/config 为部署命令的唯一来源，每个参数一行；部署时复制原文件并核对 SHA256，不另拼启动命令。

| 主机 | 角色 | rank | HTTP |
| --- | --- | --- | --- |
| 10.119.17.62 | P + PD master | 0 | 8019 / 60011 |
| 10.119.17.55 | P | 1 | 8019 |
| 10.119.23.225 | D | 0 | 8121 |
| 10.119.23.201 | D | 1 | 8121 |

每个 P/D 组为 DPEP16、nnodes2；每台注册 master，master mode=2p2d。P 使用 DeepEP V2，D 使用 legacy/IBGDA。主/草稿权重均为 `/mnt/vqa/models/DeepSeek-V4.1-Flash-1017-FP8`，DSpark5，max_req_total_len262144，mem_fraction0.75，KV自动profile。P batch/chunk8192、CPU cache1024GiB/主机、disk cache5120GiB/主机；D batch1024、graph batch64/rank。内核和磁盘缓存保留节点本地目录。

运行目录见 config.txt。冻结源码放 `$RUN_DIR/src`，本目录脚本放 `$RUN_DIR/deploy`。先核对 GPU/旧进程/SysV/端口空闲、模型分片可读，再启动 master，随后四台并发启动；每个 launcher 在相同 shell 中设置 PATH/PYTHONPATH/nofile1048576。完整验收需要四节点健康、master2P2D就绪、实际文本/图片请求、KV/state传输和所有长期进程FD检查。

各机器执行：

```bash
bash "$RUN_DIR/deploy/launch_pd_master.sh"  # 仅 .17.62，先启动
bash "$RUN_DIR/deploy/launch_prefill.sh" 0  # .17.62
bash "$RUN_DIR/deploy/launch_prefill.sh" 1  # .17.55
bash "$RUN_DIR/deploy/launch_decode.sh" 0   # .23.225
bash "$RUN_DIR/deploy/launch_decode.sh" 1   # .23.201
```

当前为迁移后的待验证命令，尚未启动四机服务。验证进度与性能结果追加到仓库根目录 test.md。
