# 1M · 4P2D · DeepSeek-V4.1-Flash-1017-FP8

本目录对应 11 月 26 日的六台 H100 报告。四台 prefill 组成两个独立的 TP16/DP16 双机组，两台 decode 组成一个 TP16/DP16 双机组，每台 8 卡。Master 和 P0 同机。[1M 4P2D 基线报告](http://10.9.66.47:5174/spaces/default/env/v41-58e80-msw/reports?region=ms-01-tech-cci&taskId=8c8a4f2f-cc59-474c-85ab-d38e6b767857)

## 1M 压测经验

复现本报告时保留两组 prefill、`last_chunk` 和 1.2× 回放速率。这组配置在两小时 SenseNova Chat 回放中测得 TTFT P90 为 8.93 s、OTPS P50 为 78.61 tok/s。报告只覆盖 1M 开启 `last_chunk` 的情况，单独判断该策略在 1M 的收益还需要同配置的 `early` 对照。

256K 2P2D 做过单独切换策略的实验。在相同 30 分钟窗口内，开启 `last_chunk` 后总 TPM 下降 29.9%，输出 TPM 下降 26.0%，因此 256K 目录保留提前分配策略。[256K 配置与对照数据](../2p2d-v4.1-256K-b128/README.md#256k-压测经验) · [256K 2P2D 基线报告](http://10.9.66.47:5174/spaces/default/env/v41-58e80-msw/reports?region=ms-01-tech-cci&taskId=dbb0f907-01e5-438a-a8a4-2ca10f388ec7)

## 启动

将本目录的 `config.txt` 和三个 launcher 放在各节点同一目录，在下表对应节点进入该目录执行命令。先启动 master，待 `http://10.119.18.206:60011/get_server_info` 就绪，再并发启动 P/D。Prefill 的第一个参数是组内 rank，第二个参数选 NCCL 组。两组的 rank 都从 0 开始。

| 节点 | 地址 | 命令 |
| --- | --- | --- |
| Master | 10.119.18.206 | `bash launch_pd_master.sh` |
| P0 | 10.119.18.206 | `bash launch_prefill.sh 0 1` |
| P1 | 10.119.29.200 | `bash launch_prefill.sh 1 1` |
| P2 | 10.119.23.185 | `bash launch_prefill.sh 0 2` |
| P3 | 10.119.23.182 | `bash launch_prefill.sh 1 2` |
| D0 | 10.119.20.61 | `bash launch_decode.sh 0` |
| D1 | 10.119.20.128 | `bash launch_decode.sh 1` |

## 源码、配置和回放

源码为 `e3c4fd173c4dec7c104f8df5114d1a5e7ab8618f`。配置默认指向原运行目录 `/root/v41-4p2d-e3c4fd173-lastchunk-1126-snova120/{master,prefill,decode}`。迁移时，各角色保留对应的 `releases/<commit>/source` 和 `deps`，按实际地址更新 master、两组 P 和 D 的 NCCL 主机、固定 decode hosts 及 NO_PROXY。H200-34 的归档含 `source.tar.gz`、`dependencies.json`、`live-launch-args.json` 和 `replay-effective-vus.json`。

三端使用 1048576 tokens 和 `last_chunk`。P/D 使用 DSpark 5 步，D 开启动态验证和 FlashInfer 采样，1024 个请求槽、768 batch tokens、最大 graph batch 128。P 使用 8192-token chunk 和 HRRN resume。P/D 的显存比例分别为 0.75/0.77。使用原环境时清除外部的 `MAX_REQ_TOTAL_LEN` 覆盖，以使用报告中的上下文长度。

压测入口为 `http://10.119.18.206:60011`。待 master 的 `/health` 显示 4 个 prefill、2 个 decode 均就绪后回放。报告为 SenseNova Chat 100%、1.2×、seed `w75jN`、2 小时，P50 OTPS 为 78.6 tok/s。

H200-34 归档目录为 `/mnt/vqa/wanzihao/v41-1017-fp8-1x32-20261112/20261126-4p2d-e3c4fd173-lastchunk-sensenova-120pct-rate`。
