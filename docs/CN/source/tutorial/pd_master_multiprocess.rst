.. _pd_master_multiprocess:

单机多进程 PD Master
====================

目标与边界
----------

单机多进程模式在一台 PD Master 主机上启动多个独立 HTTP worker，将 API 解析、
tokenizer、P/D 编排、token 消息处理和 SSE 序列化分散到多个 CPU 进程。客户端仍访问
一个公共端口。该模式只扩展 Master 侧 CPU 请求处理能力，不会加载多份 GPU 模型。

设置 ``--run_mode pd_master --httpserver_workers N`` 且 ``N > 1`` 时启用该模式。
``--httpserver_workers 1`` 保持原来的单进程启动路径。

架构
----

.. code-block:: text

   客户端
     │ HTTP / SSE
     ▼
   HAProxy（公共端口）
     ├── /liveness、/readiness、/health：聚合全部 worker 状态
     └── 其他 HTTP 请求：leastconn 转发
          │
          ├── PD Master worker 0（独立端口、独立 P/D 连接）
          ├── PD Master worker 1（独立端口、独立 P/D 连接）
          └── PD Master worker N-1
                 │
                 ├── RPyC ── 请求 ID 服务
                 ├── RPyC ── P 节点统计服务
                 └── RPyC ── Prompt Cache Tree 服务

   Prefill / Decode
     1. 访问公共端口的 /get_server_info，读取 pd_master_multi_ports
     2. 使用 worker 端口作为本机多进程模式的 pd_node_id
     3. 直接连接每个 worker 的 /pd_register 和 /kv_move_status

HAProxy 不代理 P/D WebSocket。公共端口上的 ``/pd_register`` 和
``/kv_move_status`` 返回 HTTP 426，P/D 必须能够直接访问
``pd_master_multi_ports`` 中的内部端口。请求的整个生命周期固定归属于接收它的
worker，逐 token 数据也不会经过共享 RPyC 服务。

启动流程
--------

Launcher 按以下顺序启动：

1. 校验 HAProxy 和启动参数，确定本次启动的共享 IPC 名称。
2. 动态分配 ``pd_master_multi_ports``；每个 worker 使用自己的监听端口作为
   ``pd_node_id``。
3. 启动 Metric Server、请求 ID 服务、P 节点统计服务和 Prompt Cache Tree 服务。
4. 启动 N 个独立的 PD Master HTTP worker。
5. 根据预先规划的 worker 地址生成 HAProxy 配置并启动公共入口；HAProxy 持续探测状态。
6. 监督全部关键进程；任一关键进程退出时清理整组服务，由部署系统重启。

每个后端进程只运行一个 Hypercorn worker。所有 worker 继承同一份启动参数和服务名，
Launcher 只通过子进程环境传递 ``worker_index``；worker 根据该索引从共享端口表读取
自己的监听端口，并连接同一组本机共享服务。
IPC 名称和完整 worker 端口表不写入 ``StartArgs``。多进程 worker 的
``config_server_host`` 和 ``config_server_port`` 均为
``None``，不会启动注册循环，也不会启动本机 config server。

P/D 节点发现
------------

未显式配置外部 config server 时，Prefill 和 Decode 首先访问：

.. code-block:: text

   GET http://<pd_master_ip>:<pd_master_port>/get_server_info

PD Master 响应包含 ``pd_master_multi_ports``。单 worker 模式返回
``[port]``，并使用响应中的 ``pd_node_id``；多 worker 模式返回 Launcher 预先规划的
完整端口列表，P/D 直接以 worker 端口作为 ``pd_node_id``，构建
``(worker_port, host:worker_port)`` 连接表。发现流程只访问一次公共入口，不依赖本机
注册表，也不会因公共入口的负载均衡而把多个 worker 误认为同一节点。

显式设置 ``--config_server_host`` 和 ``--config_server_port`` 的其他部署仍沿用外部
config server 的 ``/registered_objects`` 协议。单机多进程 PD Master 本身不接受这两个
参数，两种发现模式互不混用。

本分支保留独立 PD Master 通过 config server 异步共享 cache hint 和在途负载的
部署方式；其本地前缀树仍使用有序事件更新，并保留提示年龄和 P 节点实例校验。
单机多进程模式使用本 PR 的 RPyC 共享服务，不启动这条元数据同步通道。

请求 ID
-------

多进程模式启动一个独立的请求 ID RPyC 服务。每个 HTTP worker 的 client 一次申请
1024 个 group request ID，在本地号段内无 RPC 生成 ID；号段用完后再申请下一段。
服务端按 ``MAX_BEST_OF = 8`` 对齐并在锁内分配左闭右开区间
``[start_id, end_id)``，保证同一次启动中的所有 worker 不会产生重复请求 ID。

单机多进程模式不需要 config server 的大号段分配，也不通过公共入口代理多模态 ID
分配接口。独立部署之间和服务重启之间不承诺 ID 持久唯一。

共享选点信息
------------

每个 worker 维护自己的 P/D WebSocket、候选节点和请求状态。只有需要跨 worker 统一的
启发式信息放入两个独立 RPyC 组件：

* ``pd_node_stats_client.py`` / ``pd_node_stats_server.py`` 管理 Prefill pending
  字符数和请求数。
* ``prompt_cache_tree_client.py`` / ``prompt_cache_tree_server.py`` 管理 prompt
  连续块摘要的前缀树。

Selector 先从统计服务读取当前候选 P 节点的计数快照，再在本地候选集合中选点；选中后
通过 ``add_dispatched_info`` 原子增加两个计数，请求进入后续阶段时通过
``sub_dispatched_info`` 扣减。共享端只按 ``connection_key`` 管理数据，不保存 worker
节点注册表，也不决定候选集合。

Cache-aware 策略在 worker 内将每个连续的 ``sample_stride`` 字符块（默认 2048）
计算为一个摘要，再调用共享树的 ``prefix_match_sampled`` 和 ``insert_sampled``。
摘要覆盖块内全部字符，避免只取少数采样字符时将不同会话误判为高前缀匹配。
前缀树仅记录历史处理位置，不证明 GPU KV 仍驻留；实际命中需使用请求返回的
``cached_tokens``。命中率窗口和动态阈值仍是各 worker 私有数据。逐 token 数据、
GPU token load 和请求对象始终保留在本地进程。

独立 Master 的 config-server 同步使用带版本的 namespace；从旧采样 key 升级到
连续块摘要时，应同时更新全部 Master，避免不同版本无法共享提示。

健康检查和代理行为
------------------

HAProxy 为 ``/liveness``、``/readiness`` 和 ``/health`` 分别维护探测池。只有全部
worker 的对应探测成功时，公共接口才返回 HTTP 200；任一 worker 失败或超时则返回
HTTP 503。``/healthz`` 使用 ``/health`` 的聚合结果。

普通请求使用 ``leastconn`` 转发，不依据上述探测结果摘除 worker。推理请求关闭代理
自动重试，避免已提交的非幂等请求被重复发送。SSE 直接流式转发。

部署示例
--------

入口使用 `ray-haproxy <https://pypi.org/project/ray-haproxy/>`_ wheel 内置的原生
HAProxy。依赖已固定在 ``requirements.txt``：

.. code-block:: bash

   pip install -r requirements.txt

.. code-block:: bash

   python -m lightllm.server.api_server \
     --run_mode pd_master \
     --model_dir /path/to/model \
     --host 10.0.0.10 \
     --port 8089 \
     --httpserver_workers 4 \
     --select_p_d_node_strategy cache_aware

``--port`` 是公共入口端口。Launcher 会在启动 worker 前根据
``--httpserver_workers`` 动态分配完整的内部端口列表。``--host`` 和所有 worker
端口必须能被 P/D 节点访问。

P/D 仍只配置公共入口：

.. code-block:: text

   --pd_master_ip 10.0.0.10 --pd_master_port 8089

验证时可以检查公开拓扑和聚合状态：

.. code-block:: bash

   curl http://10.0.0.10:8089/get_server_info
   curl http://10.0.0.10:8089/liveness
   curl http://10.0.0.10:8089/readiness
   curl http://10.0.0.10:8089/health

端到端测试和性能验证
--------------------

CPU-only 端到端测试会启动真实 Launcher、HAProxy、HTTP worker 和 RPyC 服务，并使用
轻量 P/D 协议节点代替 GPU 模型。测试覆盖：

* 单 worker 与 2/4 worker 的发现、普通生成和 SSE；
* busy 返回、客户端取消、请求 ID 唯一性和共享指标；
* 全部 worker 的 liveness/readiness/health 聚合与内部 WebSocket 隔离；
* worker、共享服务和 HAProxy 异常退出后的整组清理；
* Prompt Cache Tree、P 节点统计和请求 ID 服务的并发一致性。

.. code-block:: bash

   PYTHONPATH=. pytest -q \
     unit_tests/server/test_pd_master_multiprocess_e2e.py \
     unit_tests/server/test_pd_master_launcher_integration.py \
     unit_tests/server/test_pd_master_topology_api.py \
     unit_tests/server/test_pd_master_req_ids.py \
     unit_tests/server/test_pd_node_stats.py \
     unit_tests/server/test_prompt_cache_tree_service.py

高负载对比默认执行 2000 个请求、128 并发，对比 1 worker 和最多 4 worker：

.. code-block:: bash

   LIGHTLLM_RUN_PD_MASTER_PERF=1 \
   PYTHONPATH=. pytest -q -s \
     unit_tests/server/test_pd_master_multiprocess_e2e.py::test_multiprocess_pd_master_high_load_speedup

可以通过 ``LIGHTLLM_PD_MASTER_PERF_REQUESTS``、
``LIGHTLLM_PD_MASTER_PERF_CONCURRENCY`` 和 ``LIGHTLLM_PD_MASTER_PERF_WORKERS``
调整负载。``LIGHTLLM_PD_MASTER_MIN_SPEEDUP`` 默认是 ``1.0``，用于确认多 worker
吞吐高于单 worker；专用性能机器可以设置更高门槛，设置为 ``0`` 时仅记录结果。
普通 CI 不开启该性能用例，避免机器负载波动造成不稳定失败。输出包含 req/s、P50/P95
延迟、服务 CPU 核使用量和吞吐加速比。

该基准走完整 HTTP、PD Master 和 P/D WebSocket 协议，但不加载模型权重，衡量的是
Master 控制面的扩展能力；GPU 推理吞吐仍需使用真实 P/D 部署单独测试。
