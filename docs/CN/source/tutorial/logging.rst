日志配置与分级规范
==================

默认行为
--------

LightLLM 使用 Python 标准库 logging。默认服务级别为 INFO，输出到 stdout。
模块 logger 继承 ``lightllm`` 的配置；模块本身不添加 handler。
默认文本日志使用 ``HH:mm:ss.SSS LEVEL [组件] 消息``，例如 ``14:51:26.838 INFO [infer] event=backend.selected ...``。
P/D 角色合并到组件中，例如 ``[P/infer]`` 和 ``[D/http]``。
多卡、多 DP、多节点时才显示对应的已知 rank，例如 ``[P/infer tp=1]``；未知 rank 不输出。
主机名、PID、完整模块名在默认文本中省略；DEBUG、ERROR/CRITICAL 和附带 traceback 的日志保留文件行号。
JSON 保留带日期/时区的毫秒时间、主机、PID、完整模块/组件、并行拓扑和来源文件行号。
进程上下文不向 fork/spawn 子进程传递，子进程自行初始化；节点和 P/D 角色通过环境变量传递。

HTTP access 默认关闭；成功请求完成摘要默认采样 1%，失败摘要不采样。
请求正文、采样参数、token ID 和张量不会因为开启请求摘要而输出。
现有 ``--detail_log`` 保留为显式内容诊断开关，其 DEBUG 输出仍需要对应模块启用 DEBUG。

配置
----

显式 CLI 参数优先于环境变量；未设置 CLI 时继承环境变量，最后使用默认值。
CLI 的连字符和下划线形式均可使用，例如 ``--log-level`` 和 ``--log_level``。
启动器将配置写入环境变量，使 spawn 和 HTTP 子进程使用同一配置。

.. list-table::
   :header-rows: 1
   :widths: 27 38 35

   * - CLI
     - 环境变量
     - 默认值 / 含义
   * - ``--log-level``
     - ``LIGHTLLM_LOG_LEVEL``
     - ``info``；服务日志级别
   * - ``--log-format``
     - ``LIGHTLLM_LOG_FORMAT``
     - ``text``；也支持 ``json``
   * - ``--log-module-levels``
     - ``LIGHTLLM_LOG_MODULE_LEVELS``
     - ``{}``；模块前缀到级别的 JSON 映射
   * - ``--log-request-sample-rate``
     - ``LIGHTLLM_LOG_REQUEST_SAMPLE_RATE``
     - ``0.01``；成功请求采样 1%，0 关闭，1 全量；失败不采样
   * - ``--log-stats-interval``
     - ``LIGHTLLM_LOG_STATS_INTERVAL``
     - ``10``；运行摘要间隔，单位为秒，必须大于 0
   * - ``--log-http-level``
     - ``LIGHTLLM_LOG_HTTP_LEVEL``
     - ``warning``；Hypercorn 错误日志级别
   * - ``--enable-http-access-log``
     - ``LIGHTLLM_LOG_HTTP_ACCESS``
     - ``0``；设为 1 开启 HTTP access 日志
   * - 无
     - ``LIGHTLLM_LOG_HTTP_EXCLUDE_PATHS``
     - ``/health,/healthz,/readiness,/liveness,/metrics``；应用 access 日志排除的路径及其子路径，逗号分隔；空字符串取消排除
   * - 无
     - ``LIGHTLLM_LOG_DIR``
     - 未设置；设置后同时写入按 PID 隔离的日志文件
   * - 无
     - ``LIGHTLLM_LOG_MAX_BYTES``
     - ``104857600``；单文件轮转阈值，单位为字节
   * - 无
     - ``LIGHTLLM_LOG_BACKUP_COUNT``
     - ``3``；每进程保留的轮转备份数，必须大于 0

只排查调度模块，不开启其他模块 DEBUG：

.. code-block:: bash

   export LIGHTLLM_LOG_LEVEL=INFO
   export LIGHTLLM_LOG_MODULE_LEVELS='{"lightllm.server.router":"DEBUG"}'

结构化输出，并抽样记录 1% 的请求完成摘要：

.. code-block:: bash

   export LIGHTLLM_LOG_FORMAT=json
   export LIGHTLLM_LOG_REQUEST_SAMPLE_RATE=0.01

``lightllm.requests`` 和 ``lightllm.access`` 在各自开启时使用 INFO，不受全局服务级别提高的影响；
可通过模块覆盖将它们调为 WARNING 等级以进一步过滤。
应用 access 日志由 LightLLM 中间件输出，避免再开启一份 Hypercorn access 日志。
配置服务使用 Hypercorn access；应用的路径过滤仅适用于 LightLLM 中间件。
用户提供的 Hypercorn 配置文件也可以自行定义 access 输出，需要避免重复配置。
Python warnings 通过统一日志输出；原生库 stdout/stderr 和 Hypercorn 自身日志不强制转换为 LightLLM JSON 格式。

输出类别与范围
--------------

* 服务日志：生命周期、最终配置、重要状态变化、降级与失败。
* ``lightllm.stats``：节点内 router 的 running、waiting、paused，以及各本地 DP 的 KV 使用率快照。
  所有摘要严格遵守配置间隔；持续空闲满一个间隔才输出 idle，恢复工作后按间隔输出；持续空闲不重复输出。
  多节点纯 TP 由主节点输出，避免镜像 router 重复记录。
  队列与 KV 是瞬时快照，吞吐使用引擎计数的窗口增量；长期吞吐和延迟分析使用 Prometheus 指标。
* ``lightllm.requests``：请求完成摘要，包含内部请求 ID、输入/输出 token 数、TTFT、总耗时和缓存/MTP 信息。
  normal/P/D 与 PD master 采用同一基础字段；P/D 摘要表示该角色内的处理结果。
  PD 分段请求的摘要保留对应分段的请求 ID，不代表完整客户端请求的汇总。
* ``lightllm.access``：HTTP 方法、路径、状态码和耗时；不记录 query、请求正文或认证头。

JSON 每行一个对象，额外字段可用于检索；异常 traceback 位于 ``exception`` 字段。
文本模式给多行消息和 traceback 的每一行补齐前缀。
各进程共享的模型配置、后端选择和 CUDA graph 初始化信息，在 TP 组代表 rank 上使用 INFO，
其他 rank 使用 DEBUG。各 worker 的 WARNING/ERROR 不应用这种过滤。
节点硬件、实际容量和 worker 故障仍需保留发生位置。

写日志的规则
------------

.. list-table::
   :header-rows: 1

   * - 级别
     - 使用条件
   * - DEBUG
     - 请求流转、批次选择、缓存分配/释放、KV 传输步骤、IPC 和配置推导细节。
   * - INFO
     - 启动阶段、最终配置、后端选择、模型加载完成、容量、ready、周期汇总和重要状态变化。
   * - WARNING
     - 非预期但能够恢复、重试或降级的情况；避免把正常取消、客户端断连和幂等清理当作警告。
   * - ERROR
     - 一次应完成的操作最终失败，例如推理失败或 KV 传输失败。
   * - CRITICAL
     - 服务或进程无法继续可靠工作，例如健康监控决定终止服务。

``logger.exception`` 是附带 traceback 的 ERROR，不是独立日志级别。
同一个捕获边界中不要连续使用 ``error`` 和 ``exception`` 记录同一个失败。
中间层继续抛出的异常应避免层层重复记录完整 traceback。
未知异常的 traceback 应由能明确处理失败的请求、后台任务或进程边界记录。

消息使用稳定的英文描述，新增事件使用 ``event`` 字段；请求/批次/传输日志携带相应 ID。
不要直接在日志中输出完整启动参数或对象 repr；启动摘要仅列出允许的运行参数，避免泄露密钥。
昂贵的调试参数需要在 ``isEnabledFor(logging.DEBUG)`` 判断之后构建，
使用 ``logger.debug("Batch: %s", batch.simple_log())`` 仍会提前执行 ``simple_log``。

升级说明
--------

原来的默认 DEBUG 改为 INFO。需要原有详细输出时显式设置 DEBUG，推荐只打开相关模块。
成功请求完成摘要默认采样 1%；HTTP access 按需显式开启。
设置日志目录后统一写入 ``process.<PID>.log``，不再额外创建 ``default.log``；
轮转备份为 ``process.<PID>.log.1`` 等。fork 子进程会切换到自己的 PID 文件。
备份数限制每个 PID 的文件数量，不会自动删除历史 PID 文件；部署系统需要配置目录保留周期。
容器部署建议由日志采集系统收集 stdout 并负责归档。

请求关联与非交互输出
--------------------

HTTP 入口保留客户端的 ``X-Request-ID``，未提供时生成 ID，并通过响应头返回。
ID 必须为不超过 128 字符的可见 ASCII 字符串，否则替换为生成的 ID。
``http_request_id`` 贯穿 access 与请求完成摘要；摘要中的 ``request_id`` 仍表示内部推理请求 ID。
关联 ID 即使在 access 关闭时也生效，并为并发和流式请求保持独立上下文。
日志来源文件和行号指向业务调用处。

权重加载和预热在非终端环境关闭 tqdm 进度条，改为阶段开始/完成事件。
JSON 输出和高于 INFO 的日志级别同样关闭进度条。

简洁前缀通过 ``LIGHTLLM_LOG_TP_SIZE``、``LIGHTLLM_LOG_DP_SIZE``、``LIGHTLLM_LOG_NNODES``
识别拓扑；这些内部变量由启动器自动设置，无需手动配置。

P/D 转发通过可选元数据传递 ``http_request_id``；旧消息仍可接收。
请求摘要省略为零的缓存字段，仅在启用 MTP 时输出 MTP 字段。
``timing_scope`` 为 ``end_to_end``、``prefill`` 或 ``decode``，标明 TTFT 和耗时的计时边界；
P/D 节点耗时不可直接当作客户端端到端耗时。
通信后端探测失败在 WARNING 保留简短原因和禁用动作，完整诊断在 DEBUG。

DEBUG 体积约束
--------------

常规 DEBUG 输出请求阶段、调度决定及原因、缓存状态变化；不输出 prompt、生成文本、
token 数组、grammar 全文或内存描述符。诊断事件最多 16 个字段；字符串最多 256 字符，
序列仅保留数量与前 8 项，嵌套对象仅保留类型或数量。

KV 成功传输不逐页记录握手和完成。发送/接收进程按 ``--log-stats-interval``
（默认 10 秒）聚合页数、物理传输字节、KV token 数和任务耗时合计，并保留最多 8 个请求 ID 样本。
首次完成立即输出；之后限频，空闲时补齐最后一个窗口，不输出空窗口。
``task_duration_ms_total`` 是各页面任务从创建到完成的耗时之和，不是请求端到端耗时。
接收端另保留 GPU 拷贝耗时合计；显式启用 NIXL 遥测时输出聚合遥测。
进程被强制终止时，尚未输出的最后一个统计窗口可能丢失。

TP 公共缓存状态和暂停/恢复事件只由组 leader 输出；各 rank 的独有异常仍保留。
反复发生的 P/D 连接故障保留首个完整 traceback；相同调用点、异常类型和简短原因的
后续故障每 10 秒输出重复计数。不同原因保留独立 traceback；去重状态最多 128 项。
请求和传输失败未套用此连接故障去重策略。

引擎窗口吞吐
------------

``engine.stats`` / ``engine.idle`` 展示 ``prefill_tok_s``、``output_tok_s``、
``completed_req_s``；JSON 另外保留 ``window_s`` 以及窗口原始计数
``prefill_tokens``、``output_tokens``、``completed_reqs``，便于核对。使用单调时钟的实际窗口长度计算速率，
默认仍按 10 秒输出；长请求未完成时也会报告已处理的 token。

每个 TP 组由负责发布进度的 leader 累加 CPU 共享计数，router 汇总本节点 DP 行。
纯跨节点 TP 仍仅主 router 输出。prefill 统计实际计算的输入 token，排除 GPU/CPU/磁盘
缓存恢复和 P/D 传入 KV；暂停后实际重算的输入会再次计入。生成统计只包含提交的输出
（含接受的 MTP token），不包含越过停止条件的 token 或模拟终止 token。
多候选、分段模式的完成计数按引擎内部请求计算，仅包含正常 stop/length 结束。

P 节点计算首 token，D 节点只计后续本地生成 token，不重复统计 P 传来的首 token。
因此两节点的生成计数相加才是分离推理的总生成量；不要将 master 的端到端请求摘要
再次相加。短请求即使在两个窗口之间完成，其计数仍会在下一窗口输出；稳定空闲后只输出
一次空闲转换。所有 token 计数在现有推理后处理同步之后更新，不额外同步 GPU。

INFO 启动日志以 HTTP /readiness 成功后的 service.ready 为准，包含地址和总启动时间。
运行时每个节点仅打印一份吞吐与队列摘要；KV 活跃占用与可复用缓存分开显示。
文本省略原始窗口计数，JSON 保留全部计数与时间窗口。P/D 内部请求摘要只在 DEBUG 输出。
失败摘要不采样，取消请求只在 DEBUG 输出；终止原因包含 stop、length、abort 或 error。
