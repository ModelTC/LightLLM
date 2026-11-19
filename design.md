当前的方案是这样的，

1. 新建一个转发进程 manager，名字要改得贴近实际含义，推理那边要上报的信息，通过zmq 发送给这个进程，这个进程再将信息转发给特定的pd master对应的类似过去up kv status的接口。

2. 允许 prefill 和 decode 节点的 httpserver 的数量大于 1， 然后各自建立与 pd master的连接， pd master 要可以识别对应的子连接，并添加相应的管理和分发功能。这个需要修改pd master中的一些管理分发逻辑。

## 模块

- `decode_trans_process`：Decode 推理侧。任务进入 `waiting_dict` 之后，把上报交给本机转发进程。
- `pd_report`：本机转发进程，管理类 `PdReportManager`。只接收推理进程的上报，并按目标 PD Master 转发出去。
- PD Master 的 `/pd_report`：只接收上报。不注册节点，不向这条连接发请求。
- prefill / decode 的 httpserver：每个 worker 自己连 PD Master 的 `/pd_register`，接收请求、回 token 和 prompt id。不接收 `PDUpKVStatus`。
- PD Master 的 `/pd_register`：登记 httpserver 子连接，选逻辑节点，再选一条子连接发送请求和后续控制消息。

## 上报

`decode_trans_process` 不把 `PDUpKVStatus` 推给 httpserver。

1. `decode_trans_process` 在 `waiting_dict` 插入之后，用本机 ZMQ PUSH 把 `(pd_master_node_id, PDUpKVStatus)` 发给 `pd_report`。第一个元素是目标 PD Master 的 id。
2. `pd_report` 用 PULL 收这条消息。它按 `pd_master_node_id` 找到自己到该 Master 的连接，把 `(ObjType.PD_UPLOAD_KV_STATUS, PDUpKVStatus)` 写到该 Master 的 `/pd_report`。
3. 该 Master 进程收包后调用现有的 `update_req_status`，唤醒正在等 KV 握手的请求。

`pd_report` 只在 Decode 节点启动，并且早于 `decode_trans_process`。它先 bind PULL。每个 PD Master 一条到 `/pd_report` 的 websocket，Master 掉线后只重连这一条。`pd_master_node_id` 对不上这条连接时丢掉并记日志。

httpserver 上不再有 `PDMasterForwarder`，也不再 bind 这条 ZMQ。

## 多个 httpserver

每个 httpserver worker 进程自己运行 `pd_handle_loop`，各自与每个 PD Master 建立 `/pd_register`。注册信息里已有的 `connection_id` 区分同一节点上的不同 worker。

PD Master 不把这些 worker 当成多台机器。

1. 逻辑节点仍按 `(node_id, mode, dp_index)` 管理。负载、前缀缓存和选点继续用现在的 `connection_key`，也就是 `client_ip_port/dp{dp_index}`。同一节点各 worker 报来的负载更新这一条逻辑节点。
2. 同一逻辑节点下，子连接放在 `sub_connections` 里，key 是注册报文里的 `connection_id`，value 就是这条连接的 websocket。逻辑节点上不保存 `connection_id` 和 websocket。用这个 id 查找、覆盖和摘除。相同 id 再次注册只换这个 websocket，旧 websocket 仍留给还没结束的请求。不同 id 是另一条 worker 连接，不会互相替换。
3. 进入 `fetch_pd_stream` 时，从该逻辑节点的活子连接里随机选一条 websocket。
4. `ReqStatus` 记下这次选中的 prefill websocket 和 decode websocket，两者都不能为空。后续的 `PD_REQ_DECODE_NODE_INFO` 写回 prefill 这条。找到 `ReqStatus` 时，ABORT 也只发到它记下的两条 websocket。找不到 `ReqStatus` 时，向给出的 prefill 节点和 decode 节点上的全部连接各发一次。
5. 一条子连接断开时，按它的 `connection_id` 从 `sub_connections` 里 pop 掉。字典里留下的都是存活连接。还记着这条 websocket 的请求按现有断连方式失败。同一节点的其他子连接继续接新请求。

Prefill 打开 DP 拆分连接时，每个 worker 仍为本机每个 DP rank 各连一条。分发只在相同 `dp_index` 的子连接之间进行。Decode 的 `dp_index` 为空，该节点所有 httpserver 子连接都可以接。

token、prompt id、忙和错误仍由持有该请求的 httpserver 经它自己的 `/pd_register` 送回，不经过 `pd_report`。
