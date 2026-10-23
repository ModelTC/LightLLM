# KV Cache 多 Token 页设计

## 目标与不变量

启动参数 `--page_size N` 控制注意力 KV Cache 的物理页大小，默认值为 `1`。`N > 1` 时保持 token 级的
KV 存储和请求表，但资源的申请、缓存和释放以完整物理页为单位。

实现依赖以下不变量：

1. `cur_kv_len` 是请求已经写入有效 KV 的逻辑 token 数；`hold_kv_len` 是请求拥有的物理容量，始终满足
   `cur_kv_len <= hold_kv_len` 且 `hold_kv_len % page_size == 0`。
2. 一个物理页内的 KV 槽连续，页首索引可被 `page_size` 整除。请求表保存已持有页的全部 token 索引，
   包括尚未使用的尾部槽位。
3. `req_to_token_indexs` 保存请求拥有的完整页；创建 `InferState` 时根据请求索引和序列位置直接聚合本轮
   真实参与计算的 token，`ModelInput` 不携带物理 KV 索引，`InferState.mem_index` 不包含预留尾部。
4. Radix Cache 只插入、拆分、命中和淘汰完整页；不足一页的请求尾部在请求结束或暂停时整页回收。
5. `page_size=1` 与多 token 页使用相同的调度期预留和模型执行期索引选择路径。

页容量计算和整页申请由调度层在请求获准进入 Prefill/Decode batch 时完成；输入构造层只构造本轮输入，
真实推理索引由模型执行层从请求表选取。
`ReqManager` 只保留请求 ID、请求表及其原有生命周期职责，不提供 page allocator 接口。

## KV 页分配

`KvCacheAllocator(size, page_size)` 的 `size`、申请数量、释放索引和共享剩余容量均使用 token 单位。
内部 pinned `mem_state` 每页仅保存一个物理 token 页首索引，形如 `[0, page_size, 2 * page_size, ...]`，
游标按页计数。申请和释放必须覆盖完整页；释放时按 `page_size` 取页首，重新申请时展开回 token 索引。
分配结果直接写入 token 级 pinned 返回 tensor：`page_size=1` 使用 `copy_`，
其他页大小使用广播 `torch.add(..., out=...)`，不再分配中间展开 tensor。
返回缓冲区按需创建，单次申请超过 `1024 * 1024` 个 token 时独立分配；其他非空申请从
`1024 * 1024` 个 `int32` 元素的 pinned 缓冲区切片，仅保留未使用的尾部，不回绕覆盖旧结果。
尾部不足时新建缓冲区，旧切片自行持有原存储；空申请不创建大缓冲区。
`mem_state` 的容量随 `page_size` 缩小，分配器额外持有的返回缓冲区底层存储最多为 4 MiB，
不再按 KV 总容量预分配三倍 token 表。在途结果及 PyTorch pinned allocator 缓存的存储另计。

## 请求 Token 索引

`ReqManager` 只持有 GPU `req_to_token_indexs`，不再分配全局 CPU 页表。
每个 `InferReq` 通过成员 `token_index` 持有一个 `ReqTokenIndex`，实现位于 `server/router/model_infer/infer_req_ext1.py`。
索引操作直接使用 `req.token_index.append_token_indexes(...)` 等接口，`InferReq` 不再提供同名转发方法。
基类仅在构造函数中初始化公共对象和状态，其余方法只定义抽象接口；
各子类实现或复用读写、上传和重置逻辑。
扩展构造函数接收 `infer_req` 和必填的 `index_stream`，仅提取并保存固定的 `req_idx`、GPU 请求表行视图及索引流，
避免反向持有请求对象形成循环引用；索引流由外部选择，Helper 内部不查询默认流或捕获构造时的当前流。
CPU 缓存实现同时记录输入长度 `input_len`，helper 在共享请求初始化后、前缀匹配前构造。
工厂 `create_req_token_index(infer_req, index_stream)` 从请求参数中选择实现，流参数直接传给子类，
不再单独传 GPU 行和模式。在线服务和静态 benchmark 在统一创建入口传入 `torch.cuda.current_stream()`，
Helper 保存该流，后续不随调用点的当前流切换：

| `--req_token_index_mode` | 存储与优点 | 代价 |
| --- | --- | --- |
| `pin_mem_cache`（默认） | `PinMemCacheReqTokenIndex` 按请求直接申请 pinned CPU token 缓存；CPU 读取无需 D2H，GPU 更新可批量提交 | 占用 pinned CPU 内存，需要维护上传进度 |
| `pin_mem_page_cache` | `PinMemPageCacheReqTokenIndex` 每个 KV 页仅保存一个 pinned CPU 首 token 索引，常驻索引项数约缩小为原来的 `1 / page_size` | CPU 读取和 GPU 上传时需要展开 token 索引，适用于 `page_size > 1` |
| `pure_gpu` | `PureGpuReqTokenIndex` 直接维护 GPU 请求表，无常驻 CPU 索引缓存 | CPU 读取需要 D2H 并等待结果，频繁的 CPU 索引操作可能更慢 |

三种模式均保留相同的 GPU 请求表，显存占用基本相同。追加后的所需长度为 `end = token_index_len + mem_indexes.numel()`。
完整 token 模式的 CPU 缓存首次按 `max(input_len + 2K, end)`
向上对齐到 1K token 申请，为输出预留空间；后续容量不足时，在本次所需长度 `end` 上多预留 2K token，再向上对齐到 1K，
容量上限为 GPU 请求表行长度。两种 CPU 缓存模式均直接使用 `torch.empty(..., pin_memory=True, requires_grad=False)`
为每个请求独立申请常驻存储；完整模式保存每个 token，页模式在首次追加时按 GPU 请求表的最大长度
一次性申请 `ceil(max_sequence_length / page_size)` 个页首索引，后续追加无需扩容或复制旧缓存。
页模式继承完整模式的 GPU 上传、释放快照和 reset 流程，覆盖 CPU 索引的存储和展开。
压缩依赖已有的页内连续索引约束；追加完整页，输入从下标 0 按 page_size 取页首。
页模式读取支持页内任意 token 区间：内部展开覆盖区间的完整页，再按页内偏移截取精确范围。
CPU 快照独立于压缩缓存，PD 尾段传输和最终 metadata 无需自行处理页对齐。
缓冲区在首次非空写入时按需申请，重置时保留输入长度，
丢弃已申请的缓冲区引用，由 PyTorch 在引用和异步传输依赖结束后回收。
页模式上传时先将待提交区间展开到临时 pinned token 索引，使用 `PinMemPageCacheReqTokenIndex._alloc_pinned_tensor`
按 1M token（4 MiB）申请 int32 slab。该接口及共享状态仅属于页缓存类，保存完整 buffer 和下一个空闲位置 `_pinned_offset`，
每次只创建返回值这一个切片。
offset 只前进，不回绕；空间不足时换用新 slab 并从零开始，请求 reset 不重置共享 offset。
超出 slab 容量的申请独立分配，空申请不改变 slab 或 offset。旧 slab 的 storage 在所有切片引用和异步传输依赖结束后
由 PyTorch 回收。
纯 GPU 模式对普通 CPU 输入使用 `mem_indexes.pin_memory()` 转为独立 pinned 存储，已有 pinned 输入继续复用。
三种模式均不借用 KV allocator 的返回缓冲区作为请求常驻缓存。

统一接口为 `append_token_indexes(mem_indexes)`、`get_cpu_token_indexes(start, end)`、
`get_free_token_indexes(start, end)` 和 `get_gpu_token_indexes(start, end)`，均使用 token 单位，
追加接口将完整输入接到已记录索引的尾部，非空输入在三种模式下均断言为 CPU Tensor；空输入直接返回。
CPU 缓存模式直接复制到自身的 pinned 缓冲区，无需先 pin 输入；GPU 模式仅在输入未 pinned 时内部转换，已有 pinned 输入直接复用。
前缀匹配和索引重建路径直接传入 CPU 索引，不再自行 pin。读取接口均支持页内任意区间；提交给 allocator 的释放区间仍须覆盖完整页。
CPU getter 返回独立快照，缓存、PD 元数据和异步任务可跨请求释放持有；GPU getter 返回当前请求的
GPU 表视图，不应跨请求槽位复用持有。Helper 保存外部传入的索引流，GPU 索引读写固定在该流，
调用方保证 token 区间有效；GPU getter 返回前让调用流等待索引流，没有新增上传时也需承接已有写入的依赖。
调用流就是索引流时不额外等待；这里只排入 GPU 依赖，不阻塞 CPU，也不改变调用方的当前流。
模型和 KV 传输流在各自入口统一等待默认流的流程继续保留，Helper 不维护逐请求 CUDA 写事件。

释放专用接口 `get_free_token_indexes` 在 `pin_mem_cache` 模式返回 CPU 快照，在 `pure_gpu`
模式复用 GPU getter 的范围校验和流等待，再返回 GPU 请求表切片，省掉完整 token 索引的 D2H 快照。
流等待只保证已有索引写入就绪；重写请求槽位前，仍须在同一流按序消费旧视图，或让索引流等待消费流。
批量释放先合并视图，再归还请求 ID，暂停恢复也遵循此顺序。
allocator 仍维护 CPU 空闲页表，因此最终按 `page_size` 取页首并阻塞 D2H，返回时才能重新分配；
此接口不取消该同步，也不能代替缓存和元数据需要的独立 CPU 快照。

### 上传与源内存生命周期

`token_index_len` 是当前连续记录的 token 索引长度，包含页内预留槽位；每次追加从该位置开始，
并按输入元素个数增长。CPU 缓存扩容时保留
`[0, token_index_len)` 的已有映射。它与 CPU 缓存容量、已算好 KV 的长度 `cur_kv_len` 不同；
完整模式的 CPU token 容量为 `_cpu_buffer.numel()`，页模式为 `_cpu_buffer.numel() * page_size`。
`gpu_index_submitted_len` 是最新 GPU 索引写入已提交的前缀右端点，后续索引流操作可按序使用
`[0, gpu_index_submitted_len)`。它表示提交进度，不表示 GPU 传输已完成，也不是累计上传量；
CPU 缓存模式的追加接口只更新 CPU 索引和 `token_index_len`，不修改 GPU 提交进度。
需要替换前缀时，先保存仍需保留的索引（包括预留尾部），再调用 `reset()` 并按顺序追加，重建完整连续映射。
首次前缀匹配时索引为空；暂停恢复前已由统一释放流程重置，两处匹配入口直接追加命中前缀，不重复 `reset()`。
释放或暂停请求时也由索引扩展的 `reset()` 统一清零两个长度。
CPU 模式重置时将 `_cpu_buffer` 设为 `None`，下一次追加重新申请，随后从零开始提交 GPU 更新。
单独的 GPU getter 上传完整缓存视图或页缓存展开得到的暂存区：普通追加不会改写已有区间，扩容和重置使用新存储，
PyTorch `copy_(non_blocking=True)` 会登记 pinned 源的流使用，旧存储在 H2D 完成前不会被回收复用。
因此不再为单个请求上传额外申请 pinned 快照或执行一次 CPU 拷贝，也不需要在 reset 中同步 CUDA。

`get_upload_indexes()` 不接受区间参数，CPU 模式返回内部计算的
`[gpu_index_submitted_len, token_index_len)` 的 pinned token 索引，完整模式返回缓存视图，页模式按需展开，
不改变提交进度；无待上传内容时返回空 tensor，
纯 GPU 模式返回 `None`。CPU 缓存模式的 GPU getter 校验 `end <= token_index_len` 后按需提交完整待上传区间，
纯 GPU 模式校验 `end <= gpu_index_submitted_len`；读取均不允许推进超出已有索引的提交进度。
批量入口遇到存在未提交区间、但没有 CPU 上传源的情况会直接断言，不再替 GPU 模式补写提交进度。

例如 CPU 缓存模式追加 64 个索引后，两个字段分别为 `64/0`；提交到 GPU 后为 `64/64`；
再追加 32 个索引后为 `96/64`，下次仅需提交 `[64,96)`。需要替换映射时，`reset()` 将两者清为 `0/0`，
重新追加 96 个完整索引后为 `96/0`，再从头提交 GPU 更新。纯 GPU 模式在追加接口中
直接提交写表，所以追加后两者相同。
`_get_classed_reqs` 返回前，backend 收集各请求待上传的 pinned 视图，直接使用
`cudaMemcpyBatchAsync` 提交，不再申请聚合缓冲区或额外复制 CPU 索引。
同一批请求必须共用一个索引流，backend 对此进行断言；随后只准备源、目标视图列表，
在索引流上下文调用批量拷贝并更新提交进度。返回前仅在调用流不同于索引流时排入等待，
没有新上传时也承接已有写入的依赖。backend 不再持有上传流，也不逐条交接多条索引流。
原生 API 不接受 legacy 默认流，适配由 `PinnedH2DBatchCopier` 内部处理：按设备延迟创建并复用非默认上传流，
先让上传流等待默认流，提交后再让默认流等待上传流；非默认索引流直接提交，PyTorch 回退直接使用当前流。
这些操作只排入 GPU 依赖，不阻塞 CPU。
模型入口原有的推理流等待默认流继续保留；普通增量追加不等待上一批模型执行结束。

原生拷贝使用 `cudaMemcpySrcAccessOrderStream`，需要源内存在实际传输结束前有效且不变。
仅调用 `index_stream.wait_stream(upload_stream)` 不能保护 CPU 源的生命周期，因为 CPU 会立即返回。
`cuda_wrapper.py` 将原生接口开关、内部上传流、后台线程池和提交/保活方法统一封装在 `PinnedH2DBatchCopier` 中，
原有 `cuda_memcpy_pinned_h2d_batch_async` 函数保留为委托入口。原生提交成功后，在同一上传流上记录一个
`torch.cuda.Event(blocking=True)`，交给进程内复用的单线程 `ThreadPoolExecutor` 等待。
线程池在首次需要保活时加锁创建，工作线程在首次 submit 时启动；模块导入、空批次和 PyTorch 回退均不创建线程池，
也不因线程池初始化而初始化 CUDA。
后台任务持有 `list(src_pinned_tensors)`，独立于调用方可变的列表；源切片会保留完整 pinned storage，
即使调用方立即丢弃源 Tensor 或清空自己的列表，底层内存仍不会被回收复用。
后台 `event.synchronize()` 休眠等待并释放 GIL，完成后清空源列表，不向 Future 返回 Tensor；
主推理线程不等待后台任务，默认索引流由封装承接内部上传流的依赖，调用流按需等待索引流。
索引扩展不维护逐请求事件，backend 不维护保活列表，也不需要额外的小片段 H2D 或 C++ 扩展。
该方式只保护引用生命周期，不阻止 CPU 主动改写数据。请求缓存只追加，reset 和扩容换用新存储，
因此可直接保活缓存视图；上传期间不能改写已提交的区间，无需独立聚合快照。
单线程按提交顺序处理；多上传流共用线程池时可能延后回收，不改变 GPU 工作顺序。
正常进程退出时由线程池等待未完成任务，不取消保活任务。

事件创建、记录、后台任务提交及完成等待属于信任域；异常直接传播，不增加同步收尾、源引用隔离或恢复逻辑。
原生 batch API 不可用时回退到普通 PyTorch `copy_`，由其内部登记 host allocator，无需创建后台保活任务。

GPU 模式在追加接口中直接向构造时指定的索引流提交 GPU 更新，槽位复用后的新旧写入遵循同一流顺序；
追加后由调用流等待索引流，D2H 读取也固定在索引流并阻塞到 CPU 快照就绪。
已 pinned 的输入直接用于 H2D，省去中间快照的申请和 CPU 拷贝；普通 CPU 输入在追加接口内部 pin 后上传。
KV allocator 返回的 pinned 切片不会被后续申请覆盖；前缀匹配和重建路径直接传入 CPU 索引，由 Helper 按需处理。
PyTorch `copy_(non_blocking=True)` 内部向 host allocator 登记源 storage 的流使用，
因此调用方可立即释放输入引用，无需手工记录事件；但在 H2D 完成前不能主动改写源内容。

### 请求结束与跨流使用

暂停或释放时重置请求 Helper；前缀重映射通过重置后重新追加，重新建立 CPU 模式的上传进度。
PD Decode 传输失败时，等待全部任务返回后将 `cur_kv_len` 回退到原共享前缀长度（无共享前缀则为 0），
保留 `hold_kv_len` 和索引状态，由后续统一释放回收尾部页面，避免失败 KV 进入缓存。
CPU KV Cache 在卸载流上下文中调用 GPU getter；Helper 在请求索引流上准备数据，再让当前卸载流等待索引流，调用方无需额外同步。
纯 GPU 模式直接使用请求表视图，不再执行无谓的 D2H/H2D。
DP 发布前复用统一上传入口，所有 TP rank 通过默认流上的 NCCL barrier 确认索引可见后再发布。
diverse 模式会重写旧前缀，在重写入口统一让默认流等待推理流，防止覆盖上一批模型仍在读取的映射。
KV 卸载完成事件和请求延迟释放机制仍保留，它们负责实际 KV 任务和槽位生命周期。
模型预热不创建请求对象，直接将 allocator 返回的 token 索引拷入 GPU 请求表。
静态 benchmark 使用包含 `req_idx`、`req_manager`、`args` 和 `shm_req.input_len` 的轻量请求对象创建索引 Helper，
不进入真实 `InferReq` 的共享内存生命周期。
HOLD 请求直接在 GPU 表初始化为预留物理页的循环索引。

## 生命周期

- Prefill：请求通过调度后，按目标 KV 长度将容量向上对齐，新页一次申请并完整写入请求表。模型执行时再选择
  `[cur_kv_len, target_kv_len)` 对应的真实索引。
- Decode：请求通过调度后，若尾页仍有预留槽位则不再访问 allocator；容量不足时先补齐完整页再执行。
- Prefix Cache 命中：命中长度天然是整页，`cur_kv_len` 与 `hold_kv_len` 同时初始化为命中长度。
- 完成/暂停：完整逻辑页可进入 Radix Cache；重复页和未完成尾页按物理页展开后释放。
- Attention：FA3/FlashInfer 页表每项由物理 token 页首索引除以 `page_size` 得到，KV buffer 视为
  `[page_count, page_size, ...]`。

## 当前兼容范围

当前支持普通与分块 Prefill、Decode、动态 Prompt Cache、FA3/FlashInfer（含 MLA Decode）、MTP、
PD 分离、CPU KV Cache、DP Prompt Cache 拉取、DP Prefill Balance 和混合线性注意力。
diverse mode 仍只支持 `page_size=1`，启动阶段会对其他取值明确报错。

## 边界处理

- KV 总容量和请求表宽度分别向下、向上对齐，防止物理页越界。
- CUDA Graph 的 HOLD request 映射到额外保留的一整个物理页；请求表按
  `[HOLD, HOLD+1, ..., HOLD+page_size-1]` 循环填充，而不是重复同一个 token 索引。
- Radix 子节点用“首个完整 token 页”作为键，避免不同序列仅首 token 相同造成页级分支冲突。
- 非法的 `page_size < 1` 以及尚未支持的功能组合在模型加载前失败。
