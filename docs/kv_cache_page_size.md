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
基类初始化公共对象和状态，并统一维护上传提交协议及索引流等待；
各子类实现或复用存储、读写和重置逻辑。
扩展构造函数接收 `infer_req` 和必填的 `index_stream`，仅提取并保存固定的 `req_idx`、GPU 请求表行视图及索引流，
避免反向持有请求对象形成循环引用；索引流由外部选择，Helper 内部不查询默认流或捕获构造时的当前流。
CPU 缓存实现同时记录输入长度 `input_len`，helper 在共享请求初始化后、前缀匹配前构造。
工厂 `create_req_token_index(infer_req, index_stream)` 从请求参数中选择实现，流参数直接传给子类，
不再单独传 GPU 行和模式。在线服务和静态 benchmark 在统一创建入口传入 `torch.cuda.current_stream()`，
Helper 保存该流，后续不随调用点的当前流切换：

| `--req_token_index_mode` | 存储与优点 | 代价 |
| --- | --- | --- |
| `pin_mem_cache`（默认） | `PinMemCacheReqTokenIndex` 按请求申请普通 CPU token 缓存，上传时仅为待提交区间创建 pinned 快照；CPU 读取无需 D2H，GPU 更新可批量提交 | 每次上传增加一次 CPU 拷贝和临时 pinned 区间申请，需要维护上传进度 |
| `pin_mem_page_cache` | `PinMemPageCacheReqTokenIndex` 每个 KV 页仅保存一个 pinned CPU 首 token 索引，常驻索引项数约缩小为原来的 `1 / page_size` | CPU 读取和 GPU 上传时需要展开 token 索引，适用于 `page_size > 1` |
| `pure_gpu` | `PureGpuReqTokenIndex` 直接维护 GPU 请求表，无常驻 CPU 索引缓存 | CPU 读取需要 D2H 并等待结果，频繁的 CPU 索引操作可能更慢 |

三种模式均保留相同的 GPU 请求表，显存占用基本相同。追加后的所需长度为 `end = token_index_len + mem_indexes.numel()`。
完整 token 模式的 CPU 缓存首次按 `max(input_len + 2K, end)`
向上对齐到 1K token 申请，为输出预留空间；后续容量不足时，在本次所需长度 `end` 上多预留 2K token，再向上对齐到 1K，
容量上限为 GPU 请求表行长度。完整模式使用普通 CPU Tensor 保存每个 token，首次分配及扩容均不申请 pinned 内存。
页模式的常驻存储仍使用 `torch.empty(..., pin_memory=True, requires_grad=False)`，在首次追加时按 GPU 请求表的最大长度
一次性申请 `ceil(max_sequence_length / page_size)` 个页首索引，后续追加无需扩容或复制旧缓存。
页模式继承完整模式的 GPU 上传、释放快照和 reset 流程，覆盖 CPU 索引的存储和展开。
压缩依赖已有的页内连续索引约束；追加完整页，输入从下标 0 按 page_size 取页首。
页模式读取支持页内任意 token 区间：内部展开覆盖区间的完整页，再按页内偏移截取精确范围。
CPU 快照独立于压缩缓存，PD 尾段传输和最终 metadata 无需自行处理页对齐。
缓冲区在首次非空写入时按需申请，重置时保留输入长度，
丢弃已申请的常驻缓冲区引用。上传使用独立暂存区，其引用和异步传输依赖由上传路径管理。
完整模式的上传快照和页模式的展开暂存区统一使用基类 `ReqTokenIndex._alloc_return_pinned_tensor()`，
按 1M token（4 MiB）申请 int32 slab。函数内部通过函数属性 `return_buffer_state` 惰性保存完整 slab 和下一个空闲 offset，
两种模式共享同一分配进度，每次只创建返回值这一个切片。返回区间互不重叠，但可持有同一块底层 storage。
offset 只前进，不回绕；空间不足时换用新 slab 并从零开始，请求 reset 不重置共享 offset。
超出 slab 容量的申请独立分配，空申请不改变 slab 或 offset。旧 slab 的 storage 在所有切片引用和异步传输依赖结束后
由 PyTorch 回收。
纯 GPU 模式对普通 CPU 输入使用 `mem_indexes.pin_memory()` 转为独立 pinned 存储，已有 pinned 输入继续复用。
三种模式均不借用 KV allocator 的返回缓冲区作为请求常驻缓存。

统一接口为 `append_token_indexes(mem_indexes)`、`get_cpu_token_indexes(start, end)`、
`get_free_token_indexes(start, end)` 和 `get_gpu_token_indexes(start, end)`，均使用 token 单位，
追加接口将完整输入接到已记录索引的尾部，非空输入在三种模式下均断言为 CPU Tensor；空输入直接返回。
CPU 缓存模式直接复制到自身的常驻缓冲区，无需先 pin 输入；完整模式为普通 CPU 内存，页模式为 pinned 内存。
GPU 模式仅在输入未 pinned 时内部转换，已有 pinned 输入直接复用。
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
单独的 GPU getter 通过 `get_upload_indexes()` 准备独立 pinned 暂存区：完整模式复制待提交区间，页模式展开压缩索引。
上传源不借用常驻缓存，常驻缓存的扩容和重置不影响在途 H2D，也不需要在 reset 中同步 CUDA。
PyTorch `copy_(non_blocking=True)` 会登记 pinned 源的流使用，暂存区在 H2D 完成前不会被回收复用。

`get_upload_indexes()` 不接受区间参数，CPU 模式返回内部计算的
`[gpu_index_submitted_len, token_index_len)` 的 pinned token 索引。完整模式每次通过基类接口申请恰好覆盖该区间的切片，
同步复制普通 CPU 缓存中的索引后返回；连续调用返回互不重叠的快照，不覆写上次结果，不包含容量预留区间。
切片的逻辑长度只覆盖待上传索引，其底层 storage 通常为共享的 4 MiB slab。
页模式按需展开。两种模式均不改变提交进度；无待上传内容时返回空 tensor，
纯 GPU 模式返回 `None`。CPU 缓存模式的 GPU getter 校验 `end <= token_index_len` 后按需提交完整待上传区间，
纯 GPU 模式校验 `end <= gpu_index_submitted_len`；读取均不允许推进超出已有索引的提交进度。
批量入口不会替 GPU 模式补写提交进度；若缺少待上传数据，由后续拷贝操作报错，提交进度保持不变。

例如 CPU 缓存模式追加 64 个索引后，两个字段分别为 `64/0`；提交到 GPU 后为 `64/64`；
再追加 32 个索引后为 `96/64`，下次仅需提交 `[64,96)`。需要替换映射时，`reset()` 将两者清为 `0/0`，
重新追加 96 个完整索引后为 `96/0`，再从头提交 GPU 更新。纯 GPU 模式在追加接口中
直接提交写表，所以追加后两者相同。
`_get_classed_reqs` 返回前，backend 将本批索引对象交给 `ReqTokenIndex.submit_pending_uploads`，
并传入批量拷贝函数；backend 不再读写索引流、GPU 行和提交进度。
统一入口检查所有请求共用一个索引流，准备源、目标视图列表，在索引流提交拷贝，
仅在整批提交成功后更新提交进度。单请求 GPU getter 复用该协议，默认通过 PyTorch `copy_` 提交；
backend 路径使用 `cudaMemcpyBatchAsync`，直接提交索引对象准备的 pinned 快照，不再额外聚合或复制。
返回前仅在调用流不同于索引流时排入等待，没有新上传时也承接已有写入的依赖。
backend 不再持有上传流，也不逐条交接多条索引流。
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
该方式只保护引用生命周期，不阻止 CPU 主动改写上传源。完整模式的 pinned 快照和页模式的展开暂存区
均独立于常驻缓存，reset 和扩容不会改写它们；调用方在传输完成前不能主动修改返回的 pinned Tensor。
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
  混合注意力在 Prefill 尚未完成时，也可插入已保存大页 checkpoint 且 KV 已计算完成的最长前缀；
  缓存长度不超过目标缓存边界，其余 KV 统一释放；大页 checkpoint 随前缀移交缓存，重复 checkpoint 在插入时回收。
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

## 附录：选择批量异步 H2D 的原因

当前实现采用“CPU 写入 pinned 索引 → CUDA 异步 H2D → GPU kernel 读取 device 请求表”的流程。
选择显式拷贝的一项工程依据是：根据以往的使用经验，CPU 写入 pinned 内存后，直接让 GPU kernel
读取这块 host 内存的方案，曾在极端时序下观察到读取值异常。因此，本实现通过 CUDA 内存传输接口
建立明确的数据交接过程，模型 kernel 统一读取 GPU 上的 `req_to_token_indexs`。

排查上述现象时，需要考虑 CPU 存储缓冲、写合并、cache 一致性以及 CPU/GPU 访问顺序对数据可见性的影响。
“CPU 写入仍滞留于写缓冲，尚未对设备可见”是历史排查中考虑的可能原因，本文没有将其确认为所有平台上的根因。
具体的 cache 维护与可见性实现取决于硬件和驱动，正确性应依赖 CUDA API 的访问顺序与完成语义，
不假定每次异步拷贝都会执行固定的 CPU cache flush，或必须先把数据写回 DRAM 才能被设备正确读取。
CUDA 也支持正确同步的 mapped pinned memory；其文档明确要求应用协调 host/device 访问，避免读写竞争。
参见 [CUDA 13.0：Mapped Memory](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#mapped-memory)。

本实现的正确性由以下约束共同保证：

1. CPU 完成源索引写入后才提交 H2D；上传完成前不再改写已提交的源区间。
2. `cudaMemcpyBatchAsync` 使用 `cudaMemcpySrcAccessOrderStream`，源读取遵循上传流顺序；
   API 返回只表示已经提交，源内存须保持有效到实际传输结束。
3. GPU 消费流通过已有的流等待承接上传依赖，随后读取 device 请求表。
4. 原生 batch 路径由完成事件和后台任务持有 pinned 源引用；PyTorch `copy_` 回退路径
   由其 host allocator 跟踪异步使用，避免传输完成前回收复用源内存。

源访问顺序的定义参见 [CUDA 13.0 Runtime API：Memory Management](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-runtime-api/group__CUDART__MEMORY.html)。
批量接口用于将多个请求的索引上传合并为一次 API 提交，并直接使用各请求的 pinned 视图，避免额外的 CPU 聚合拷贝；
这些正确性约束同样适用于普通的异步 H2D，batch 接口本身不能替代源保活、访问排序和禁止在途改写。

PyTorch 文档介绍了使用 pinned CPU Tensor 配合 `to(..., non_blocking=True)` / `cuda(non_blocking=True)` 进行异步传输的方式，
并说明 pinned 源在传输完成前被 CPU 改写可能导致 GPU 数据错误。本实现遵循这一显式传输范式，
`cudaMemcpyBatchAsync` 则是多个索引区间的批量提交选择。
参见 [PyTorch 2.11：Use pinned memory buffers](https://docs.pytorch.org/docs/2.11/notes/cuda.html#use-pinned-memory-buffers)
和 [PyTorch：non_blocking 与 pin_memory 使用指南](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html#other-copy-directions-gpu-cpu-cpu-mps)。

## 附录：分离常驻索引与 pinned 上传缓冲区

### 设计动机与容量估算

请求索引有两种不同的生命周期：完整映射需要保存到请求结束或重建，用于前缀匹配、缓存管理和资源释放；
H2D 源只需保存到本次上传完成。将完整映射长期保存在 pinned 内存中，会让短期传输需要的锁页资源
随全部存活请求的索引容量增长。在当前离散 GPU 部署中，大量长期锁页会增加主机内存压力；
NVIDIA 也明确指出，过量分配 page-locked memory 会影响系统整体性能。
参见 [NVIDIA CUDA FAQ：Host/Device transfer](https://developer.nvidia.com/cuda/faq)。

面向 V4.1 这类模型的后续部署，需要提前考虑上下文长度、并发请求数和推理进程数同时增长的情况。
这里按部署容量分析，不假定 V4.1 的具体上下文上限、并发量或实测内存占用。
对每个推理进程，设已分配缓存的请求数为 `R`，请求 `i` 的完整 token 缓存容量为 `C_i`，
每个索引为 `int32`，则完整索引的逻辑字节数为 `4 × Σ C_i`。
如果每个请求各自申请 pinned 缓存，还需要计算底层分配块的容量取整。
本次核对的 PyTorch 2.11 host caching allocator 将申请字节数向上取整到 2 的幂，
因此这部分常驻 pinned 块约占 `Σ round_pow2(4 × C_i)`，扩容期间的新旧块和空闲缓存另计。
这是该版本 pinned allocator 的行为，不是所有 CPU Tensor 的通用分配规则。
参见 [PyTorch 2.11：host_memory_stats](https://docs.pytorch.org/docs/2.11/generated/torch.cuda.memory.host_memory_stats.html)。

例如，假设输入长 `128 × 1024` token，GPU 请求表行长度允许再预留 2K token，
当前完整缓存策略会申请 `130 × 1024` 个索引，逻辑大小为 520 KiB；若直接申请 pinned tensor，
对应块会取整为 1 MiB。每进程 128 个这样的存活请求、同机 8 个进程分别保存这些映射时，
仅常驻索引就会占用约 1 GiB pinned 块，尚未计入上传、旧块和其他模块。
多进程占用应按各进程的实际请求集合求和；TP 进程可能保存相同映射，DP 进程则应按各自请求数计算。

默认 `pin_mem_cache` 现将这部分长期存储改为普通 CPU Tensor，上述算例的常驻缓存逻辑大小合计为 520 MiB，
另加临时 pinned 上传区间。Linux 默认 CPU allocator 按请求大小进行对齐分配，
没有上述 host caching allocator 的统一 2 的幂取整步骤；系统 malloc 仍可能存在对齐、缓存和碎片开销。
当前代码自己的 1K 对齐、2K 预留以及扩容复制也仍然存在，普通 CPU 缓存不能当作零成本。
参见 [PyTorch 2.11：alloc_cpu 实现](https://github.com/pytorch/pytorch/blob/v2.11.0/c10/core/impl/alloc_cpu.cpp)。

### 为什么采用普通 CPU 常驻缓存和共享 pinned slab

完整模式的 `get_upload_indexes()` 只复制 `[gpu_index_submitted_len, token_index_len)`，
已经提交的历史索引和 CPU 缓存预留空间都不会重复进入上传快照。
这样保留了 CPU 读取无需 D2H 的能力，同时将 pinned 源的生命周期缩短到本次传输。
代价是每次上传增加一次同步 CPU 拷贝；前缀重建后需要重新上传完整映射，不能总按少量 decode 增量估算。

如果每次快照都单独调用 `torch.empty(..., pin_memory=True)`，小额上传仍会频繁进入 host allocator，
且不同大小的快照分别受容量取整影响。`_alloc_return_pinned_tensor()` 为常规申请维护一个
`1024 × 1024` 个 int32 元素的 slab，让多个请求取得互不重叠的区间，摊薄底层分配和取整开销。
4 MiB 是当前的小额分配与保留粒度取舍，并非所有模型、并发量和上传大小下的最优值。
空申请不创建 slab，单次超过 slab 容量时独立申请，避免把长期保留的工作块扩成一个大块。

完整模式将索引复制进这个区间，页模式将压缩页索引直接展开进这个区间；
两者共享相同的分配与生命周期约定。返回值是一次上传使用的快照，调用方不应把它保存为长期索引缓存。
多个快照可以共享 storage，但不得共享可写区间。offset 只前进，空间不足时换新 slab，
请求 reset 不回绕 offset，避免后续 CPU 写入覆盖尚未完成的 H2D 源。
PyTorch `copy_` 的 allocator 跟踪和原生 batch 路径的事件保活，继续负责传输结束前不回收源存储。

### 职责划分与状态归属

| 组件 | 负责的内容 |
| --- | --- |
| `PinMemCacheReqTokenIndex` | 普通 CPU 完整映射的追加、容量管理、快照生成和重置 |
| `PinMemPageCacheReqTokenIndex` | 页首映射的存储及 token 索引展开，复用公共上传协议 |
| `ReqTokenIndex` | 上传区间与提交进度、索引流依赖、临时 pinned 返回区间分配 |
| backend | 选择本批请求并调用统一提交入口 |
| `PinnedH2DBatchCopier` | CUDA 拷贝提交、原生接口适配及异步源保活 |

这种划分让常驻存储策略可以变化，而 backend 无需了解 CPU 缓存布局或维护上传进度。
`get_upload_indexes()` 只准备数据，`submit_pending_uploads()` 在整批提交成功后推进进度，
流完成与引用保活由传输层处理；缓存扩容和 reset 无需等待上传完成。
页模式继承完整模式时覆盖所有依赖压缩布局的追加、读取和分配操作，复用 GPU getter、释放入口及 reset。
保留 `PinMemCacheReqTokenIndex` 和 `pin_mem_cache` 名称以兼容现有调用和配置；名称中的 PinMem
现在对应上传缓冲区，常驻完整缓存的实际类型以实现、类说明和 CLI 帮助为准。

`return_buffer_state = (buffer, offset)` 放在分配函数对象上，由函数内部惰性建立和更新，
使两个字段的使用范围集中在唯一分配入口。它仍是进程内共享的持久状态，函数返回后不会消失，
移动字段本身不减少内存；主要收益来自常驻缓存改用普通 CPU 内存及上传区间共享 slab。
该入口依赖同一推理线程串行申请，未提供多线程分配保护；未来若增加并行申请线程，需重新设计同步或状态归属。

### 内存边界与后续评估

4 MiB 仅表示此分配函数当前持有的一块 slab，不能解释为整个进程的 pinned 内存上限。
上传缓冲区的实际占用还包括旧 slab 的存活切片、在途传输、超过 slab 的独立申请，以及 PyTorch 缓存的空闲块。
一个小切片也会保留整块 slab；GPU 上传积压、后台回收延迟或调用方长期持有返回值，都会延长整块存储的存活时间。
当前没有对同时存活的 slab 数量设置硬上限，也没有上传积压的内存背压。
KV allocator 的返回缓冲区和其他模块的 pinned 申请应独立计入整机预算，不能只统计这个函数。

`pin_mem_page_cache` 的常驻缓存目前仍使用 pinned 内存，并在首次追加时按 GPU 行长度 `S`
申请 `ceil(S / page_size)` 个页首索引，而不是按本次实际长度增长。
其常驻 pinned 块约为 `R × round_pow2(4 × ceil(S / page_size))`，上传暂存区另计。
`page_size=1` 时没有压缩收益；即使页较大，配置很大的最大序列长度且并发很多时，也仍需评估这部分预分配。
默认完整模式的改进不能代表页模式已经消除了长期锁页。若该项成为瓶颈，后续可评估普通 CPU 页缓存或按需增长，
并测量对页索引展开耗时的影响。

本次设计主要降低完整索引的长期 pinned 占用。首次长 prefill 或大规模重建时，普通 CPU 缓存与 pinned 快照会同时存在，
因此不保证主机总内存峰值下降，也不保证推理吞吐提升。
GPU `req_to_token_indexs` 仍为 token 级表，约占 `4 × (max_request_num + 1) × S` 字节；
模型权重、KV Cache 和该 GPU 表的显存都不因这项改动缩小。

评估 V4.1 等模型的实际部署时，应同时观察长 prefill、稳态 decode、前缀重建和上传积压场景，
按主机汇总各推理进程的 CPU RSS、pinned 块占用及峰值，并测量索引准备、上传和端到端延迟。
PyTorch 2.11 的 `host_memory_stats()` 中，`allocated_bytes.current` 包含活跃块和缓存块，
`active_bytes.current` 表示尚未变成可复用空闲块的容量，`num_host_alloc` 用于观察底层分配次数；
这些是进程级 host allocator 指标，不能直接归因到单个索引类，也不能用 GPU 的 `memory_allocated()` 替代。
各 bucket 的峰值是汇总近似值，取整与指标定义参见上述 PyTorch 文档。

仓库中的 [CPU/pinned 保活分配基准](../test/benchmark/benchmark_cpu_pinned_alloc_keepalive.py)
保留所有申请结果，可观察不同大小的直接分配耗时和 pinned 块取整；
[索引分配基准](../test/benchmark/benchmark_req_token_index_alloc.py) 比较直接申请与 slab 切片路径。
两者都不测量完整上传流程，前者也不写入普通 CPU Tensor 的全部页面，不能据此直接判断真实工作负载的 RSS 或推理性能。
现有 [索引测试](../unit_tests/common/test_req_token_index.py) 覆盖快照独立性、跨模式共享、换块及在途 H2D 源的生命周期。
