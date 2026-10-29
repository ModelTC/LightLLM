"""注意力存储字段与传输位置的只读描述，不执行数据搬运。

TensorField 描述本地 Tensor；AxisSlice 描述某个轴的源、目标范围。
TransferRegion / StateTransferLayout 用于 State 与 CPU/PD 页之间的搬运；
DP 在对应 TP 分片间直接复制，不需要字段映射类型。

byte_start、byte_end 使用字节单位，表示左闭右开的区间；shape、轴切片起点和 length 使用元素单位。
层维度完整、同序对应，token/request 索引由每次搬运的调用参数提供。
frozen=True 只禁止重新赋值描述属性，不冻结被引用 Tensor 的数据。
KV 示例统一使用 head_dim=128。

PageKind 使用示例，选择 bounded attention 大页池：
    >>> page_kind: PageKind = "big"
    >>> assert page_kind == "big"
"""

from dataclasses import dataclass
from typing import Literal

import torch

# ========================================================================
# 一、状态页类别
# ========================================================================

# 选择 bounded attention 的页池及共享分配域，不表示 KV token 页或传输 buffer 的大小。
# "big" 对应 big_pages_pool 与 bounded_att_big_page_allocator；
# "small" 对应 small_pages_pool 与 bounded_att_small_page_allocator。
# 两个域中的同一个 slot_id 指向不同状态页。调用方先在对应 allocator 分配槽位，
# 再将类别传给 save_bounded_att_pages / restore_bounded_att_pages 等操作，例如：
# manager.save_bounded_att_pages([0], page_kind="big", slot_ids=[3])
# 上例假定请求槽位 0 的已接受长度为 128，且大页槽位 3 已由调用方分配。
PageKind = Literal["big", "small"]


# ========================================================================
# 二、本地存储字段与轴范围
# ========================================================================


@dataclass(frozen=True)
class TensorField:
    """为本地 Tensor 补充轴语义，供 State 内部布局计算及搬运使用。

    创建描述只引用已有 Tensor，不分配或复制其底层存储。具体 State 按固定布局组织
    k、v、动态量化 scale、Conv、SSM 等搬运字段，字段名称由所在字典的 key 提供；
    packed buffer 的字段可使用切片视图。

    示例：本地 K 数据有 2 层、8 个 token、2 个 heads，每个 head 有 128 个元素。
        >>> k = torch.empty((2, 8, 2, 128), dtype=torch.float16)
        >>> field = TensorField(
        ...     tensor=k, axes=("layer", "token", "head", "dim")
        ... )
        >>> assert field.tensor is k
    """

    # 实际存储或其视图，提供 dtype、shape、stride、device 等物理信息。
    # 可引用非连续的 packed/分层存储视图，搬运代码沿用其真实 stride。
    tensor: torch.Tensor

    # 按 Tensor 维度顺序列出语义名称，数量等于 tensor.ndim，名称不能重复。
    # 例如 token/head 用于 KV 选取和 TP 切片，request/page 用于运行态或页池描述；
    # 选择某个请求或状态页后，搬运视图会移除已选定的 request/page 轴。
    axes: tuple[str, ...]


@dataclass(frozen=True)
class AxisSlice:
    """某一轴上的连续范围映射，起点与长度均以元素为单位。

    TransferRegion 中的方向固定为本地 State 到外部页，读取时反向应用。

    示例：本地 head[0:2] 写到全局传输区域 head[2:4]，读回时反向复制。
        >>> head_slice = AxisSlice(axis="head", src_start=0, dst_start=2, length=2)
        >>> assert head_slice.dst_start + head_slice.length == 4
    """

    # 要切片的轴名，必须同时存在于源、目标字段中，例如 head、channel、dim。
    # layer 固定完整复制，token 由本次传输数量控制，这两个轴不放入 AxisSlice。
    axis: str

    # 源轴的起始元素索引，包含该位置且必须非负。
    # TransferRegion 中本地字段的位置。
    src_start: int

    # 目标轴的起始元素索引，包含该位置且必须非负。
    # TransferRegion 中外部页区域的位置。
    # 用于将本地 TP 分片放到全局 head/channel 的对应位置。
    dst_start: int

    # 搬运的元素数量，必须为正；源和目标的半开区间均为 [start, start + length)。
    # 它是该轴上的长度，不是字节数，也不是整个 Tensor 的元素总数。
    length: int


# ========================================================================
# 三、CPU/PD 外部页布局
# ========================================================================


@dataclass(frozen=True)
class TransferRegion:
    """描述 State 内一个字段与 CPU/PD 外部页区域之间的拷贝映射。

    由 StateTransferLayout.regions 组织，同一字段可以对应多个区域映射。
    区域的 shape 覆盖外部完整数据，axis_slices 选择当前 rank 的子范围。
    本地和外部区域的层数必须一致，并按相同存储顺序完整复制。

    示例：外部 K 区域有 4 个全局 heads，head_dim=128，当前 rank 持有后 2 个 heads。
    整个区域占 2 * 8 * 4 * 128 * 2 = 16384 字节（16 KiB），位于页内 [16384, 32768)。
        >>> k_region = TransferRegion(
        ...     field="k",
        ...     byte_start=16384, byte_end=32768,
        ...     dtype=torch.float16,
        ...     axes=("layer", "token", "head", "dim"),
        ...     shape=(2, 8, 4, 128),
        ...     axis_slices=(
        ...         AxisSlice(axis="head", src_start=0, dst_start=2, length=2),
        ...         AxisSlice(axis="dim", src_start=0, dst_start=0, length=128),
        ...     ),
        ...     write_enabled=True,
        ... )
    """

    # 本地搬运字段字典的 key，用来定位 k/v/scale/conv/ssm 等 Tensor。
    # 同一字段可出现在多个 region 中，例如 Conv 的 Q/K/V 使用不同 channel 切片。
    field: str

    # 字段区域起点，包含该字节，必须按 dtype 的元素大小对齐。
    # State 的 get_cpu/pd_transfer_regions 返回相对本 State 起点的区间；
    # 管理器将起止位置同时加上 State 的页内起点，最终区间统一相对整页。
    # CPU/PD 都从页集合的第一维选择物理页；字段区间不包含物理页索引。
    byte_start: int

    # 字段区域终点，不包含该字节，与 byte_start 使用相同的坐标基准。
    # byte_end - byte_start 必须等于 prod(shape) * dtype.itemsize，
    # 仅包含字段数据；字段之间的对齐空隙不计入该区间。
    byte_end: int

    # 外部区域按此类型解释原始字节，需与本地搬运字段的 dtype 一致。
    # packed 字段可分别使用 uint8、float16 等实际类型；此描述不执行类型转换。
    dtype: torch.dtype

    # 外部区域各维度的语义顺序，与 shape 一一对应且名称唯一。
    # 可与本地 Tensor 的轴顺序不同，搬运时按名称对齐；本地视图需具有相同轴集合。
    axes: tuple[str, ...]

    # 完整外部区域的各轴元素数量，不包含外部页池的 page 轴，也不是本 rank 的局部形状。
    # 例如全局 4 heads、本地 2 heads 时，这里的 head 维仍为 4。
    # 区域连续排列，stride 由 shape 和 dtype 推导；区域末尾必须位于当前页内。
    shape: tuple[int, ...]

    # 描述本地字段到外部区域的轴范围，每个非 layer/token 轴必须恰好出现一次。
    # 无分片的轴也显式填写完整范围，例如 dim: 0 -> 0、length=head_dim。
    # 读回时交换源、目标起点；同一组切片应用于全部层和本次有效 token。
    axis_slices: tuple[AxisSlice, ...]

    # 当前 TP rank 是否负责向外部区域写入；False 不影响从该区域读回。
    # KV head 有 TP 副本时，每组只选一个 rank 写入；复制的 indexer tail 只由 rank 0 写。
    # 标记属于 region，因为同一 State 内各字段的 TP 分片/复制规则可能不同。
    write_enabled: bool = True


@dataclass(frozen=True)
class StateTransferLayout:
    """描述一个 State 的 CPU/PD 传输布局，按字段组织 TransferRegion。

    每个 layout 通过 resource_id 绑定一个 State；regions 描述该 State 各字段
    与外部页之间的映射，具体轴范围由各 region 的 AxisSlice 描述。
    byte_start / byte_end 记录本 State 在联合页中的完整预留区间，包含自身对齐空间。
    CPU/PD 根据页集合的行选择物理页；PD 通过显式 pd_page_index 指定目标页。
    实际页容量由 manager 统一计算，可以大于当前 State 的终点。
    示例：一张 64 KiB 的页中，本 State 的 K/V 各占 16 KiB，
    分别位于 [16 KiB, 32 KiB) 和 [32 KiB, 48 KiB)，前后空间可属于其他资源。
        >>> from dataclasses import replace
        >>> k_region = TransferRegion(
        ...     field="k", byte_start=16384, byte_end=32768, dtype=torch.float16,
        ...     axes=("layer", "token", "head", "dim"), shape=(2, 8, 4, 128),
        ...     axis_slices=(
        ...         AxisSlice(axis="head", src_start=0, dst_start=2, length=2),
        ...         AxisSlice(axis="dim", src_start=0, dst_start=0, length=128),
        ...     ),
        ... )
        >>> layout = StateTransferLayout(
        ...     resource_id="main_full_att_fp16_state",
        ...     tokens_per_page=8,
        ...     byte_start=16384, byte_end=49152,
        ...     regions=(k_region, replace(k_region, field="v", byte_start=32768, byte_end=49152)),
        ... )
        >>> assert layout.regions[1].byte_end == layout.byte_end == 49152
    """

    # 当前 State 的规范身份，与管理器成员名、state.resource_id 一致。
    # 用于从联合页布局字典中选择对应资源，区分 main/draft 及不同存储方案。
    resource_id: str

    # 整张外部页可容纳的 token 数，不是本次实际传输数量；含 KV 的页使用正整数。
    # CPU/PD 分别使用各自的全局页配置，页容量不随本次传输长度改变。
    # CPU 的 KV + bounded 联合页中，bounded layout 也沿用该 token 容量。
    # 纯运行态页（如 PD 的 att_state 页）为 None，表示按快照而非 token 序列搬运。
    tokens_per_page: int | None

    # 本 State 在外部页中的起始字节，包含该位置，始终相对整页起点。
    # 多个 State 依次排列，后一 State 的起点为前一 State 的终点。
    # regions 已使用页内坐标，搬运时不再重复加上此起点。
    byte_start: int

    # 本 State 在外部页中的结束字节，不包含该位置；涵盖全部字段及本 State 的对齐、预留空间。
    # byte_end - byte_start 是本 State 的占用，不是整张物理页的跨度。
    # 同类布局的最大 byte_end 表示整组 State 的占用；CPU 物理页再按 16 字节对齐，
    # PD 物理页由 manager.get_pd_page_bytes() 取 KV 与完整运行态布局占用的最大值。
    byte_end: int

    # 当前 State 参与传输的字段区域；其他 State 的区域放在它们各自的 layout 中。
    # 包含所需的 KV、动态量化参数或运行态字段，所有偏移均相对同一张完整外部页。
    regions: tuple[TransferRegion, ...]
