"""用量复核核心判定逻辑。

对每段相邻读数 (r1, r2)，真实增量为 delta = (r2 - r1) + k * modulus，
其中 k 为整圈回绕次数（非负整数）。约束条件：

    delta >= 0                          （非负增量）
    delta <= max_rate * elapsed_seconds （速率上限）

所有满足约束的 k 构成一个连续整数区间 [k_min, k_max]：

    k_min = 满足 diff + k*M >= 0 的最小非负整数
    k_max = 满足 diff + k*M <= max_delta 的最大整数

判定：
    k_max <  k_min  -> 无解 (no_solution)
    k_max == k_min  -> 唯一解，取该 k
    k_max >  k_min  -> 多解 (ambiguous)，不猜测圈数

速率上限使用 Fraction 精确计算，避免浮点边界误差。

账单总量对账（reconcile_total）：
    设账单总用量为 B，各段首尾读数差之和望远镜相消为首尾读数差：
        B = Σ delta_i = Σ diff_i + M * Σ k_i = (r_last - r_first) + M * K
    故总回绕次数 K = (B - (r_last - r_first)) / M 必须为非负整数
    （账单总量与模数不相容），且满足 Σ k_min <= K <= Σ k_max
    （超出可行总量）。固定 K 后，各段仍可行的圈数范围完全由整数区间
    运算给出，无需枚举组合：

        lo_i' = max(lo_i, K - Σ_j hi_j + hi_i)
        hi_i' = min(hi_i, K - Σ_j lo_j + lo_i)

    lo_i' == hi_i' 的段圈数唯一，可确定用量；其余段仅保留范围，不猜测。

换表账单复核（reconcile_meter_total）：
    旧表、新表两段模数 M1、M2 可不同。两段用量分别为
        u1 = d1 + k1*M1, u2 = d2 + k2*M2
    （d 为段内首尾读数差），账单总量 B 要求 u1 + u2 = B，即
        M1*k1 + M2*k2 = N，N = B - d1 - d2
    该二元一次丢番图方程有整数解当且仅当 g = gcd(M1, M2) 整除 N
    （total_incompatible_modulus）。有解时由扩展欧几里得取特解
    (k10, k20)，全部整数解为单参数族：
        k1 = k10 + (M2/g)*t, k2 = k20 - (M1/g)*t   （t 为任意整数）
    两段各自的可行圈数区间 [lo_i, hi_i] 把 t 夹到连续整数区间
    [t_min, t_max]（四个 ceil/floor 不等式取交）；t_min > t_max 即账单
    超界（total_out_of_range）。全程仅整数区间运算，不枚举圈数组合。
    t_min == t_max 时两段圈数同时唯一；否则按等差数列端点给出各段用量
    上下界（每个可行 t 均严格满足 u1 + u2 = B）。

漏采读数推断（solve_missing_reading）：
    漏采时刻把账期切成左、右两段，容量分别为 floor(速率 × 各段秒数)。
    账单总量 B 须满足 B = diff + K*M（diff = 末读数 - 首读数，K >= 0）；
    同余且 B >= 0 时 K 自动非负。左侧用量 x 的可行闭区间为
    [max(0, B - C_R), min(C_L, B)]，候选读数 (首读数 + x) mod M 在模 M
    环上形成连续弧：跨零点拆段、覆盖全模数时返回全域。

最小统一速率上限（solve_minimum_rate）：
    不预设速率上限，在账单总量 B = Σ(d_i + k_i*M)（k_i >= 0 整数）约束下，
    求使各段速率 (d_i + k_i*M)/t_i 均不超过它的最小统一速率 R*。
    令 a_i = d_i + k_min_i*M ∈ [0, M) 为各段最小非负用量、
    E = K - Σk_min_i 为待分配的额外圈数，则

        R* = max( max_i a_i/t_i,  V_E )

    其中 V_E 为全体"抬升门槛" (a_i + m*M)/t_i（m >= 1）中第 E 小者：
    分配 E 圈必然使用 E 个门槛，取最小的 E 个即最优。V_E 由精确二分
    求得（候选值两两之差 >= 1/max(t_i)^2，收窄后按整除条件取出唯一
    候选），全程整数/分数运算，不枚举圈数组合。定出 R* 后各段可行圈数
    区间 [k_min_i, k_min_i + floor((R*·t_i - a_i)/M)] 再交由
    reconcile_total 以账单总量收紧。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from fractions import Fraction
from typing import Optional, Sequence, Tuple

STATUS_OK = "ok"
STATUS_NO_SOLUTION = "no_solution"
STATUS_AMBIGUOUS = "ambiguous"
STATUS_CONFLICT = "conflict"

# 账单总量对账冲突原因
REASON_TOTAL_INCOMPATIBLE_MODULUS = "total_incompatible_modulus"
REASON_TOTAL_OUT_OF_RANGE = "total_out_of_range"
# 跨日分配冲突原因：不存在可行的子段整数分配
REASON_NO_FEASIBLE_ALLOCATION = "no_feasible_allocation"
# 漏采读数推断冲突原因：漏采时刻两侧容量之和小于账单总量
REASON_INSUFFICIENT_CAPACITY = "insufficient_capacity"


@dataclass(frozen=True)
class SegmentSolution:
    """单个相邻读数区间的判定结果。"""

    index: int
    status: str
    elapsed_seconds: int
    # 唯一解时填充
    wraps: Optional[int]
    usage: Optional[int]
    # 可行回绕次数区间 [min_wraps, max_wraps]；无解时 max_wraps < min_wraps
    min_wraps: int
    max_wraps: int
    max_allowed: Fraction  # 该区间允许的最大增量（速率上限 * 秒数）


def solve_segment(
    index: int,
    start_value: int,
    end_value: int,
    elapsed_seconds: int,
    modulus: int,
    max_rate_per_second: Fraction,
) -> SegmentSolution:
    """判定一段相邻读数的回绕次数是否唯一。"""
    diff = end_value - start_value
    max_allowed = max_rate_per_second * elapsed_seconds

    # k_min: diff + k*M >= 0 的最小非负整数 k
    if diff >= 0:
        k_min = 0
    else:
        # ceil(-diff / M)，此时 0 < -diff < M，结果恒为 1
        k_min = (-diff + modulus - 1) // modulus

    # k_max: diff + k*M <= max_allowed 的最大整数 k
    k_max = math.floor((max_allowed - diff) / modulus)

    if k_max < k_min:
        status, wraps, usage = STATUS_NO_SOLUTION, None, None
    elif k_max == k_min:
        status, wraps = STATUS_OK, k_min
        usage = diff + k_min * modulus
    else:
        status, wraps, usage = STATUS_AMBIGUOUS, None, None

    return SegmentSolution(
        index=index,
        status=status,
        elapsed_seconds=elapsed_seconds,
        wraps=wraps,
        usage=usage,
        min_wraps=k_min,
        max_wraps=k_max,
        max_allowed=max_allowed,
    )


@dataclass(frozen=True)
class NarrowedSegment:
    """账单总量约束后，单段仍可行的回绕次数整数区间。"""

    index: int
    diff: int
    min_wraps: int
    max_wraps: int

    @property
    def unique(self) -> bool:
        """圈数是否已被总量约束唯一确定。"""
        return self.min_wraps == self.max_wraps

    @property
    def wraps(self) -> Optional[int]:
        return self.min_wraps if self.unique else None

    def usage(self, modulus: int) -> Optional[int]:
        """圈数唯一时该段确定用量，否则为 None。"""
        return self.diff + self.min_wraps * modulus if self.unique else None

    def usage_bounds(self, modulus: int) -> Tuple[int, int]:
        """该区间用量的最小/最大可能值。"""
        return (
            self.diff + self.min_wraps * modulus,
            self.diff + self.max_wraps * modulus,
        )


@dataclass(frozen=True)
class ReconcileResult:
    """账单总量对账结果。"""

    status: str  # STATUS_OK 或 STATUS_CONFLICT
    total_wraps: Optional[int]  # 总量隐含的总回绕次数 K（不相容时为 None）
    narrowed: Tuple[NarrowedSegment, ...]
    conflicts: Tuple[str, ...]  # REASON_TOTAL_* 原因
    feasible_min_total: int
    feasible_max_total: int


def reconcile_total(
    diffs: Sequence[int],
    wrap_bounds: Sequence[Tuple[int, int]],
    total_usage: int,
    modulus: int,
) -> ReconcileResult:
    """以账单总用量约束整条读数序列的回绕次数总和。

    :param diffs: 各段首尾读数差 r2 - r1
    :param wrap_bounds: 各段在非负增量与速率上限下的可行圈数区间
        ``(k_min, k_max)``，调用方需保证每段均可行（k_max >= k_min）
    :param total_usage: 账单总用量（非负整数）
    :param modulus: 计数器模数
    :return: 对账结果；可行时 ``narrowed`` 给出各段收紧后的圈数区间
    """
    if len(diffs) != len(wrap_bounds):
        raise ValueError("diffs 与 wrap_bounds 数量必须一致")

    lows = [lo for lo, _ in wrap_bounds]
    highs = [hi for _, hi in wrap_bounds]
    sum_low, sum_high = sum(lows), sum(highs)
    end_minus_start = sum(diffs)  # 望远镜相消：末读数 - 首读数

    feasible_min_total = end_minus_start + modulus * sum_low
    feasible_max_total = end_minus_start + modulus * sum_high

    numerator = total_usage - end_minus_start
    conflicts = []
    # 总量必须与首尾读数差模 M 同余（等价于总回绕次数为非负整数）
    if numerator % modulus != 0:
        conflicts.append(REASON_TOTAL_INCOMPATIBLE_MODULUS)
        total_wraps: Optional[int] = None
    else:
        total_wraps = numerator // modulus
        if not (sum_low <= total_wraps <= sum_high):
            conflicts.append(REASON_TOTAL_OUT_OF_RANGE)

    if conflicts:
        return ReconcileResult(
            status=STATUS_CONFLICT,
            total_wraps=total_wraps,
            narrowed=(),
            conflicts=tuple(conflicts),
            feasible_min_total=feasible_min_total,
            feasible_max_total=feasible_max_total,
        )

    # K 固定：其余各段圈数之和为 K - k_i，由其余段的区间和反推出 k_i 的范围
    narrowed = []
    for i, (lo, hi) in enumerate(wrap_bounds):
        other_high = sum_high - hi  # 除 i 外各段最大圈数之和
        other_low = sum_low - lo  # 除 i 外各段最小圈数之和
        new_lo = max(lo, total_wraps - other_high)
        new_hi = min(hi, total_wraps - other_low)
        narrowed.append(
            NarrowedSegment(
                index=i,
                diff=diffs[i],
                min_wraps=new_lo,
                max_wraps=new_hi,
            )
        )

    return ReconcileResult(
        status=STATUS_OK,
        total_wraps=total_wraps,
        narrowed=tuple(narrowed),
        conflicts=(),
        feasible_min_total=feasible_min_total,
        feasible_max_total=feasible_max_total,
    )


# ---------------------------------------------------------------------------
# 跨日用量上下界（daily-range）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DaySubsegment:
    """读数区间按 UTC 午夜切出的一个子段（完全落在单个 UTC 自然日内）。"""

    day: date
    seconds: int
    capacity: int  # floor(max_rate_per_second * seconds)


@dataclass(frozen=True)
class DailySegmentResult:
    """一个读数区间跨日切分后的可行分配结果。"""

    index: int
    feasible: bool
    # 该区间可行的回绕次数范围；无解时 max_wraps < min_wraps
    min_wraps: int
    max_wraps: int
    subsegments: Tuple[DaySubsegment, ...]
    # 每个子段的精确边际上下界（与 subsegments 一一对应）；无解时为空
    min_usages: Tuple[int, ...]
    max_usages: Tuple[int, ...]

    @property
    def total_capacity(self) -> int:
        return sum(s.capacity for s in self.subsegments)


def split_utc_days(start: datetime, end: datetime) -> Tuple[Tuple[date, int], ...]:
    """将 [start, end) 按 UTC 午夜切成 ``(UTC 日期, 该日子段秒数)`` 序列。

    端点均为秒精度、带时区的严格递增时间戳。子段在交界处首尾相接
    （零长度子段被丢弃），秒数之和恰为区间总时长。
    """
    start = start.astimezone(timezone.utc)
    end = end.astimezone(timezone.utc)
    pieces = []
    cursor = start
    while cursor < end:
        next_midnight = datetime.combine(
            cursor.date() + timedelta(days=1), time.min, tzinfo=timezone.utc
        )
        boundary = min(end, next_midnight)
        seconds = int((boundary - cursor).total_seconds())
        if seconds > 0:
            pieces.append((cursor.date(), seconds))
        cursor = boundary
    return tuple(pieces)


def solve_daily_segment(
    index: int,
    start_value: int,
    end_value: int,
    start: datetime,
    end: datetime,
    modulus: int,
    max_rate_per_second: Fraction,
) -> DailySegmentResult:
    """求单个读数区间跨各 UTC 日子段用量的精确边际上下界。

    区间真实增量为 ``delta = diff + k*M``（diff = r2 - r1，k >= 0 整数），
    切日子段容量为 ``c_j = floor(rate * s_j)``。需存在非负整数分配
    ``(x_1, ..., x_n)`` 满足 ``0 <= x_j <= c_j`` 且 ``Σ x_j = delta``。

    该类分配存在当且仅当 ``delta <= C``（``C = Σ c_j``）：因为所有容量为
    整数，``[0, C]`` 内任意整数总量都可逐子段填满。故可行 k 仍为连续
    整数区间 ``[k_min, k_max]``，其中

        k_min = ceil(max(0, -diff) / M)
        k_max = floor((C - diff) / M)

    对固定子段 j，给定可行 k 时其边际范围为

        max(0, delta - (C - c_j)) <= x_j <= min(c_j, delta)

    下界随 delta 单调增，故全区间最小值在 k_min 处取得；上界随 delta
    单调增，最大值在 k_max 处取得。两端点均确实可由完整分配达到
    （箱型可行域与固定总量超平面之交的逐分量极值）。
    """
    diff = end_value - start_value

    subsegments = tuple(
        DaySubsegment(
            day=day,
            seconds=seconds,
            # Fraction 精确取整，避免浮点边界误差
            capacity=math.floor(max_rate_per_second * seconds),
        )
        for day, seconds in split_utc_days(start, end)
    )
    total_capacity = sum(s.capacity for s in subsegments)

    k_min = 0 if diff >= 0 else (-diff + modulus - 1) // modulus
    k_max = (total_capacity - diff) // modulus

    if k_max < k_min:
        return DailySegmentResult(
            index=index,
            feasible=False,
            min_wraps=k_min,
            max_wraps=k_max,
            subsegments=subsegments,
            min_usages=(),
            max_usages=(),
        )

    delta_min = diff + k_min * modulus
    delta_max = diff + k_max * modulus

    min_usages, max_usages = [], []
    for s in subsegments:
        other_capacity = total_capacity - s.capacity
        # x_j >= delta - 其余子段容量之和；delta 取最小时下界最小
        low = max(0, delta_min - other_capacity)
        # x_j <= delta（且不超本段子容量）；delta 取最大时上界最大
        high = min(s.capacity, delta_max)
        min_usages.append(low)
        max_usages.append(high)

    return DailySegmentResult(
        index=index,
        feasible=True,
        min_wraps=k_min,
        max_wraps=k_max,
        subsegments=subsegments,
        min_usages=tuple(min_usages),
        max_usages=tuple(max_usages),
    )


# ---------------------------------------------------------------------------
# 换表账单复核（meter-change）：跨两表、模数可不同
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MeterSegmentNarrowed:
    """换表账单总量约束后，一段（旧表或新表）仍可行的圈数区间。"""

    meter: str  # "old" 或 "new"
    diff: int
    modulus: int
    min_wraps: int
    max_wraps: int
    wraps_step: int  # 可行圈数序列的公差（恒正）

    @property
    def unique(self) -> bool:
        """该段圈数是否已被账单总量唯一确定。"""
        return self.min_wraps == self.max_wraps

    @property
    def wraps(self) -> Optional[int]:
        return self.min_wraps if self.unique else None

    def usage_bounds(self) -> Tuple[int, int]:
        """该段用量的最小/最大可能值（端点均可达到）。"""
        return (
            self.diff + self.min_wraps * self.modulus,
            self.diff + self.max_wraps * self.modulus,
        )


@dataclass(frozen=True)
class MeterReconcileResult:
    """跨两表账单总量对账结果。"""

    status: str  # STATUS_OK 或 STATUS_CONFLICT
    conflicts: Tuple[str, ...]  # REASON_TOTAL_* 原因
    gcd: int  # gcd(旧表模数, 新表模数)
    old_segment: Optional[MeterSegmentNarrowed]
    new_segment: Optional[MeterSegmentNarrowed]
    feasible_min_total: int
    feasible_max_total: int


def _extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """扩展欧几里得算法：返回 ``(g, x, y)`` 使 ``a*x + b*y = g = gcd(a, b)``。"""
    if b == 0:
        return a, 1, 0
    g, x1, y1 = _extended_gcd(b, a % b)
    return g, y1, x1 - (a // b) * y1


def _ceil_div(a: int, b: int) -> int:
    """b > 0 时的精确 ceil(a / b)，a 可为负（不依赖浮点）。"""
    return -((-a) // b)


def reconcile_meter_total(
    old_diff: int,
    old_bounds: Tuple[int, int],
    old_modulus: int,
    new_diff: int,
    new_bounds: Tuple[int, int],
    new_modulus: int,
    total_usage: int,
) -> MeterReconcileResult:
    """以跨两表的账单总用量精确联立两段回绕次数，模数可不同，不枚举圈数组合。

    旧表段用量 ``u1 = d1 + k1*M1``，新表段用量 ``u2 = d2 + k2*M2``
    （``d`` 为段内读数差，``k`` 为非负整圈数）。账单要求：

        u1 + u2 = B
        => M1*k1 + M2*k2 = N，其中 N = B - d1 - d2

    该线性丢番图方程：

    1. 有整数解当且仅当 ``g = gcd(M1, M2)`` 整除 ``N``；否则账单与两表
       模数不相容（``total_incompatible_modulus``）；
    2. 有整数解时，由扩展欧几里得给出一组特解 ``(k10, k20)``，全部解为
       单参数整数族 ``k1 = k10 + p*t``、``k2 = k20 - q*t``
       （``p = M2/g``，``q = M1/g``）。两段各自的圈数区间
       ``[lo, hi]`` 把 ``t`` 夹到一个连续整数区间 ``[t_min, t_max]``：

           t >= ceil((lo1 - k10) / p)，t <= floor((hi1 - k10) / p)
           t >= ceil((k20 - hi2) / q)，t <= floor((k20 - lo2) / q)

       ``t_min > t_max`` 即账单总量超界（``total_out_of_range``）。
    3. ``t_min == t_max`` 时两段圈数同时唯一；否则各段可行圈数仍为
       等差数列（公差分别为 ``p``、``q``），据端点给出精确用量上下界，
       不猜测。注意每一个可行 ``t`` 都严格满足 ``u1 + u2 == B``。

    :param old_bounds/new_bounds: 各段在非负增量与速率上限下的可行圈数
        区间 ``(k_min, k_max)``，调用方需保证每段均可行
    """
    lo1, hi1 = old_bounds
    lo2, hi2 = new_bounds

    feasible_min_total = (
        old_diff + lo1 * old_modulus + new_diff + lo2 * new_modulus
    )
    feasible_max_total = (
        old_diff + hi1 * old_modulus + new_diff + hi2 * new_modulus
    )

    g, c1, c2 = _extended_gcd(old_modulus, new_modulus)
    numerator = total_usage - old_diff - new_diff

    if numerator % g != 0:
        return MeterReconcileResult(
            status=STATUS_CONFLICT,
            conflicts=(REASON_TOTAL_INCOMPATIBLE_MODULUS,),
            gcd=g,
            old_segment=None,
            new_segment=None,
            feasible_min_total=feasible_min_total,
            feasible_max_total=feasible_max_total,
        )

    # 特解：M1*k10 + M2*k20 = numerator
    n = numerator // g
    k10 = n * c1
    k20 = n * c2
    p = new_modulus // g  # k1 随 t 的步长
    q = old_modulus // g  # k2 随 t 的步长（方向相反）

    # k1 = k10 + p*t ∈ [lo1, hi1]
    t_lo = _ceil_div(lo1 - k10, p)
    t_hi = (hi1 - k10) // p
    # k2 = k20 - q*t ∈ [lo2, hi2]
    t_lo = max(t_lo, _ceil_div(k20 - hi2, q))
    t_hi = min(t_hi, (k20 - lo2) // q)

    if t_lo > t_hi:
        return MeterReconcileResult(
            status=STATUS_CONFLICT,
            conflicts=(REASON_TOTAL_OUT_OF_RANGE,),
            gcd=g,
            old_segment=None,
            new_segment=None,
            feasible_min_total=feasible_min_total,
            feasible_max_total=feasible_max_total,
        )

    k1_min = k10 + p * t_lo
    k1_max = k10 + p * t_hi
    k2_min = k20 - q * t_hi  # k2 随 t 递减：最小在 t_hi
    k2_max = k20 - q * t_lo

    return MeterReconcileResult(
        status=STATUS_OK,
        conflicts=(),
        gcd=g,
        old_segment=MeterSegmentNarrowed(
            meter="old",
            diff=old_diff,
            modulus=old_modulus,
            min_wraps=k1_min,
            max_wraps=k1_max,
            wraps_step=p,
        ),
        new_segment=MeterSegmentNarrowed(
            meter="new",
            diff=new_diff,
            modulus=new_modulus,
            min_wraps=k2_min,
            max_wraps=k2_max,
            wraps_step=q,
        ),
        feasible_min_total=feasible_min_total,
        feasible_max_total=feasible_max_total,
    )


# ---------------------------------------------------------------------------
# 漏采读数推断（missing-reading）：账单总量 + 漏采时刻两侧容量约束候选读数
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MissingReadingResult:
    """漏采读数推断结果。"""

    status: str  # STATUS_OK 或 STATUS_CONFLICT
    conflicts: Tuple[str, ...]  # REASON_TOTAL_INCOMPATIBLE_MODULUS / REASON_INSUFFICIENT_CAPACITY
    diff: int  # 末读数 - 首读数
    total_wraps: Optional[int]  # 账单隐含的总回绕次数 K；不相容时为 None
    left_capacity: int  # floor(速率 × 漏采时刻左侧秒数)
    right_capacity: int  # floor(速率 × 漏采时刻右侧秒数)
    # 左侧可行用量闭区间 [left_usage_min, left_usage_max]；冲突时为 None
    left_usage_min: Optional[int]
    left_usage_max: Optional[int]
    # 候选读数闭区间（精确、不重叠，按左侧用量递增的环序）；冲突时为空
    intervals: Tuple[Tuple[int, int], ...]
    candidate_count: int  # 候选读数个数（去重后的剩余类数）；冲突时为 0

    @property
    def unique(self) -> bool:
        """候选读数是否唯一。"""
        return self.candidate_count == 1


def solve_missing_reading(
    start_value: int,
    end_value: int,
    left_seconds: int,
    right_seconds: int,
    modulus: int,
    max_rate_per_second: Fraction,
    total_usage: int,
) -> MissingReadingResult:
    """由账单总量与漏采时刻两侧容量推断漏采读数的候选闭区间。

    漏采时刻把账期切成左（``left_seconds`` 秒）、右（``right_seconds``
    秒）两段，整数容量分别为 ``C_L = floor(速率 × 左秒数)``、
    ``C_R = floor(速率 × 右秒数)``。账单总量 ``B`` 须满足
    ``B = diff + K*M``（``diff = 末读数 - 首读数``，``K >= 0`` 为总回绕
    次数）：

    1. ``B - diff`` 不被 ``M`` 整除即账单与模数不相容
       （``total_incompatible_modulus``）；同余且 ``B >= 0`` 时
       ``K = (B - diff) / M`` 自动满足非负性（``K >= K_min``），无需
       另行校验；
    2. 左侧用量 ``x`` 与右侧用量 ``B - x`` 须分别落在 ``[0, C_L]``、
       ``[0, C_R]`` 内，故左侧可行用量闭区间为

           [max(0, B - C_R), min(C_L, B)]

       为空（即 ``B > C_L + C_R``）时两侧容量不足
       （``insufficient_capacity``）。容量均为整数，区间内每个整数
       ``x`` 都可由某个左右侧分配达到。

    候选漏采读数为 ``(首读数 + x) mod M``。``x`` 取遍上述连续整数区间
    时，候选在模 ``M`` 环上形成连续弧：弧上点数不少于 ``M`` 时覆盖全部
    剩余类，返回全域 ``[0, M-1]``；否则自 ``(首读数 + x_min) mod M`` 起
    展开，跨越零点时拆为 ``[a, M-1]`` 与 ``[0, b]`` 两段，保证各闭区间
    精确、互不重叠且均落在 ``[0, M-1]`` 内。
    """
    diff = end_value - start_value
    left_capacity = math.floor(max_rate_per_second * left_seconds)
    right_capacity = math.floor(max_rate_per_second * right_seconds)

    def _conflict(reason: str, total_wraps: Optional[int]) -> MissingReadingResult:
        return MissingReadingResult(
            status=STATUS_CONFLICT,
            conflicts=(reason,),
            diff=diff,
            total_wraps=total_wraps,
            left_capacity=left_capacity,
            right_capacity=right_capacity,
            left_usage_min=None,
            left_usage_max=None,
            intervals=(),
            candidate_count=0,
        )

    # 账单总量与首尾读数差必须模 M 同余（等价于总回绕次数为整数）
    if (total_usage - diff) % modulus != 0:
        return _conflict(REASON_TOTAL_INCOMPATIBLE_MODULUS, None)

    total_wraps = (total_usage - diff) // modulus
    # 两侧容量之和容纳不下账单总量：无可行的左右侧整数分配
    if total_usage > left_capacity + right_capacity:
        return _conflict(REASON_INSUFFICIENT_CAPACITY, total_wraps)

    x_lo = max(0, total_usage - right_capacity)
    x_hi = min(left_capacity, total_usage)
    count = x_hi - x_lo + 1
    if count >= modulus:
        # 候选点不少于模数：覆盖全部剩余类，返回全域
        intervals = ((0, modulus - 1),)
        count = modulus
    else:
        first = (start_value + x_lo) % modulus
        arc = x_hi - x_lo  # 弧长（步数），候选数 = arc + 1 < modulus
        if first + arc <= modulus - 1:
            intervals = ((first, first + arc),)
        else:
            # 跨零点拆段：[first, M-1] 与 [0, 弧末端]
            intervals = ((first, modulus - 1), (0, (first + arc) % modulus))

    return MissingReadingResult(
        status=STATUS_OK,
        conflicts=(),
        diff=diff,
        total_wraps=total_wraps,
        left_capacity=left_capacity,
        right_capacity=right_capacity,
        left_usage_min=x_lo,
        left_usage_max=x_hi,
        intervals=intervals,
        candidate_count=count,
    )


# ---------------------------------------------------------------------------
# 最小统一速率上限（minimum-rate）：账单总量约束下求最小可行统一速率
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MinimumRateResult:
    """最小统一速率上限求解结果。"""

    status: str  # STATUS_OK 或 STATUS_CONFLICT
    conflicts: Tuple[str, ...]  # REASON_TOTAL_* 原因
    total_wraps: Optional[int]  # 账单隐含的总回绕次数 K；不相容时为 None
    min_rate: Optional[Fraction]  # 最小统一每秒速率上限（约分分数）；冲突时为 None
    narrowed: Tuple[NarrowedSegment, ...]  # 该门槛与账单总量约束后的各段圈数区间
    feasible_min_total: int  # 可行账单总量下界（含）：各段最小非负用量之和


def solve_minimum_rate(
    diffs: Sequence[int],
    elapsed_seconds: Sequence[int],
    total_usage: int,
    modulus: int,
) -> MinimumRateResult:
    """求使账单总量可行的最小统一每秒速率上限（精确分数，不枚举圈数组合）。

    各段真实用量 ``u_i = d_i + k_i*M``（``k_i >= 0`` 整数，``d_i`` 为段内
    读数差），须满足 ``Σ u_i = B``（账单总量）；段 ``i`` 的速率为
    ``u_i / t_i``（``t_i`` 为段历时秒数）。本函数求最小统一速率上限
    ``R*``，使存在非负整数圈数分配 ``(k_i)`` 满足总量约束且所有
    ``u_i / t_i <= R*``：

    1. 同余校验：``B = Σ d_i + M*K`` 要求 ``K = (B - Σd_i)/M`` 为整数，
       否则账单与模数不相容（``total_incompatible_modulus``）；
    2. 非负性：``k_i >= k_min_i``（使 ``u_i >= 0`` 的最小圈数，取值 0 或
       1），要求 ``K >= Σ k_min_i``，否则账单低于各段最小非负用量之和
       （``total_out_of_range``；可行账单总量无上限，速率总可继续放大）；
    3. 令 ``a_i = d_i + k_min_i*M ∈ [0, M)``（段 ``i`` 的最小非负用量），
       ``E = K - Σ k_min_i`` 为待分配的额外圈数。段 ``i`` 分到 ``x_i``
       圈后速率为 ``(a_i + x_i*M)/t_i``，问题即：``x_i >= 0``、
       ``Σ x_i = E`` 下最小化 ``max_i (a_i + x_i*M)/t_i``；
    4. 下界一：任何分配的 max 速率 >= 各段基础速率 ``a_i/t_i`` 的最大值
       ``R0``。下界二：第 ``m`` 圈把段 ``i`` 速率抬到门槛
       ``v_i(m) = (a_i + m*M)/t_i``，分配 ``E`` 圈必然使用 ``E`` 个门槛，
       故 max 速率 >= 全体门槛中第 ``E`` 小者 ``V_E``。于是
       ``R* = max(R0, V_E)``——取 ``E`` 个最小门槛对应的圈数分配即可
       达到（段内门槛递增，自动满足先后顺序）；
    5. ``V_E`` 用精确二分查找：``f(R) = Σ floor((R*t_i - a_i)/M)`` 统计
       不超过 ``R`` 的门槛数，取使 ``f(R) >= E`` 的最小 ``R``。候选值
       两两之差 >= ``1/max(t_i)^2``（同段间距 ``M/t_i``，跨段为整数非零
       分子除以 ``t_i*t_j``），区间收窄到该宽度以下时 ``(lo, hi]`` 内恰
       剩一个候选，按整除条件直接取出，全程整数/分数精确运算；
    6. 定出 ``R*`` 后，各段可行圈数
       ``k_i ∈ [k_min_i, k_min_i + floor((R*·t_i - a_i)/M)]``，再交由
       ``reconcile_total`` 以账单总量收紧为最终区间；圈数唯一的段给出
       确定用量，其余仅保留范围，不猜测。

    :param diffs: 各段首尾读数差 r2 - r1
    :param elapsed_seconds: 各段历时秒数（正整数，与 diffs 等长）
    :param total_usage: 账单总用量（非负整数）
    :param modulus: 计数器模数
    """
    if len(diffs) != len(elapsed_seconds):
        raise ValueError("diffs 与 elapsed_seconds 数量必须一致")
    if not diffs:
        raise ValueError("至少需要一个读数区间")
    if any(t <= 0 for t in elapsed_seconds):
        raise ValueError("各段历时必须为正整数秒")

    # k_min_i：使段用量非负的最小圈数（0 或 1）；a_i：对应最小非负用量
    k_mins = [0 if d >= 0 else (-d + modulus - 1) // modulus for d in diffs]
    base_usages = [d + k * modulus for d, k in zip(diffs, k_mins)]
    sum_k_min = sum(k_mins)
    feasible_min_total = sum(base_usages)

    end_minus_start = sum(diffs)  # 望远镜相消：末读数 - 首读数
    numerator = total_usage - end_minus_start
    # 总量必须与首尾读数差模 M 同余（等价于总回绕次数为整数）
    if numerator % modulus != 0:
        return MinimumRateResult(
            status=STATUS_CONFLICT,
            conflicts=(REASON_TOTAL_INCOMPATIBLE_MODULUS,),
            total_wraps=None,
            min_rate=None,
            narrowed=(),
            feasible_min_total=feasible_min_total,
        )

    total_wraps = numerator // modulus
    # 账单低于各段最小非负用量之和：任何速率下都不存在非负整数分配
    if total_wraps < sum_k_min:
        return MinimumRateResult(
            status=STATUS_CONFLICT,
            conflicts=(REASON_TOTAL_OUT_OF_RANGE,),
            total_wraps=total_wraps,
            min_rate=None,
            narrowed=(),
            feasible_min_total=feasible_min_total,
        )

    extra = total_wraps - sum_k_min  # 待分配的额外圈数 E >= 0
    times = list(elapsed_seconds)

    def threshold_count(rate: Fraction) -> int:
        # 不超过 rate 的抬升门槛 v_i(m) = (a_i + m*M)/t_i（m >= 1）总数，
        # 即 Σ max(0, floor((rate*t_i - a_i)/M))；用分子/分母做整数 floor
        # 除法，与 Fraction 精确等价
        p, q = rate.numerator, rate.denominator
        total = 0
        for a, t in zip(base_usages, times):
            term = (p * t - a * q) // (q * modulus)
            if term > 0:
                total += term
        return total

    # R* 的下界之一：各段基础速率 a_i/t_i 的最大值 R0
    base_rate = max(Fraction(a, t) for a, t in zip(base_usages, times))

    if extra == 0 or threshold_count(base_rate) >= extra:
        # 无额外圈数，或 V_E 不超过 R0（R0 处已能容纳全部 E 圈）
        min_rate = base_rate
    else:
        # 二分 V_E：不变式 threshold_count(lo) < E <= threshold_count(hi)
        lo = base_rate
        # 段 i 自身的前 E 个门槛即覆盖 E 个，故 hi 取各段第 E 门槛的最小值
        hi = min(
            Fraction(a + extra * modulus, t)
            for a, t in zip(base_usages, times)
        )
        max_t = max(times)
        # 不同候选值两两之差 >= 1/max_t^2：同段间距 M/t_i >= 1/t_i；跨段
        # 为 |(a_i+mM)t_j - (a_j+lM)t_i| / (t_i*t_j) >= 1/(t_i*t_j)
        gap = Fraction(1, max_t * max_t)
        while hi - lo > gap:
            mid = (lo + hi) / 2
            if threshold_count(mid) >= extra:
                hi = mid
            else:
                lo = mid
        # (lo, hi] 内恰剩一个候选值（数学保证），按整除条件直接取出
        values = set()
        for a, t in zip(base_usages, times):
            top = math.floor(hi * t)
            if top > lo * t and top >= a + modulus and (top - a) % modulus == 0:
                values.add(Fraction(top, t))
        if len(values) != 1:
            raise RuntimeError("最小速率候选提取失败：候选不唯一或不存在")
        min_rate = values.pop()

    # 该门槛下各段可行圈数区间，再以账单总量收紧（整数区间运算，不枚举）
    bounds = [
        (k0, k0 + math.floor((min_rate * t - a) / modulus))
        for k0, t, a in zip(k_mins, times, base_usages)
    ]
    reconciled = reconcile_total(diffs, bounds, total_usage, modulus)
    if reconciled.status != STATUS_OK:
        raise RuntimeError("最小速率构造失败：账单对账意外冲突")

    return MinimumRateResult(
        status=STATUS_OK,
        conflicts=(),
        total_wraps=total_wraps,
        min_rate=min_rate,
        narrowed=reconciled.narrowed,
        feasible_min_total=feasible_min_total,
    )
