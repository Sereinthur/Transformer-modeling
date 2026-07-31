"""算子级Roofline成本与阶段汇总。"""

from __future__ import annotations

from math import ceil, log2
from typing import Any

from ..config import Config
from .work_item import OPERATOR_LABELS, WorkItem


def _interpolate_by_rows(table: tuple[tuple[int, float], ...], rows: int) -> float:
    """在log2(rows)域对效率表做分段线性插值，两端取边界值。"""

    if rows <= table[0][0]:
        return table[0][1]
    if rows >= table[-1][0]:
        return table[-1][1]
    for (low_rows, low_eff), (high_rows, high_eff) in zip(table, table[1:]):
        if low_rows <= rows <= high_rows:
            span = log2(high_rows) - log2(low_rows)
            weight = (log2(rows) - log2(low_rows)) / span if span else 0.0
            return low_eff + weight * (high_eff - low_eff)
    return table[-1][1]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║            OperatorCostModel — 将 WorkItem 账单转为耗时估算              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class OperatorCostModel:
    """Roofline 性能模型：接收 WorkItem 列表，根据 GPU 算力/带宽估算延迟。"""

    # ─────────────────────────────────────────────────────────────────────
    # 初始化：绑定一份 Config（含 GPU 硬件参数和效率系数）
    # ─────────────────────────────────────────────────────────────────────
    def __init__(self, config: Config):
        self.config = config

    # ─────────────────────────────────────────────────────────────────────
    # 有效 HBM 带宽
    #   有实测值 → 用实测值
    #   没有     → 峰值带宽 × hbm_efficiency（默认 80%）
    # ─────────────────────────────────────────────────────────────────────
    @property
    def effective_memory_bandwidth(self) -> float:
        hardware = self.config.hardware
        if hardware.measured_memory_bandwidth_bytes_per_second is not None:
            return hardware.measured_memory_bandwidth_bytes_per_second
        return (
            hardware.peak_memory_bandwidth_bytes_per_second
            * self.config.execution.hbm_efficiency
        )

    # ─────────────────────────────────────────────────────────────────────
    # Tile 对齐效率
    #   GEMM 形状不能整除 GPU Tensor Core 的 tile 大小时，会 pad 浪费。
    #   返回值 ≤ 1.0，表示有效计算量占比。
    #   例如 tile 128×128，矩阵 130×130 → pad 到 256×256，效率约 25%
    # ─────────────────────────────────────────────────────────────────────
    def tile_efficiency(self, item: WorkItem) -> float:
        hardware = self.config.hardware
        if item.gemm_shape is None:
            return 1.0
        if not all((hardware.matrix_tile_m, hardware.matrix_tile_n, hardware.matrix_tile_k)):
            return 1.0
        m, n, k = item.gemm_shape
        tm = int(hardware.matrix_tile_m or 1)
        tn = int(hardware.matrix_tile_n or 1)
        tk = int(hardware.matrix_tile_k or 1)
        padded = ceil(m / tm) * tm * ceil(n / tn) * tn * ceil(k / tk) * tk
        return (m * n * k) / padded

    # ─────────────────────────────────────────────────────────────────────
    # 核心：单算子 Roofline 估算
    #   输入：一个 WorkItem（账单）+ phase（"prefill" 或 "decode"）
    #   输出：该算子的详细耗时字典
    #
    #   流程：
    #     1. 根据算子类型选效率系数（gemm/attention/vector）
    #     2. 算三个时间：
    #        compute_time = FLOPs ÷ (峰值算力 × 效率 × tile效率)
    #        memory_time  = HBM字节数 ÷ 有效带宽
    #        launch_time  = kernel数 × 每次启动延迟
    #     3. 有界重叠模型：
    #        lower = max(compute, memory) + launch   ← 完全重叠
    #        upper = compute + memory + launch       ← 完全不重叠
    #        final = lower + overlap_rho × (upper - lower)  ← 插值
    # ─────────────────────────────────────────────────────────────────────
    def estimate(self, item: WorkItem, phase: str) -> dict[str, Any]:
        execution = self.config.execution
        hardware = self.config.hardware
        if item.kind == "vector":
            efficiency = execution.vector_efficiency
        elif item.kind == "attention":
            efficiency = (
                execution.prefill_attention_efficiency
                if phase == "prefill"
                else execution.decode_attention_efficiency
            )
        elif (
            execution.gemm_efficiency_by_rows is not None
            and item.gemm_shape is not None
        ):
            # 形状感知效率：按GEMM的M维插值，代替Prefill/Decode常数。
            efficiency = _interpolate_by_rows(
                execution.gemm_efficiency_by_rows, item.gemm_shape[0]
            )
        else:
            efficiency = (
                execution.prefill_gemm_efficiency
                if phase == "prefill"
                else execution.decode_gemm_efficiency
            )
        tile_efficiency = self.tile_efficiency(item)
        effective_compute = (
            hardware.effective_compute_ops_per_second * efficiency * tile_efficiency
        )
        compute_time = item.executed_ops / effective_compute
        memory_time = item.hbm_payload_bytes / self.effective_memory_bandwidth
        launch_time = item.kernel_count * hardware.kernel_launch_latency_seconds
        lower = max(compute_time, memory_time) + launch_time
        upper = compute_time + memory_time + launch_time
        final = lower + execution.overlap_rho * (upper - lower)
        service_parts = {
            "compute": compute_time,
            "memory": memory_time,
            "launch": launch_time,
        }
        bottleneck = max(service_parts, key=service_parts.get)
        return {
            "name": item.name,
            "中文名称": OPERATOR_LABELS.get(item.name, item.name),
            "kind": item.kind,
            "kernel_count": item.kernel_count,
            "gemm_shape": list(item.gemm_shape) if item.gemm_shape is not None else None,
            "ops": {
                "logical": item.logical_ops if item.logical_ops is not None else item.executed_ops,
                "executed": item.executed_ops,
            },
            "hbm_payload_bytes": item.hbm_payload_bytes,
            "arithmetic_intensity_flop_per_byte": (
                item.executed_ops / item.hbm_payload_bytes if item.hbm_payload_bytes else None
            ),
            "efficiency": {
                "shape_or_operator": efficiency,
                "tile": tile_efficiency,
            },
            "time_seconds": {
                "compute": compute_time,
                "memory": memory_time,
                "launch": launch_time,
                "lower_bound": lower,
                "upper_bound": upper,
                "estimated": final,
            },
            "achieved": {
                "ops_per_second": item.executed_ops / final if final else None,
                "hbm_bandwidth_bytes_per_second": item.hbm_payload_bytes / final if final else None,
            },
            "bottleneck": bottleneck,
        }

    # ─────────────────────────────────────────────────────────────────────
    # 汇总：把一个阶段（prefill 或 decode）的所有算子耗时加起来
    #   输入：WorkItem 列表 + phase
    #   输出：总延迟、总 FLOPs、总 HBM 流量、瓶颈分析
    #   如果 details=True，还会附上每个算子的明细
    # ─────────────────────────────────────────────────────────────────────
    def summarize(self, work: list[WorkItem], phase: str, details: bool = True) -> dict[str, Any]:
        costs = [self.estimate(item, phase) for item in work]
        latency = sum(cost["time_seconds"]["estimated"] for cost in costs)
        lower = sum(cost["time_seconds"]["lower_bound"] for cost in costs)
        upper = sum(cost["time_seconds"]["upper_bound"] for cost in costs)
        ops_logical = sum(cost["ops"]["logical"] for cost in costs)
        ops_executed = sum(cost["ops"]["executed"] for cost in costs)
        traffic = sum(cost["hbm_payload_bytes"] for cost in costs)
        contributions = {kind: 0.0 for kind in ("compute", "memory", "launch")}
        for cost in costs:
            for kind in contributions:
                contributions[kind] += cost["time_seconds"][kind]
        summary: dict[str, Any] = {
            "说明": "本阶段所有算子按依赖顺序串行汇总；算子内部按配置估算计算与HBM重叠。",
            "latency_seconds": latency,
            "latency_lower_bound_seconds": lower,
            "latency_upper_bound_seconds": upper,
            "ops": {"logical": ops_logical, "executed": ops_executed},
            "hbm": {
                "payload_bytes": traffic,
                "effective_bandwidth_bytes_per_second": self.effective_memory_bandwidth,
                "average_achieved_bandwidth_bytes_per_second": traffic / latency if latency else None,
                "average_fraction_of_effective_bandwidth": (
                    traffic / latency / self.effective_memory_bandwidth if latency else None
                ),
            },
            "bottleneck": max(contributions, key=contributions.get),
            "candidate_time_sums": contributions,
        }
        if details:
            summary["operators"] = costs
        return summary
