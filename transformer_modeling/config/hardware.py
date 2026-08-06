"""芯片、显存和互联配置。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .common import positive

_positive = positive

@dataclass(frozen=True)
class InterconnectSpec:
    """单一同构TP通信域的拓扑与集合通信参数。"""

    topology: str                                          # 互联拓扑：ring / bus / crossbar / mesh
    effective_channel_bandwidth_bytes_per_second: float | None  # 拓扑口径下的有效通信带宽（字节/秒），TP>1 时必填
    collective_step_latency_seconds: float | None             # 每步集合通信的固定延迟（秒），TP>1 时必填
    mesh_rows: int | None = None                              # Mesh 行数；留空时按 TP 自动选择接近方形的因数
    mesh_columns: int | None = None                           # Mesh 列数；必须与行数同时填写
    pipeline_effective_bandwidth_bytes_per_second: float | None = None  # PP相邻Stage点到点有效带宽；留空时复用通道带宽
    pipeline_transfer_latency_seconds: float | None = None     # PP单次点到点固定延迟；留空时复用collective step延迟

    @property
    def effective_pipeline_bandwidth(self) -> float | None:
        """PP点到点带宽；未单独配置时沿用互联有效通道带宽。"""

        return (
            self.pipeline_effective_bandwidth_bytes_per_second
            if self.pipeline_effective_bandwidth_bytes_per_second is not None
            else self.effective_channel_bandwidth_bytes_per_second
        )

    @property
    def effective_pipeline_latency(self) -> float | None:
        """PP点到点固定延迟；未单独配置时沿用collective step延迟。"""

        return (
            self.pipeline_transfer_latency_seconds
            if self.pipeline_transfer_latency_seconds is not None
            else self.collective_step_latency_seconds
        )

#这一部分的作用是从json格式中提取出参数，但其实这个json格式是人为规定的
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InterconnectSpec":
        return cls(
            topology=str(data.get("topology", "ring")).lower(),
            effective_channel_bandwidth_bytes_per_second=(
                float(data["effective_channel_bandwidth_bytes_per_second"])
                if data.get("effective_channel_bandwidth_bytes_per_second") is not None
                else None
            ),
            collective_step_latency_seconds=(
                float(data["collective_step_latency_seconds"])
                if data.get("collective_step_latency_seconds") is not None
                else None
            ),
            mesh_rows=(int(data["mesh_rows"]) if data.get("mesh_rows") is not None else None),
            mesh_columns=(
                int(data["mesh_columns"]) if data.get("mesh_columns") is not None else None
            ),
            pipeline_effective_bandwidth_bytes_per_second=(
                float(data["pipeline_effective_bandwidth_bytes_per_second"])
                if data.get("pipeline_effective_bandwidth_bytes_per_second") is not None
                else None
            ),
            pipeline_transfer_latency_seconds=(
                float(data["pipeline_transfer_latency_seconds"])
                if data.get("pipeline_transfer_latency_seconds") is not None
                else None
            ),
        )



@dataclass(frozen=True)
class HardwareSpec:
    """芯片算力、显存和启动开销。"""
    name: str                                                              # 芯片名称
    peak_ops_per_second: float | None                                      # 峰值算力；MX格式可只做容量分析
    measured_ops_per_second: float | None                                  # 用户提供的外部实测计算上限；存在时优先使用
    device_memory_capacity_bytes: int                                      # 显存总容量（字节）
    peak_memory_bandwidth_bytes_per_second: float                          # HBM 峰值带宽（字节/秒）
    measured_memory_bandwidth_bytes_per_second: float | None                # HBM 实测读取带宽（字节/秒），没有则用峰值×效率
    kernel_launch_latency_seconds: float                                   # 单次 kernel 启动延迟（秒）
    matrix_tile_m: int | None = None                                      # Tensor Core tile 的 M 维度（用于算 tile 对齐效率）
    matrix_tile_n: int | None = None                                      # Tensor Core tile 的 N 维度
    matrix_tile_k: int | None = None                                      # Tensor Core tile 的 K 维度
    runtime_reserved_bytes: int = 0                                        # 用户设置的运行时安全余量（字节）
    baseline_unavailable_bytes: int = 0                                   # 校准时已被系统/其他进程占用的显存
    device_count: int = 1                                                 # 总芯片数量 = TP × EP × PP
    interconnect: InterconnectSpec = InterconnectSpec("ring", None, None)  # TP/EP集合通信与PP传输参数

    peak_ops_by_dtype: dict[str, float] | None = None
    measured_ops_by_dtype: dict[str, float] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], weight_dtype: str) -> "HardwareSpec":
        compute = data.get("compute", {})
        throughput = compute.get("throughput", {})
        measured_throughput = compute.get("measured_throughput", {})
        dtype_key = (
            "mxfp4_mxfp8_ops_per_second"
            if weight_dtype.lower() == "mxfp4"
            else f"{weight_dtype.lower()}_dense_ops_per_second"
        )
        peak = throughput.get(dtype_key)
        measured_peak = measured_throughput.get(dtype_key)
        peak_by_dtype = {
            key.removesuffix("_dense_ops_per_second"): float(value)
            for key, value in throughput.items()
            if key.endswith("_dense_ops_per_second") and value is not None
        }
        if throughput.get("mxfp4_mxfp8_ops_per_second") is not None:
            shared = float(throughput["mxfp4_mxfp8_ops_per_second"])
            peak_by_dtype.update({"mxfp4": shared, "mxfp8": shared})
        measured_by_dtype = {
            key.removesuffix("_dense_ops_per_second"): float(value)
            for key, value in measured_throughput.items()
            if key.endswith("_dense_ops_per_second") and value is not None
        }
        if measured_throughput.get("mxfp4_mxfp8_ops_per_second") is not None:
            shared = float(measured_throughput["mxfp4_mxfp8_ops_per_second"])
            measured_by_dtype.update({"mxfp4": shared, "mxfp8": shared})
        memory = data.get("device_memory", {})
        runtime = data.get("runtime", {})
        tile = compute.get("matrix_tile", {}) or {}
        spec = cls(
            name=str(data.get("name", "unnamed_accelerator")),
            peak_ops_per_second=(
                float(peak if peak is not None else measured_peak)
                if peak is not None or measured_peak is not None else None
            ),
            measured_ops_per_second=(
                float(measured_peak) if measured_peak is not None else None
            ),
            device_memory_capacity_bytes=int(memory.get("capacity_bytes", 0)),
            peak_memory_bandwidth_bytes_per_second=float(
                memory.get("peak_bandwidth_bytes_per_second", 0)
            ),
            measured_memory_bandwidth_bytes_per_second=(
                float(memory["measured_read_bandwidth_bytes_per_second"])
                if memory.get("measured_read_bandwidth_bytes_per_second") is not None
                else None
            ),
            kernel_launch_latency_seconds=float(
                runtime.get("kernel_launch_latency_seconds", 0)
            ),
            matrix_tile_m=int(tile["m"]) if tile.get("m") else None,
            matrix_tile_n=int(tile["n"]) if tile.get("n") else None,
            matrix_tile_k=int(tile["k"]) if tile.get("k") else None,
            runtime_reserved_bytes=int(memory.get("reserved_capacity_bytes", 0)),
            baseline_unavailable_bytes=int(
                memory.get("baseline_unavailable_bytes", 0)
            ),
            device_count=int(data.get("device_count", 1)),
            interconnect=InterconnectSpec.from_dict(data.get("interconnect", {})),
            peak_ops_by_dtype=peak_by_dtype,
            measured_ops_by_dtype=measured_by_dtype,
        )
        # 进行参数检查
        if spec.peak_ops_per_second is not None:
            _positive("hardware peak compute throughput", spec.peak_ops_per_second)
        if spec.measured_ops_per_second is not None:
            _positive("hardware measured compute throughput", spec.measured_ops_per_second)
        _positive("device memory capacity", spec.device_memory_capacity_bytes)
        _positive("peak device-memory bandwidth", spec.peak_memory_bandwidth_bytes_per_second)
        if spec.measured_memory_bandwidth_bytes_per_second is not None:
            _positive(
                "measured device-memory bandwidth",
                spec.measured_memory_bandwidth_bytes_per_second,
            )
        if spec.kernel_launch_latency_seconds < 0:
            raise ValueError("kernel launch latency cannot be negative")
        if spec.runtime_reserved_bytes < 0:
            raise ValueError("reserved device-memory capacity cannot be negative")
        if spec.baseline_unavailable_bytes < 0:
            raise ValueError("baseline unavailable device memory cannot be negative")
        _positive("hardware.device_count", spec.device_count)
        for name in ("matrix_tile_m", "matrix_tile_n", "matrix_tile_k"):
            value = getattr(spec, name)
            if value is not None:
                _positive(f"hardware.compute.{name}", value)
        return spec

    def effective_compute_ops_per_second(self, dtype: str | None = None) -> float:
        """用于成本模型的计算上限：优先采用用户提供的外部实测值。"""

        value, _ = self.compute_throughput(dtype)
        if value is None:
            raise ValueError("当前硬件缺少所选计算格式的吞吐，无法估算性能")
        return value

    def compute_throughput(self, dtype: str | None) -> tuple[float | None, str]:
        """Return throughput and provenance for one WorkItem compute format."""
        selected = (dtype or "").lower()
        measured = (self.measured_ops_by_dtype or {}).get(selected)
        if measured is not None:
            return measured, "measured"
        peak = (self.peak_ops_by_dtype or {}).get(selected)
        if peak is not None:
            return peak, "peak"
        fallback = self.measured_ops_per_second or self.peak_ops_per_second
        return fallback, "model_default_fallback" if fallback is not None else "missing"

    @property
    def performance_supported(self) -> bool:
        return self.measured_ops_per_second is not None or self.peak_ops_per_second is not None
