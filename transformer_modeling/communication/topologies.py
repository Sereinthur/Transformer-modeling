"""Ring、Bus、Crossbar与Mesh解析通信公式。"""

from __future__ import annotations

from math import ceil, log2, sqrt

#通信时间模型
def ring_all_reduce_seconds(
    payload_bytes: float,
    ranks: int,
    bandwidth_bytes_per_second: float,
    step_latency_seconds: float,
) -> float:
    """理想Ring All-Reduce时间；alpha为每轮已含同步的延迟。"""

    if ranks == 1:
        return 0.0
    return (
        2 * (ranks - 1) * step_latency_seconds
        + 2 * (ranks - 1) / ranks * payload_bytes / bandwidth_bytes_per_second
    )


def ring_all_gather_seconds(
    local_payload_bytes: float,
    ranks: int,
    bandwidth_bytes_per_second: float,
    step_latency_seconds: float,
) -> float:
    """理想Ring All-Gather时间；local_payload是每rank初始shard。"""

    if ranks == 1:
        return 0.0
    return (
        (ranks - 1) * step_latency_seconds
        + (ranks - 1) * local_payload_bytes / bandwidth_bytes_per_second
    )


def bus_all_reduce_seconds(
    payload_bytes: float,
    ranks: int,
    bandwidth_bytes_per_second: float,
    step_latency_seconds: float,
) -> float:
    """共享Bus：各非根rank依次上传，根rank完成归约后广播一次。"""

    if ranks == 1:
        return 0.0
    return ranks * step_latency_seconds + ranks * payload_bytes / bandwidth_bytes_per_second


def bus_all_gather_seconds(
    local_payload_bytes: float,
    ranks: int,
    bandwidth_bytes_per_second: float,
    step_latency_seconds: float,
) -> float:
    """共享Bus：每个rank依次广播自己的本地shard。"""

    if ranks == 1:
        return 0.0
    return ranks * step_latency_seconds + ranks * local_payload_bytes / bandwidth_bytes_per_second


def crossbar_all_reduce_seconds(
    payload_bytes: float,
    ranks: int,
    bandwidth_bytes_per_second: float,
    step_latency_seconds: float,
) -> float:
    """非阻塞Crossbar上的Recursive Doubling All-Reduce。"""

    if ranks == 1:
        return 0.0
    rounds = ceil(log2(ranks))
    return rounds * (step_latency_seconds + payload_bytes / bandwidth_bytes_per_second)


def crossbar_all_gather_seconds(
    local_payload_bytes: float,
    ranks: int,
    bandwidth_bytes_per_second: float,
    step_latency_seconds: float,
) -> float:
    """非阻塞Crossbar上的Recursive Doubling All-Gather。"""

    if ranks == 1:
        return 0.0
    rounds = ceil(log2(ranks))
    return (
        rounds * step_latency_seconds
        + (ranks - 1) * local_payload_bytes / bandwidth_bytes_per_second
    )


def mesh_dimensions(
    ranks: int,
    rows: int | None = None,
    columns: int | None = None,
) -> tuple[int, int]:
    """返回二维Mesh尺寸；未显式给定时选择最接近方形的整数因数。"""

    if rows is not None and columns is not None:
        if rows * columns != ranks:
            raise ValueError("mesh rows * columns must equal ranks")
        return rows, columns
    candidate = max(1, int(sqrt(ranks)))
    while ranks % candidate:
        candidate -= 1
    return candidate, ranks // candidate


def mesh_all_reduce_seconds(
    payload_bytes: float,
    ranks: int,
    bandwidth_bytes_per_second: float,
    step_latency_seconds: float,
    rows: int | None = None,
    columns: int | None = None,
) -> float:
    """二维Mesh上的维度有序生成树Reduce + Broadcast保守模型。"""

    if ranks == 1:
        return 0.0
    mesh_rows, mesh_columns = mesh_dimensions(ranks, rows, columns)
    diameter = mesh_rows + mesh_columns - 2
    return 2 * diameter * (
        step_latency_seconds + payload_bytes / bandwidth_bytes_per_second
    )


def mesh_all_gather_seconds(
    local_payload_bytes: float,
    ranks: int,
    bandwidth_bytes_per_second: float,
    step_latency_seconds: float,
    rows: int | None = None,
    columns: int | None = None,
) -> float:
    """二维Mesh按行后按列的Dimension-Ordered All-Gather。"""

    if ranks == 1:
        return 0.0
    mesh_rows, mesh_columns = mesh_dimensions(ranks, rows, columns)
    diameter = mesh_rows + mesh_columns - 2
    return (
        diameter * step_latency_seconds
        + (ranks - 1) * local_payload_bytes / bandwidth_bytes_per_second
    )
