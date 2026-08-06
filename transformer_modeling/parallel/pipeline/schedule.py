"""PP层划分与前向流水调度。"""

from __future__ import annotations

from typing import Any

TOKEN_ID_BYTES = 4


def layer_partition(layer_count: int, pipeline_parallel: int) -> list[int]:
    """把连续Transformer层尽量均衡地分配到PP Stage。"""

    base, remainder = divmod(layer_count, pipeline_parallel)
    return [base + (1 if stage < remainder else 0) for stage in range(pipeline_parallel)]


def pipeline_schedule(
    stage_compute_seconds: list[float],
    stage_send_seconds: list[float],
    microbatches: int,
) -> dict[str, Any]:
    """用flow-shop递推计算前向流水的填充、稳态和排空时间。

    发送可与本Stage下一个microbatch的计算重叠，但同一相邻Stage链路上的
    发送会串行化，避免多个microbatch竞争同一有效带宽。
    """

    stage_count = len(stage_compute_seconds)
    if not stage_count:
        raise ValueError("pipeline schedule requires at least one stage")
    if len(stage_send_seconds) != stage_count:
        raise ValueError("stage_send_seconds must match stage_compute_seconds")
    if microbatches <= 0:
        raise ValueError("microbatches must be greater than zero")
    if any(value < 0 for value in (*stage_compute_seconds, *stage_send_seconds)):
        raise ValueError("stage compute and send times must be non-negative")
    finishes = [[0.0 for _ in range(microbatches)] for _ in range(stage_count)]
    starts = [[0.0 for _ in range(microbatches)] for _ in range(stage_count)]
    send_finishes = [[0.0 for _ in range(microbatches)] for _ in range(stage_count)]
    services = [
        stage_compute_seconds[index] + stage_send_seconds[index]
        for index in range(stage_count)
    ]
    for microbatch in range(microbatches):
        for stage in range(stage_count):
            upstream_ready = send_finishes[stage - 1][microbatch] if stage else 0.0
            stage_ready = finishes[stage][microbatch - 1] if microbatch else 0.0
            starts[stage][microbatch] = max(upstream_ready, stage_ready)
            finishes[stage][microbatch] = (
                starts[stage][microbatch] + stage_compute_seconds[stage]
            )
            link_ready = send_finishes[stage][microbatch - 1] if microbatch else 0.0
            send_finishes[stage][microbatch] = max(
                finishes[stage][microbatch], link_ready
            ) + stage_send_seconds[stage]

    makespan = send_finishes[-1][-1]
    round_trip = send_finishes[-1][0]  # 单个microbatch的全程延迟
    busy_slot_seconds = microbatches * sum(stage_compute_seconds) #所有stage实际工作总时间
    available_slot_seconds = stage_count * makespan #所有stage可用总时间
    idle_slot_seconds = max(0.0, available_slot_seconds - busy_slot_seconds) #所有stage空闲总时间
    # 稳态每轮间隔：瓶颈资源（最慢stage算力或最慢链路）的占用 × microbatch数，
    # 且不低于单个microbatch的流水全程；用于token间可流水的稳态吞吐口径。
    cycle = max(max(stage_compute_seconds), max(stage_send_seconds))
    steady_interval = max(microbatches * cycle, round_trip)
    return {
        "makespan_seconds": makespan,
        "first_microbatch_latency_seconds": round_trip,
        "steady_state_interval_seconds": steady_interval,
        "stage_service_seconds": services,
        "critical_stage_index": max(range(stage_count), key=services.__getitem__),
        "average_stage_utilization": (
            busy_slot_seconds / available_slot_seconds if available_slot_seconds else None
        ),
        "bubble_fraction": (
            idle_slot_seconds / available_slot_seconds if available_slot_seconds else None
        ),
        "bubble_slot_seconds": idle_slot_seconds,
        "balanced_pipeline_efficiency": microbatches / (microbatches + stage_count - 1),
        "completion_matrix_seconds": finishes,
    }


def _stage_ranges(partition: list[int]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = 0
    for count in partition:
        ranges.append((start, start + count - 1))
        start += count
    return ranges
