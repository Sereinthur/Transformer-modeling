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
    """用flow-shop递推计算前向流水的填充、稳态和排空时间。"""

    stage_count = len(stage_compute_seconds)
    if len(stage_send_seconds) != stage_count:
        raise ValueError("stage_send_seconds must match stage_compute_seconds")
    finishes = [[0.0 for _ in range(microbatches)] for _ in range(stage_count)]
    starts = [[0.0 for _ in range(microbatches)] for _ in range(stage_count)]
    services = [
        stage_compute_seconds[index] + stage_send_seconds[index]
        for index in range(stage_count)
    ]
    for microbatch in range(microbatches):
        for stage in range(stage_count):
            upstream_ready = finishes[stage - 1][microbatch] if stage else 0.0
            stage_ready = finishes[stage][microbatch - 1] if microbatch else 0.0
            starts[stage][microbatch] = max(upstream_ready, stage_ready)
            finishes[stage][microbatch] = starts[stage][microbatch] + services[stage]

    makespan = finishes[-1][-1]
    busy_slot_seconds = microbatches * sum(services) #所有stage实际工作总时间
    available_slot_seconds = stage_count * makespan #所有stage可用总时间
    idle_slot_seconds = max(0.0, available_slot_seconds - busy_slot_seconds) #所有stage空闲总时间
    return {
        "makespan_seconds": makespan,
        "first_microbatch_latency_seconds": finishes[-1][0],
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
