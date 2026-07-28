"""供算子组装器使用的统一Collective解析成本。"""

from __future__ import annotations

from math import ceil, log2
from typing import Any

from ..operators.base import CommunicationRequest
from .topologies import (
    bus_all_gather_seconds, bus_all_reduce_seconds,
    crossbar_all_gather_seconds, crossbar_all_reduce_seconds,
    mesh_all_gather_seconds, mesh_all_reduce_seconds, mesh_dimensions,
    ring_all_gather_seconds, ring_all_reduce_seconds,
)


def _all_to_all_profile(topology: str, payload: float, ranks: int,
                        bandwidth: float, alpha: float) -> tuple[float, int, str, str]:
    """S为每rank全部目标的API payload；返回关键rank解析时间。"""

    if ranks == 1:
        return 0.0, 0, "none", "单rank无需All-to-All。"
    remote = (ranks - 1) / ranks * payload
    if topology == "ring":
        return (
            (ranks - 1) * alpha + (ranks - 1) / 2 * payload / bandwidth,
            ranks - 1, "ring_forwarded_all_to_all",
            "单向Ring按平均转发距离估算，不显式模拟链路拥塞。",
        )
    if topology == "bus":
        return (
            ranks * (ranks - 1) * alpha + (ranks - 1) * payload / bandwidth,
            ranks * (ranks - 1), "serialized_bus_all_to_all",
            "共享Bus串行传输所有远端消息。",
        )
    if topology == "crossbar":
        return (
            (ranks - 1) * alpha + remote / bandwidth,
            ranks - 1, "pairwise_exchange_all_to_all",
            "无阻塞Crossbar采用Pairwise Exchange。",
        )
    rows, columns = mesh_dimensions(ranks)
    diameter = rows + columns - 2
    base = (ranks - 1) * alpha + remote / bandwidth
    return (
        max(1, diameter) * base, max(1, diameter) * (ranks - 1),
        "dimension_ordered_mesh_all_to_all",
        "Mesh按直径对Pairwise成本保守缩放，不模拟热点链路。",
    )


def collective_profile(config, request: CommunicationRequest) -> dict[str, Any]:
    """把算子声明的通信需求换算为时间和收发量。"""

    ranks = (
        config.parallelism.tensor_parallel
        if request.group == "tp" else config.parallelism.expert_parallel
    )
    interconnect = config.hardware.interconnect
    topology = interconnect.topology
    bandwidth = float(interconnect.effective_channel_bandwidth_bytes_per_second or 1)
    alpha = float(interconnect.collective_step_latency_seconds or 0)
    payload = request.payload_bytes
    if ranks == 1:
        seconds, steps, sent, received, algorithm = 0.0, 0, 0.0, 0.0, "none"
        assumption = "单rank无需集合通信。"
    elif request.kind == "all_to_all":
        seconds, steps, algorithm, assumption = _all_to_all_profile(
            topology, payload, ranks, bandwidth, alpha
        )
        sent = received = (ranks - 1) / ranks * payload
    elif topology == "ring":
        if request.kind == "all_reduce":
            seconds = ring_all_reduce_seconds(payload, ranks, bandwidth, alpha)
            sent = received = 2 * (ranks - 1) / ranks * payload
            steps, algorithm = 2 * (ranks - 1), "ring_reduce_scatter_all_gather"
        else:
            seconds = ring_all_gather_seconds(payload, ranks, bandwidth, alpha)
            sent = received = (ranks - 1) * payload
            steps, algorithm = ranks - 1, "ring_all_gather"
        assumption = "理想单向Ring，有效带宽已包含协议损失。"
    elif topology == "bus":
        function = bus_all_reduce_seconds if request.kind == "all_reduce" else bus_all_gather_seconds
        seconds = function(payload, ranks, bandwidth, alpha)
        sent, received, steps = payload, (ranks - 1) * payload, ranks
        algorithm, assumption = f"shared_bus_{request.kind}", "共享Bus串行近似。"
    elif topology == "crossbar":
        function = crossbar_all_reduce_seconds if request.kind == "all_reduce" else crossbar_all_gather_seconds
        seconds = function(payload, ranks, bandwidth, alpha)
        steps = ceil(log2(ranks))
        sent = received = steps * payload if request.kind == "all_reduce" else (ranks - 1) * payload
        algorithm, assumption = f"recursive_doubling_{request.kind}", "无阻塞Crossbar近似。"
    else:
        rows, columns = mesh_dimensions(ranks)
        function = mesh_all_reduce_seconds if request.kind == "all_reduce" else mesh_all_gather_seconds
        seconds = function(payload, ranks, bandwidth, alpha, rows, columns)
        diameter = rows + columns - 2
        sent = received = (diameter if request.kind == "all_reduce" else ranks - 1) * payload
        steps, algorithm = diameter, f"mesh_{request.kind}"
        assumption = "自动近方形Mesh维度有序路由近似。"
    total = seconds * request.occurrences
    return {
        "name": request.name,
        "中文名称": request.chinese_name,
        "kind": "communication",
        "collective": {
            "type": request.kind, "group": request.group, "ranks": ranks,
            "topology": topology, "algorithm": algorithm,
            "occurrences": request.occurrences, "steps_per_occurrence": steps,
            "api_payload_bytes_per_rank_per_occurrence": payload,
            "wire_bytes_sent_per_rank_per_occurrence": sent,
            "wire_bytes_received_per_rank_per_occurrence": received,
            "model_assumption": assumption,
        },
        "hbm_payload_bytes": (payload + received) * request.occurrences,
        "time_seconds": {"communication": total, "estimated": total},
        "bottleneck": "communication",
    }
