"""关键rank容量汇总：参数、持久状态与工作区峰值。"""

from __future__ import annotations

from math import ceil
from typing import Any

from ..operators.base import blocked_bytes
from .composed_phase import balanced_layer_partition, build_estimates


def _extra_local_parameters(config) -> int:
    model, parallel = config.model, config.parallelism
    divisor = {
        "replicated": 1,
        "tp": parallel.tensor_parallel,
        "ep": parallel.expert_parallel,
        "tp_ep": parallel.tensor_parallel * parallel.expert_parallel,
    }[model.extra_parameter_sharding]
    # 未归属参数视为分布在全部层中，因此除声明的TP/EP外还随PP Stage切分。
    return ceil(model.extra_parameters / (divisor * parallel.pipeline_parallel))


def _accumulate_dtype(buckets: dict[str, int], item, default_dtype: str, sign: int = 1) -> None:
    """把单个算子的本地参数按权重精度分桶累加；空映射表示全部按模型精度。"""

    by_dtype = item.local_parameters_by_dtype or {default_dtype: item.local_parameters}
    for dtype, value in by_dtype.items():
        buckets[dtype] = buckets.get(dtype, 0) + sign * int(value)


def capacity_summary(config, stages: list[list[object]] | None = None) -> dict[str, Any]:
    """按算子参数、持久状态和最大临时缓冲汇总关键rank容量。"""

    model, serving = config.model, config.serving
    if stages is None:
        stages = balanced_layer_partition(config, model.expanded_layers())
    active_ep = min(config.parallelism.expert_parallel, serving.batch_size)
    local_batch = ceil(serving.batch_size / active_ep)
    stage_results = []
    logical_parameters = 0
    tied = bool(model.embedding.get("tied_lm_head", True))
    for index, layers in enumerate(stages):
        estimates = build_estimates(
            config, layers, "prefill", local_batch, serving.prompt_length,
            serving.prompt_length, index == 0, index == len(stages) - 1,
        )
        local_parameters = sum(item.local_parameters for item in estimates)
        global_parameters = sum(item.global_parameters for item in estimates)
        buckets: dict[str, int] = {}
        for item in estimates:
            _accumulate_dtype(buckets, item, model.weight_dtype)
        # 单Stage且Embedding/LM Head共享时，物理容量和逻辑参数都只保留一份。
        if tied and len(stages) == 1:
            head = next((item for item in estimates if item.type_id == "lm_head"), None)
            if head:
                local_parameters -= head.local_parameters
                global_parameters -= head.global_parameters
                _accumulate_dtype(buckets, head, model.weight_dtype, -1)
        extra = _extra_local_parameters(config)
        local_parameters += extra
        if extra:
            buckets[model.weight_dtype] = buckets.get(model.weight_dtype, 0) + extra
        weights_by_dtype = {
            dtype: blocked_bytes(config, count, dtype)
            for dtype, count in sorted(buckets.items()) if count > 0
        }
        weights = sum(weights_by_dtype.values())
        states = sum(item.persistent_state_bytes for item in estimates)
        activations = max((item.temporary_bytes for item in estimates), default=0)
        communication = max(
            (request.payload_bytes for item in estimates for request in item.communication_requests),
            default=0,
        )
        # 峰值按同一算子的临时空间与通信buffer共同存活来估计，避免把
        # 不同执行时刻的两个独立最大值直接相加。
        workspace = max((
            item.temporary_bytes + max(
                (request.payload_bytes for request in item.communication_requests),
                default=0,
            )
            for item in estimates
        ), default=0)
        total = (
            weights + states + ceil(workspace)
            + config.hardware.runtime_reserved_bytes
            + config.hardware.baseline_unavailable_bytes
        )
        breakdown: dict[str, int] = {}
        for item in estimates:
            for name, value in item.parameter_breakdown.items():
                breakdown[name] = breakdown.get(name, 0) + value
        if model.extra_parameters:
            breakdown["unmodeled_parameters"] = extra
        state_results: dict[str, int] = {}
        for item in estimates:
            for name, value in item.state_breakdown.items():
                state_results[name] = state_results.get(name, 0) + value
        stage_results.append({
            "stage_index": index,
            "layer_count": len(layers),
            "local_parameters": local_parameters,
            "weights_bytes": weights,
            "weights_by_dtype": weights_by_dtype,
            "weights_by_operator": breakdown,
            "persistent_state_bytes": states,
            "states_by_operator": state_results,
            "activations_peak_bytes": activations,
            "communication_buffer_peak_bytes": ceil(communication),
            "combined_workspace_peak_bytes": ceil(workspace),
            "runtime_reserved_bytes": config.hardware.runtime_reserved_bytes,
            "baseline_unavailable_bytes": config.hardware.baseline_unavailable_bytes,
            "peak_total_bytes": total,
            "headroom_bytes": config.hardware.device_memory_capacity_bytes - total,
        })
        logical_parameters += global_parameters
    # PP物理复制不应重复计入模型的逻辑参数量。
    if tied and len(stages) > 1:
        logical_parameters -= model.padded_vocab_size * model.hidden_size
    logical_parameters += model.extra_parameters
    critical = max(stage_results, key=lambda item: item["peak_total_bytes"])
    capacity = config.hardware.device_memory_capacity_bytes
    return {
        "capacity_feasible": critical["peak_total_bytes"] <= capacity,
        "performance_is_theoretical": critical["peak_total_bytes"] > capacity,
        "scope": "critical_rank",
        "model_logical_parameters": logical_parameters,
        "critical_stage_index": critical["stage_index"],
        "required_bytes_per_critical_rank": critical["peak_total_bytes"],
        "available_bytes_per_rank": capacity,
        "capacity_shortfall_bytes": max(0, critical["peak_total_bytes"] - capacity),
        "headroom_bytes": capacity - critical["peak_total_bytes"],
        "per_stage": stage_results,
        "notes": [
            "权重和持久状态求和；工作区按单算子临时空间与其通信buffer的共同峰值近似。",
            "算子级权重精度只改变权重字节数（容量与HBM流量），峰值算力仍按模型计算精度选取。",
            "容量不足不阻止理论性能估算。",
        ],
    }
