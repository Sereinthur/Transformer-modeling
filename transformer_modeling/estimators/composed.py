"""Schema v2统一算子组合估算器。"""

from __future__ import annotations

from dataclasses import replace
from math import ceil
from typing import Any

from ..operators.base import blocked_bytes
from .composed_phase import (
    balanced_layer_partition, build_estimates, summarize_parallel_phase,
)


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


def capacity_summary(config) -> dict[str, Any]:
    """按算子参数、持久状态和最大临时缓冲汇总关键rank容量。"""

    model, serving = config.model, config.serving
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
        # 单Stage且Embedding/LM Head共享时，物理容量和逻辑参数都只保留一份。
        if tied and len(stages) == 1:
            head = next((item for item in estimates if item.type_id == "lm_head"), None)
            if head:
                local_parameters -= head.local_parameters
                global_parameters -= head.global_parameters
        local_parameters += _extra_local_parameters(config)
        weights = blocked_bytes(config, local_parameters, model.weight_dtype)
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
            breakdown["unmodeled_parameters"] = _extra_local_parameters(config)
        state_results: dict[str, int] = {}
        for item in estimates:
            for name, value in item.state_breakdown.items():
                state_results[name] = state_results.get(name, 0) + value
        stage_results.append({
            "stage_index": index,
            "layer_count": len(layers),
            "local_parameters": local_parameters,
            "weights_bytes": weights,
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
            "容量不足不阻止理论性能估算。",
        ],
    }


def _decode_summary(config, details: bool) -> tuple[dict[str, Any], float]:
    steps = max(0, config.serving.output_length - 1)
    if not steps:
        return {
            "steps": 0, "total_latency_seconds": 0.0,
            "device_inter_token_interval": {"mean_seconds": None},
            "first_step": None, "last_step": None,
            "performance_complete": True,
        }, 0.0
    first = summarize_parallel_phase(
        config, "decode", config.serving.prompt_length, details
    )
    last_length = config.serving.prompt_length + steps - 1
    last = first if steps == 1 else summarize_parallel_phase(
        config, "decode", last_length, details
    )
    mean = (first["latency_seconds"] + last["latency_seconds"]) / 2
    total = mean * steps
    return {
        "steps": steps,
        "total_latency_seconds": total,
        "device_inter_token_interval": {
            "mean_seconds": mean,
            "first_seconds": first["latency_seconds"],
            "last_seconds": last["latency_seconds"],
        },
        "first_step": first,
        "last_step": last,
        "performance_complete": first["performance_complete"] and last["performance_complete"],
        "steady_state_output_tokens_per_second": config.serving.batch_size / mean if mean else None,
    }, total


def _pd_summary(config, prefill_seconds: float, decode_seconds: float,
                capacity: dict[str, Any]) -> dict[str, Any]:
    deployment = config.deployment
    if deployment.mode != "disaggregated":
        return {"enabled": False, "handoff_visible_seconds": 0.0}
    critical = capacity["per_stage"][capacity["critical_stage_index"]]
    payload = critical["persistent_state_bytes"]
    bandwidth = float(deployment.transfer_bandwidth_bytes_per_second or 1)
    transfer = deployment.transfer_latency_seconds + payload / bandwidth
    combined = max(prefill_seconds, transfer) + deployment.overlap_rho * min(prefill_seconds, transfer)
    visible = max(0.0, combined - prefill_seconds)
    batch = config.serving.batch_size
    prefill_rate = deployment.prefill_replicas * batch / prefill_seconds if prefill_seconds else None
    decode_rate = deployment.decode_replicas * batch / decode_seconds if decode_seconds else None
    link_rate = bandwidth / payload if payload else None
    candidates = {"prefill_pool": prefill_rate, "decode_pool": decode_rate, "transfer_link": link_rate}
    valid = {key: value for key, value in candidates.items() if value is not None}
    return {
        "enabled": True, "payload_bytes_per_rank": payload,
        "transfer_seconds": transfer, "handoff_visible_seconds": visible,
        "max_requests_per_second": candidates,
        "system_max_requests_per_second": min(valid.values()) if valid else None,
        "bottleneck": min(valid, key=valid.get) if valid else None,
    }


def _model_summary(config, capacity: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    layers = config.model.expanded_layers()
    for layer in layers:
        for spec in (layer.attention, layer.ffn, layer.residual, layer.residual):
            counts[spec.type] = counts.get(spec.type, 0) + 1
    if layers and layers[-1].residual.type == "attnres":
        counts["attnres"] = counts.get("attnres", 0) + 1
    return {
        "id": config.model.model_id, "name": config.model.name,
        "layers": config.model.layer_count, "hidden_size": config.model.hidden_size,
        "parameters": capacity["model_logical_parameters"],
        "weight_dtype": config.model.weight_dtype,
        "operator_mix": counts, "metadata": config.model.metadata,
    }


def estimate_composed(config, details: bool = True, include_scaling: bool = True) -> dict[str, Any]:
    capacity = capacity_summary(config)
    incomplete_extra = config.model.extra_parameters > 0
    warnings: list[str] = []
    if not capacity["capacity_feasible"]:
        warnings.append("关键rank容量不足；性能结果是假设权重与状态可驻留时的理论值。")
    if incomplete_extra:
        warnings.append(
            "模型包含未归属参数；这些参数计入容量，延迟只对应已建模代理结构，"
            "不能解释为严格上界或下界。"
        )
    result: dict[str, Any] = {
        "schema_version": 2,
        "model": _model_summary(config, capacity),
        "workload": {
            "batch_size": config.serving.batch_size,
            "prompt_length": config.serving.prompt_length,
            "output_length_including_first_token": config.serving.output_length,
        },
        "parallelism": {
            "tensor_parallel": config.parallelism.tensor_parallel,
            "expert_parallel": config.parallelism.expert_parallel,
            "pipeline_parallel": config.parallelism.pipeline_parallel,
            "pipeline_microbatches": config.parallelism.pipeline_microbatches,
            "pipeline_stage_boundaries": config.parallelism.pipeline_stage_boundaries,
            "device_count": config.hardware.device_count,
            "topology": config.hardware.interconnect.topology,
        },
        "capacity": capacity,
        "validity": {
            "capacity_feasible": capacity["capacity_feasible"],
            "performance_is_theoretical": not capacity["capacity_feasible"],
            "performance_complete": not incomplete_extra,
        },
        "warnings": warnings,
    }
    if not config.hardware.performance_supported:
        result["performance"] = None
        result["errors"] = ["容量已计算，但硬件缺少当前计算格式的原生吞吐。"]
        return result

    prefill = summarize_parallel_phase(
        config, "prefill", config.serving.prompt_length, details
    )
    decode, decode_seconds = _decode_summary(config, details)
    complete = prefill["performance_complete"] and decode["performance_complete"] and not incomplete_extra
    result["validity"]["performance_complete"] = complete
    prefill_seconds = prefill["latency_seconds"]
    pd = _pd_summary(config, prefill_seconds, decode_seconds, capacity)
    completion = prefill_seconds + pd["handoff_visible_seconds"] + decode_seconds
    performance = {
        "prefill": prefill,
        "decode": decode,
        "first_token": {"ttft_seconds": prefill_seconds},
        "request": {"completion_latency_seconds": completion},
        "throughput": {
            "batch_average_output_tokens_per_second": (
                config.serving.batch_size * config.serving.output_length / completion
            ),
            "batch_average_requests_per_second": config.serving.batch_size / completion,
            "steady_state_decode_output_tokens_per_second": decode.get(
                "steady_state_output_tokens_per_second"
            ),
        },
        "pd_disaggregation": pd,
        "modeled_proxy_latency_seconds": completion,
        "known_latency_lower_bound_seconds": completion,
    }
    result["performance"] = performance
    if include_scaling and config.hardware.device_count > 1:
        parallel = replace(
            config.parallelism, tensor_parallel=1, expert_parallel=1,
            pipeline_parallel=1, pipeline_microbatches=1,
            pipeline_stage_boundaries=None,
        )
        baseline_config = replace(
            config, parallelism=parallel,
            hardware=replace(config.hardware, device_count=1),
        )
        baseline = estimate_composed(baseline_config, False, False)
        baseline_perf = baseline.get("performance")
        if baseline_perf:
            baseline_latency = baseline_perf["request"]["completion_latency_seconds"]
            speedup = baseline_latency / completion
            performance["scaling"] = {
                "available": True, "baseline_completion_seconds": baseline_latency,
                "speedup": speedup,
                "parallel_efficiency": speedup / config.hardware.device_count,
            }
    return result
