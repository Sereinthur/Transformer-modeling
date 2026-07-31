"""Schema v2统一算子组合估算器。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .capacity import capacity_summary
from .composed_phase import balanced_layer_partition, summarize_parallel_phase


def _decode_summary(config, details: bool,
                    stages: list[list[object]] | None = None) -> tuple[dict[str, Any], float]:
    steps = max(0, config.serving.output_length - 1)
    if not steps:
        return {
            "steps": 0, "total_latency_seconds": 0.0,
            "device_inter_token_interval": {"mean_seconds": None},
            "first_step": None, "last_step": None,
            "performance_complete": True,
        }, 0.0
    first = summarize_parallel_phase(
        config, "decode", config.serving.prompt_length, details, stages
    )
    last_length = config.serving.prompt_length + steps - 1
    last = first if steps == 1 else summarize_parallel_phase(
        config, "decode", last_length, details, stages
    )
    mean = (first["latency_seconds"] + last["latency_seconds"]) / 2
    total = mean * steps
    # 稳态吞吐用流水稳态间隔：PP>1时token间可跨Stage流水，间隔受瓶颈Stage限制。
    steady = (
        first["pipeline_schedule"]["steady_state_interval_seconds"]
        + last["pipeline_schedule"]["steady_state_interval_seconds"]
    ) / 2
    return {
        "steps": steps,
        "total_latency_seconds": total,
        "device_inter_token_interval": {
            "mean_seconds": mean,
            "first_seconds": first["latency_seconds"],
            "last_seconds": last["latency_seconds"],
            "steady_state_seconds": steady,
        },
        "first_step": first,
        "last_step": last,
        "performance_complete": first["performance_complete"] and last["performance_complete"],
        "steady_state_output_tokens_per_second": config.serving.batch_size / steady if steady else None,
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
        # 每层骨架包含2×Norm、1×Attention、1×FFN和2×Residual。
        for spec in (layer.norm, layer.attention, layer.residual,
                     layer.norm, layer.ffn, layer.residual):
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
    # 层划分只依赖config，在一次估算内计算一次并透传给容量与各阶段。
    stages = balanced_layer_partition(config, config.model.expanded_layers())
    capacity = capacity_summary(config, stages)
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
        config, "prefill", config.serving.prompt_length, details, stages
    )
    decode, decode_seconds = _decode_summary(config, details, stages)
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
