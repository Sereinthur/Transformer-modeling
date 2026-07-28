"""算子组合、通信重叠与PP阶段汇总。"""

from __future__ import annotations

from collections import defaultdict
from math import ceil
from typing import Any

from ..communication import collective_profile
from ..core import OperatorCostModel
from ..operators import OperatorContext, OperatorEstimate, get_operator
from ..parallel.pipeline.schedule import pipeline_schedule


def _group_layer_operators(layers) -> list[tuple[object, int]]:
    """相同配置的算子聚合，避免结果中重复数千个节点。"""

    grouped: dict[tuple[str, str, str], list[object | int]] = {}
    for layer in layers:
        if layer.residual.type == "attnres":
            # Block AttnRes先聚合深度表示，再经PreNorm进入Attention/FFN。
            sequence = (
                (layer.residual, 1), (layer.norm, 1), (layer.attention, 1),
                (layer.residual, 1), (layer.norm, 1), (layer.ffn, 1),
            )
        else:
            sequence = (
                (layer.norm, 1), (layer.attention, 1), (layer.residual, 1),
                (layer.norm, 1), (layer.ffn, 1), (layer.residual, 1),
            )
        for spec, multiplier in sequence:
            key = spec.signature()
            if key not in grouped:
                grouped[key] = [spec, 0]
            grouped[key][1] = int(grouped[key][1]) + multiplier
    return [(value[0], int(value[1])) for value in grouped.values()]


def build_estimates(config, layers, phase: str, batch_size: int,
                    token_length: int, attention_length: int,
                    include_embedding: bool, include_output: bool) -> list[OperatorEstimate]:
    estimates: list[OperatorEstimate] = []

    def add(spec, count=1):
        context = OperatorContext(
            config, phase, batch_size, token_length, attention_length, count
        )
        operator = get_operator(spec.type)
        estimates.append(
            operator.prefill_cost(spec, context)
            if phase == "prefill" else operator.decode_cost(spec, context)
        )

    if include_embedding:
        add(config.model.embedding)
    for spec, count in _group_layer_operators(layers):
        add(spec, count)
    if include_output:
        # AttnRes模型在Final Norm前还需对最终partial block做一次深度聚合。
        if layers and layers[-1].residual.type == "attnres":
            add(layers[-1].residual)
        add(config.model.output.norm)
        add(config.model.output.head)
        add(config.model.output.sampling)
    return estimates


def _operator_result(config, estimate: OperatorEstimate, phase: str) -> dict[str, Any]:
    roofline = OperatorCostModel(config)
    local_costs = [roofline.estimate(item, phase) for item in estimate.work_items]
    local_time = sum(item["time_seconds"]["estimated"] for item in local_costs)
    communication = [
        collective_profile(config, request)
        for request in estimate.communication_requests
    ]
    comm_time = sum(item["time_seconds"]["estimated"] for item in communication)
    if comm_time:
        tp_time = sum(
            item["time_seconds"]["estimated"] for item in communication
            if item["collective"]["group"] == "tp"
        )
        ep_time = comm_time - tp_time
        weighted_rho = (
            (
                tp_time * (
                    config.execution.prefill_tp_overlap_rho
                    if phase == "prefill" else config.execution.decode_tp_overlap_rho
                )
                + ep_time * config.execution.ep_overlap_rho
            ) / comm_time
        )
        estimated_time = max(local_time, comm_time) + weighted_rho * min(local_time, comm_time)
    else:
        estimated_time = local_time
    logical_ops = sum(item["ops"]["logical"] for item in local_costs)
    executed_ops = sum(item["ops"]["executed"] for item in local_costs)
    hbm = sum(item["hbm_payload_bytes"] for item in local_costs)
    return {
        "type": estimate.type_id,
        "中文名称": estimate.chinese_name,
        "occurrences": estimate.occurrence_count,
        "capacity": {
            "global_parameters": estimate.global_parameters,
            "local_parameters": estimate.local_parameters,
            "parameter_breakdown": estimate.parameter_breakdown,
            "persistent_state_bytes": estimate.persistent_state_bytes,
            "state_breakdown": estimate.state_breakdown,
            "temporary_bytes": estimate.temporary_bytes,
        },
        "work": {
            "logical_ops": logical_ops, "executed_ops": executed_ops,
            "hbm_payload_bytes": hbm,
        },
        "time_seconds": {
            "local": local_time, "communication": comm_time,
            "estimated": estimated_time,
        },
        "suboperators": local_costs,
        "communication": communication,
        "confidence": estimate.confidence,
        "performance_complete": estimate.performance_complete,
        "assumptions": estimate.assumptions,
    }


def summarize_stage(config, layers, phase: str, batch_size: int,
                    token_length: int, attention_length: int,
                    include_embedding: bool, include_output: bool,
                    details: bool = True) -> dict[str, Any]:
    estimates = build_estimates(
        config, layers, phase, batch_size, token_length, attention_length,
        include_embedding, include_output,
    )
    results = [_operator_result(config, item, phase) for item in estimates]
    latency = sum(item["time_seconds"]["estimated"] for item in results)
    known = all(item["performance_complete"] for item in results)
    summary = {
        "latency_seconds": latency,
        "modeled_proxy_latency_seconds": latency,
        "known_latency_lower_bound_seconds": latency,
        "performance_complete": known,
        "ops": {
            "logical": sum(item["work"]["logical_ops"] for item in results),
            "executed": sum(item["work"]["executed_ops"] for item in results),
        },
        "hbm_payload_bytes": sum(item["work"]["hbm_payload_bytes"] for item in results),
        "communication_seconds": sum(
            item["time_seconds"]["communication"] for item in results
        ),
    }
    if details:
        summary["operators"] = results
    return summary


def _layer_weights(config, layers) -> list[float]:
    """用当前请求的Prefill+全部Decode近似成本平衡异构层。"""

    weights = []
    active = min(config.parallelism.expert_parallel, config.serving.batch_size)
    local_batch = ceil(config.serving.batch_size / active)
    for layer in layers:
        prefill = summarize_stage(
            config, [layer], "prefill", local_batch, config.serving.prompt_length,
            config.serving.prompt_length, False, False, False,
        )["latency_seconds"]
        decode = summarize_stage(
            config, [layer], "decode", local_batch, 1,
            config.serving.prompt_length, False, False, False,
        )["latency_seconds"]
        weights.append(prefill + max(0, config.serving.output_length - 1) * decode)
    return weights


def balanced_layer_partition(config, layers) -> list[list[object]]:
    """连续贪心分段；粗粒度模型中优先保证简单和可解释。"""

    pp = config.parallelism.pipeline_parallel
    boundaries = config.parallelism.pipeline_stage_boundaries
    if boundaries is not None:
        points = (0, *boundaries, len(layers))
        return [list(layers[points[index]:points[index + 1]]) for index in range(pp)]
    if pp == 1:
        return [list(layers)]
    if not config.hardware.performance_supported:
        base, remainder = divmod(len(layers), pp)
        partitions, start = [], 0
        for stage in range(pp):
            count = base + (1 if stage < remainder else 0)
            partitions.append(list(layers[start:start + count]))
            start += count
        return partitions
    weights = _layer_weights(config, layers)
    remaining_weight, start = sum(weights), 0
    partitions: list[list[object]] = []
    for stage in range(pp - 1):
        stages_left = pp - stage
        target = remaining_weight / stages_left
        end, accumulated = start, 0.0
        max_end = len(layers) - (stages_left - 1)
        while end < max_end:
            candidate = accumulated + weights[end]
            if end > start and candidate > target:
                break
            accumulated, end = candidate, end + 1
        partitions.append(list(layers[start:end]))
        remaining_weight -= accumulated
        start = end
    partitions.append(list(layers[start:]))
    return partitions


def summarize_parallel_phase(config, phase: str, attention_length: int,
                             details: bool = True) -> dict[str, Any]:
    layers = config.model.expanded_layers()
    stages = balanced_layer_partition(config, layers)
    microbatches = config.parallelism.pipeline_microbatches
    micro_batch = config.serving.batch_size // microbatches
    active_ep = min(config.parallelism.expert_parallel, micro_batch)
    local_batch = ceil(micro_batch / active_ep)
    token_length = config.serving.prompt_length if phase == "prefill" else 1
    stage_results = [
        summarize_stage(
            config, stage_layers, phase, local_batch, token_length, attention_length,
            index == 0, index == len(stages) - 1, details,
        )
        for index, stage_layers in enumerate(stages)
    ]
    pp = len(stages)
    if pp > 1:
        payload = local_batch * token_length * config.model.hidden_size * config.model.activation_bytes
        bandwidth = float(config.hardware.interconnect.effective_pipeline_bandwidth or 1)
        alpha = float(config.hardware.interconnect.effective_pipeline_latency or 0)
        sends = [alpha + payload / bandwidth] * (pp - 1) + [0.0]
    else:
        sends = [0.0]
    schedule = pipeline_schedule(
        [stage["latency_seconds"] for stage in stage_results], sends, microbatches
    )
    latency = schedule["makespan_seconds"]
    traffic = microbatches * sum(stage["hbm_payload_bytes"] for stage in stage_results)
    effective_hbm = OperatorCostModel(config).effective_memory_bandwidth
    return {
        "latency_seconds": latency,
        "modeled_proxy_latency_seconds": latency,
        "known_latency_lower_bound_seconds": latency,
        "performance_complete": all(stage["performance_complete"] for stage in stage_results),
        "ops": {
            "logical": microbatches * sum(stage["ops"]["logical"] for stage in stage_results),
            "executed": microbatches * sum(stage["ops"]["executed"] for stage in stage_results),
        },
        "hbm_payload_bytes": traffic,
        "hbm": {
            "payload_bytes": traffic,
            "effective_bandwidth_bytes_per_second": effective_hbm,
            "average_achieved_bandwidth_bytes_per_second": traffic / latency if latency else None,
        },
        "communication_seconds": microbatches * sum(stage["communication_seconds"] for stage in stage_results) + sum(sends),
        "active_ep_ranks": active_ep,
        "ep_utilization": active_ep / config.parallelism.expert_parallel,
        "pipeline_schedule": schedule,
        "stages": stage_results if details else [],
    }
