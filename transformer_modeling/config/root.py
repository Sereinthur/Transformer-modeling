"""Schema v3 顶层配置装配与跨字段校验。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .common import non_negative, positive
from .deployment import DeploymentSpec
from .execution import ExecutionSpec, ParallelSpec
from .hardware import HardwareSpec
from .model import ModelSpec
from .serving import ServingSpec


def expand_preset_config(data: dict[str, Any]) -> dict[str, Any]:
    """Resolve a scenario's preset_id into a standalone Schema v3 model."""

    if not isinstance(data, dict):
        raise ValueError("configuration must be an object")
    expanded = deepcopy(data)
    preset_id = expanded.get("preset_id")
    if preset_id is None:
        return expanded
    if not isinstance(preset_id, str) or not preset_id:
        raise ValueError("preset_id must be a non-empty string")
    if "model" in expanded:
        raise ValueError("configuration must provide either preset_id or model, not both")
    from ..models import resolve_model_definition

    resolved = resolve_model_definition(preset_id=preset_id)
    expanded["model"] = resolved["resolved_model"]
    expanded.pop("preset_id")
    return expanded


@dataclass(frozen=True)
class Config:
    schema_version: int
    hardware: HardwareSpec
    model: ModelSpec
    serving: ServingSpec
    execution: ExecutionSpec
    parallelism: ParallelSpec
    deployment: DeploymentSpec

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        data = expand_preset_config(data)
        version = int(data.get("schema_version", 0))
        if version != 3:
            if version in {1, 2}:
                raise ValueError(
                    f"configuration version expired: schema_version {version} is unsupported; "
                    "use schema_version 3 with layer operations"
                )
            raise ValueError(f"unsupported schema_version {version}; expected 3")
        model_data = data.get("model", {})
        model = ModelSpec.from_dict(model_data)
        config = cls(
            schema_version=version,
            hardware=HardwareSpec.from_dict(data.get("hardware", {}), model.weight_dtype),
            model=model,
            serving=ServingSpec.from_dict(data.get("serving", {})),
            execution=ExecutionSpec.from_dict(data.get("execution", {}), model_data),
            parallelism=ParallelSpec.from_dict(data.get("parallelism", {})),
            deployment=DeploymentSpec.from_dict(data.get("deployment", {})),
        )
        config._validate()
        return config

    def _validate(self) -> None:
        tp = self.parallelism.tensor_parallel
        ep = self.parallelism.expert_parallel
        pp = self.parallelism.pipeline_parallel
        if self.hardware.device_count != tp * ep * pp:
            raise ValueError(
                "hardware.device_count must equal tensor_parallel * "
                "expert_parallel * pipeline_parallel"
            )
        if pp > self.model.layer_count:
            raise ValueError("pipeline_parallel cannot exceed model layer_count")
        boundaries = self.parallelism.pipeline_stage_boundaries
        if boundaries is not None:
            if len(boundaries) != pp - 1:
                raise ValueError(
                    "pipeline_stage_boundaries must contain PP-1 cumulative layer indices"
                )
            if tuple(sorted(set(boundaries))) != boundaries:
                raise ValueError("pipeline_stage_boundaries must be strictly increasing")
            if boundaries[0] <= 0 or boundaries[-1] >= self.model.layer_count:
                raise ValueError("pipeline_stage_boundaries must be between 1 and layer_count-1")
        micro = self.parallelism.pipeline_microbatches
        if micro > self.serving.batch_size or self.serving.batch_size % micro:
            raise ValueError("batch_size must be divisible by pipeline_microbatches")
        if ep > 1 and not self.model.uses_operator("moe"):
            raise ValueError("expert_parallel > 1 requires at least one moe operator")

        from ..operators import get_operator

        expected = [(self.model.embedding, "embedding")]
        for layer in (
            *self.model.layer_prefix,
            *self.model.layer_pattern,
            *self.model.layer_suffix,
        ):
            expected.extend((operation, "layer") for operation in layer.main_operators)
        expected.extend((
            (self.model.output.norm, "norm"),
            (self.model.output.head, "output"),
            (self.model.output.sampling, "output"),
        ))
        for spec, slot in expected:
            operator = get_operator(spec.type)
            allowed_slots = {"any", slot}
            if slot == "layer":
                allowed_slots |= {"norm", "attention", "ffn", "residual"}
            if operator.slot not in allowed_slots:
                raise ValueError(
                    f"operator {spec.type} belongs to {operator.slot}, not {slot} slot"
                )
            operator.validate(spec, self)

        cache = self.serving.prefix_cache
        if cache.kda_cached_prefix_tokens > self.serving.prompt_length:
            raise ValueError("kda_cached_prefix_tokens cannot exceed prompt_length")
        if cache.mla_average_matched_tokens > self.serving.prompt_length:
            raise ValueError("mla_average_matched_tokens cannot exceed prompt_length")

        interconnect = self.hardware.interconnect
        if tp > 1 or ep > 1 or pp > 1:
            if interconnect.topology not in {"ring", "bus", "crossbar", "mesh"}:
                raise ValueError("interconnect topology must be ring, bus, crossbar, or mesh")
            bandwidth = interconnect.effective_channel_bandwidth_bytes_per_second
            latency = interconnect.collective_step_latency_seconds
            if bandwidth is None:
                raise ValueError("interconnect effective bandwidth is required for parallel execution")
            positive("interconnect effective bandwidth", bandwidth)
            if latency is None or latency < 0:
                raise ValueError("interconnect collective latency must be non-negative")
            if (interconnect.mesh_rows is None) != (interconnect.mesh_columns is None):
                raise ValueError("mesh_rows and mesh_columns must be provided together")
            if interconnect.mesh_rows is not None:
                positive("interconnect mesh_rows", interconnect.mesh_rows)
                positive("interconnect mesh_columns", interconnect.mesh_columns)
                if tp > 1 and ep > 1 and tp != ep:
                    raise ValueError(
                        "explicit mesh dimensions require equal TP and EP group sizes"
                    )
                mesh_ranks = max(tp, ep)
                if interconnect.mesh_rows * interconnect.mesh_columns != mesh_ranks:
                    raise ValueError(
                        "mesh_rows * mesh_columns must equal the largest TP or EP group size"
                    )
            pipeline_bandwidth = interconnect.effective_pipeline_bandwidth
            pipeline_latency = interconnect.effective_pipeline_latency
            if pipeline_bandwidth is None:
                raise ValueError("pipeline effective bandwidth is required for parallel execution")
            positive("pipeline effective bandwidth", pipeline_bandwidth)
            if pipeline_latency is None:
                raise ValueError("pipeline transfer latency is required for parallel execution")
            non_negative("pipeline transfer latency", pipeline_latency)
        required = self.serving.prompt_length + self.serving.output_length - 1
        if self.serving.max_sequence_length and required > self.serving.max_sequence_length:
            raise ValueError("request exceeds max_sequence_length")
