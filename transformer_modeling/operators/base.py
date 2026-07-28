"""通用算子接口、成本数据和共享帮助函数。"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import TYPE_CHECKING, Any

from ..core.work_item import WorkItem, _gemm

if TYPE_CHECKING:
    from ..config import Config, OperatorSpec


@dataclass(frozen=True)
class CommunicationRequest:
    name: str
    chinese_name: str
    kind: str
    group: str
    payload_bytes: float
    occurrences: int


@dataclass
class OperatorEstimate:
    """一个逻辑算子的容量、工作量和通信账单。"""

    type_id: str
    chinese_name: str
    occurrence_count: int
    global_parameters: int = 0
    local_parameters: int = 0
    parameter_breakdown: dict[str, int] = field(default_factory=dict)
    persistent_state_bytes: int = 0
    state_breakdown: dict[str, int] = field(default_factory=dict)
    temporary_bytes: int = 0
    work_items: list[WorkItem] = field(default_factory=list)
    communication_requests: list[CommunicationRequest] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    confidence: str = "medium"
    performance_complete: bool = True


@dataclass(frozen=True)
class OperatorContext:
    config: "Config"
    phase: str
    batch_size: int
    token_length: int
    attention_length: int
    occurrence_count: int = 1

    @property
    def rows(self) -> int:
        return self.batch_size * self.token_length

    @property
    def tp(self) -> int:
        return self.config.parallelism.tensor_parallel

    @property
    def ep(self) -> int:
        return self.config.parallelism.expert_parallel

    @property
    def model(self):
        return self.config.model


class TransformerOperator:
    """所有槽位算子的稳定接口；芯片时间换算由公共成本引擎完成。"""

    type_id = "abstract"
    chinese_name = "抽象算子"
    slot = "any"
    implementations = {"default"}

    def validate(self, spec: "OperatorSpec", config: "Config") -> None:
        if spec.implementation not in self.implementations:
            choices = ", ".join(sorted(self.implementations))
            raise ValueError(
                f"operator {self.type_id} implementation must be one of: {choices}"
            )

    def estimate(self, spec: "OperatorSpec", ctx: OperatorContext) -> OperatorEstimate:
        raise NotImplementedError

    def parameter_cost(self, spec: "OperatorSpec", ctx: OperatorContext) -> dict[str, Any]:
        estimate = self.estimate(spec, ctx)
        return {
            "global_parameters": estimate.global_parameters,
            "local_parameters": estimate.local_parameters,
            "breakdown": estimate.parameter_breakdown,
        }

    def state_cost(self, spec: "OperatorSpec", ctx: OperatorContext) -> dict[str, Any]:
        estimate = self.estimate(spec, ctx)
        return {
            "persistent_state_bytes": estimate.persistent_state_bytes,
            "temporary_bytes": estimate.temporary_bytes,
            "breakdown": estimate.state_breakdown,
        }

    def prefill_cost(self, spec: "OperatorSpec", ctx: OperatorContext) -> OperatorEstimate:
        if ctx.phase != "prefill":
            raise ValueError("prefill_cost requires a prefill context")
        return self.estimate(spec, ctx)

    def decode_cost(self, spec: "OperatorSpec", ctx: OperatorContext) -> OperatorEstimate:
        if ctx.phase != "decode":
            raise ValueError("decode_cost requires a decode context")
        return self.estimate(spec, ctx)

    def parallel_requirements(
        self, spec: "OperatorSpec", ctx: OperatorContext
    ) -> list[CommunicationRequest]:
        return self.estimate(spec, ctx).communication_requests


def int_param(spec: "OperatorSpec", name: str, default: int | None = None) -> int:
    value = spec.get(name, default)
    if value is None:
        raise ValueError(f"operator {spec.type}.{name} is required")
    result = int(value)
    if result <= 0:
        raise ValueError(f"operator {spec.type}.{name} must be greater than zero")
    return result


def effective_element_bytes(config: "Config", dtype: str) -> float:
    raw = {
        "mxfp4": 0.5, "int4": 0.5, "mxfp8": 1.0, "fp8": 1.0,
        "int8": 1.0, "fp16": 2.0, "bf16": 2.0, "fp32": 4.0, "tf32": 4.0,
    }[dtype]
    if dtype not in {"mxfp4", "mxfp8"}:
        return raw
    quant = config.model.quantization
    return raw + quant.scale_bytes / quant.block_size


def blocked_bytes(config: "Config", elements: int, dtype: str) -> int:
    if dtype not in {"mxfp4", "mxfp8"}:
        return ceil(elements * effective_element_bytes(config, dtype))
    quant = config.model.quantization
    raw = 0.5 if dtype == "mxfp4" else 1.0
    return ceil(elements / quant.block_size) * ceil(
        quant.block_size * raw + quant.scale_bytes
    )


def local_kv_heads(kv_heads: int, tp: int) -> int:
    return kv_heads // tp if kv_heads % tp == 0 else 1


def gemm(
    name: str, m: int, k: int, n: int, ctx: OperatorContext,
    output_bytes: float | None = None,
) -> WorkItem:
    model = ctx.model
    ba = effective_element_bytes(ctx.config, model.activation_dtype)
    bw = effective_element_bytes(ctx.config, model.weight_dtype)
    return _gemm(
        name, m, k, n, ba, bw, output_bytes or ba, ctx.occurrence_count
    )
