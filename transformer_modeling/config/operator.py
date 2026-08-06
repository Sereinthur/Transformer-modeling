"""Schema v3: ordered, independently replaceable layer operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .common import positive


@dataclass(frozen=True)
class OperatorSpec:
    """A registered performance-model operator and its implementation parameters."""

    type: str
    implementation: str
    params: dict[str, Any]

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
        default_type: str,
        default_implementation: str = "default",
    ) -> "OperatorSpec":
        source = dict(data or {})
        type_id = str(source.pop("type", default_type)).lower()
        implementation = str(source.pop("implementation", default_implementation)).lower()
        nested = source.pop("params", {})
        if not isinstance(nested, dict):
            raise ValueError(f"operator {type_id}.params must be an object")
        return cls(type_id, implementation, {**nested, **source})

    def get(self, name: str, default: Any = None) -> Any:
        return self.params.get(name, default)

    def signature(self) -> tuple[str, str, str]:
        return self.type, self.implementation, repr(sorted(self.params.items(), key=lambda item: item[0]))


@dataclass(frozen=True)
class LayerOperationSpec:
    """One stable, ordered operator card in a layer segment."""

    id: str
    operator: OperatorSpec

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LayerOperationSpec":
        if not isinstance(data, dict):
            raise ValueError("layer operation must be an object")
        operation_id = str(data.get("id", "")).strip()
        if not operation_id:
            raise ValueError("layer operation.id is required")
        source = data.get("operator")
        if not isinstance(source, dict):
            raise ValueError(f"layer operation {operation_id}.operator must be an object")
        return cls(operation_id, OperatorSpec.from_dict(source, "unmodeled"))


@dataclass(frozen=True)
class LayerPatternSpec:
    """A repeatable ordered list of independently replaceable operators."""

    repeat: int
    operations: tuple[LayerOperationSpec, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LayerPatternSpec":
        if not isinstance(data, dict):
            raise ValueError("layer segment must be an object")
        expired = {"norm", "attention", "residual", "ffn", "residual_connections"} & set(data)
        if expired:
            fields = ", ".join(sorted(expired))
            raise ValueError(
                "configuration version expired: fixed layer fields are unsupported "
                f"({fields}); use operations"
            )
        if "operations" not in data:
            raise ValueError(
                "configuration version expired: layer segments must use a non-empty operations array"
            )
        raw_operations = data["operations"]
        if not isinstance(raw_operations, list) or not raw_operations:
            raise ValueError("layer operations must be a non-empty array")
        repeat = int(data.get("repeat", 1))
        positive("model.layer_pattern.repeat", repeat)
        operations = tuple(LayerOperationSpec.from_dict(item) for item in raw_operations)
        ids = [item.id for item in operations]
        if len(set(ids)) != len(ids):
            raise ValueError("layer operation ids must be unique")
        return cls(repeat=repeat, operations=operations)

    @property
    def main_operators(self) -> tuple[OperatorSpec, ...]:
        return tuple(item.operator for item in self.operations)


@dataclass(frozen=True)
class ModelOutputSpec:
    norm: OperatorSpec
    head: OperatorSpec
    sampling: OperatorSpec

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelOutputSpec":
        return cls(
            norm=OperatorSpec.from_dict(data.get("norm"), "rms_norm"),
            head=OperatorSpec.from_dict(data.get("head"), "lm_head"),
            sampling=OperatorSpec.from_dict(data.get("sampling"), "sampling"),
        )
