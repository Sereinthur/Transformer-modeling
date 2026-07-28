"""可替换算子与固定Transformer层Pattern配置。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .common import positive


@dataclass(frozen=True)
class OperatorSpec:
    """一个粗粒度逻辑算子；专用字段统一保存在params中。"""

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
        implementation = str(
            source.pop("implementation", default_implementation)
        ).lower()
        nested = source.pop("params", {})
        if not isinstance(nested, dict):
            raise ValueError(f"operator {type_id}.params must be an object")
        return cls(type_id, implementation, {**nested, **source})

    def get(self, name: str, default: Any = None) -> Any:
        return self.params.get(name, default)

    def signature(self) -> tuple[str, str, str]:
        ordered = repr(sorted(self.params.items(), key=lambda item: item[0]))
        return self.type, self.implementation, ordered


@dataclass(frozen=True)
class LayerPatternSpec:
    """固定层骨架中的一组可替换槽位。"""

    repeat: int
    norm: OperatorSpec
    attention: OperatorSpec
    residual: OperatorSpec
    ffn: OperatorSpec

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LayerPatternSpec":
        repeat = int(data.get("repeat", 1))
        positive("model.layer_pattern.repeat", repeat)
        return cls(
            repeat=repeat,
            norm=OperatorSpec.from_dict(data.get("norm"), "rms_norm"),
            attention=OperatorSpec.from_dict(data.get("attention"), "standard_attention"),
            residual=OperatorSpec.from_dict(data.get("residual"), "standard_residual"),
            ffn=OperatorSpec.from_dict(data.get("ffn"), "gated_ffn"),
        )


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
