"""固定请求负载与混合Prefix Cache配置。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .common import positive
from .deployment import PrefixCacheSpec


@dataclass(frozen=True)
class ServingSpec:
    """固定等长Batch的Prompt和输出长度。"""

    batch_size: int
    prompt_length: int
    output_length: int
    max_sequence_length: int
    prefix_cache: PrefixCacheSpec

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ServingSpec":
        def fixed_value(field: str, default: int) -> int:
            value = data.get(field, default)
            if isinstance(value, dict):
                if value.get("distribution", "fixed") != "fixed":
                    raise ValueError(f"only fixed {field} is supported in version 1")
                value = value.get("value", default)
            return int(value)

        spec = cls(
            batch_size=int(data.get("batch_size", 1)),
            prompt_length=fixed_value("prompt_length", 1),
            output_length=fixed_value("output_length", 1),
            max_sequence_length=int(data.get("max_sequence_length", 0)),
            prefix_cache=PrefixCacheSpec.from_dict(data.get("prefix_cache", {})),
        )
        positive("serving.batch_size", spec.batch_size)
        positive("serving.prompt_length", spec.prompt_length)
        positive("serving.output_length", spec.output_length)
        if spec.max_sequence_length:
            required = spec.prompt_length + spec.output_length - 1
            if required > spec.max_sequence_length:
                raise ValueError(
                    f"request needs sequence length {required}, exceeding max_sequence_length "
                    f"{spec.max_sequence_length}"
                )
        return spec
