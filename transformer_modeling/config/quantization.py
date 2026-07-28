"""低精度块量化存储配置。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .common import positive


@dataclass(frozen=True)
class QuantizationSpec:
    block_size: int = 32
    scale_bytes: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuantizationSpec":
        spec = cls(int(data.get("block_size", 32)), int(data.get("scale_bytes", 1)))
        positive("model.quantization.block_size", spec.block_size)
        positive("model.quantization.scale_bytes", spec.scale_bytes)
        return spec
