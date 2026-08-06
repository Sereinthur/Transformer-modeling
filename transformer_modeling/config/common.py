"""配置常量和通用校验。"""

from __future__ import annotations

from math import isfinite

DTYPE_BYTES = {
    "fp32": 4.0, "tf32": 4.0, "bf16": 2.0, "fp16": 2.0,
    "fp8": 1.0, "mxfp8": 1.0, "int8": 1.0,
    "int4": 0.5, "mxfp4": 0.5,
}


def positive(name: str, value: float | int) -> None:
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be greater than zero, got {value!r}")


def non_negative(name: str, value: float | int) -> None:
    if not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")


def fraction(name: str, value: float) -> None:
    if not isfinite(value) or not 0 < value <= 1:
        raise ValueError(f"{name} must be in (0, 1], got {value!r}")


def dtype_bytes(name: str) -> float:
    try:
        return DTYPE_BYTES[name.lower()]
    except KeyError as exc:
        supported = ", ".join(sorted(DTYPE_BYTES))
        raise ValueError(f"unsupported dtype {name!r}; supported: {supported}") from exc
