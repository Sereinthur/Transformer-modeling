"""统一估算入口。"""

from __future__ import annotations

from typing import Any

from ..config import Config
from .composed import estimate_composed


def estimate(config: Config, details: bool = True) -> dict[str, Any]:
    return estimate_composed(config, details)
