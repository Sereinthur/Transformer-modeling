"""公共模型定义与映射接口。"""

from .catalog import get_preset, preset_catalog
from .resolver import resolve_model_definition

__all__ = ["get_preset", "preset_catalog", "resolve_model_definition"]
