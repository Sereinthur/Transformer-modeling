"""基于可替换算子的Transformer部署与性能解析模型。"""

from .config import Config
from .estimators import estimate

__all__ = ["Config", "estimate"]
__version__ = "0.6.0"
