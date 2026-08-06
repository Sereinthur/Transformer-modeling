"""纯滑窗Attention算子。"""

from __future__ import annotations

from .attention import StandardAttentionOperator
from .base import optional_int


class SlidingWindowAttentionOperator(StandardAttentionOperator):
    """仅访问固定局部窗口，账单复用标准Attention的窗口分支。"""

    type_id, chinese_name = "sliding_window_attention", "纯滑窗Attention"

    def validate(self, spec, config):
        super().validate(spec, config)
        if optional_int(spec, "sliding_window") <= 0:
            raise ValueError("sliding_window_attention sliding_window must be positive")
