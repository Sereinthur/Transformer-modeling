"""Tensor Parallel 切分与通信声明。

本模块集中处理粗粒度解析模型中 TP 相关的两类计算：
- 通信声明：All-Reduce / All-Gather 的 payload 与 CommunicationRequest 构造
- 切分校验：算子维度对 TP 的整除性检查

切分后的 GEMM 维度（影响 ops/bytes 账单）仍由各算子自行计算，
因为切哪个维度取决于算子自身的几何结构。
"""

from .communication import (
    tp_all_reduce_hidden,
    tp_all_gather_logits,
    tp_embedding_all_reduce,
    require_divisible,
)

__all__ = [
    "tp_all_reduce_hidden",
    "tp_all_gather_logits",
    "tp_embedding_all_reduce",
    "require_divisible",
]
