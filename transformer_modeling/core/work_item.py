"""Operator workload generation and single-device Roofline costing."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any



# 算子英文标识保持稳定，中文名称仅用于阅读结果。
OPERATOR_LABELS = {
    "layers.qkv_projection": "各层QKV投影",
    "layers.attention": "各层注意力核心",
    "layers.attention_output_projection": "各层注意力输出投影",
    "layers.kda_projection": "KDA输入投影",
    "layers.kda_gate": "KDA数据依赖门控",
    "layers.kda_short_conv": "KDA ShortConv",
    "layers.kda_recurrence": "KDA状态扫描/更新",
    "layers.kda_output": "KDA输出投影",
    "layers.mla_q_a": "Gated MLA Query压缩",
    "layers.mla_q_b": "Gated MLA Query展开",
    "layers.mla_kv_a": "Gated MLA KV压缩",
    "layers.mla_kv_b": "Gated MLA KV展开",
    "layers.mla_attention": "Gated MLA注意力核心",
    "layers.mla_gate": "Gated MLA选择门",
    "layers.mla_output": "Gated MLA输出投影",
    "layers.attnres": "Block Attention Residuals",
    "layers.kda.situ_latent": "KDA后SiTU/LatentMoE近似",
    "layers.mla.situ_latent": "MLA后SiTU/LatentMoE近似",
    "layers.ffn_gate_up": "各层FFN门控与上投影",
    "layers.ffn_up": "各层FFN上投影",
    "layers.ffn_down": "各层FFN下投影",
    "layers.moe_router": "各层MoE路由投影",
    "layers.moe_dispatch": "各层MoE Top-K路由与分派",
    "layers.moe_combine": "各层MoE专家输出加权合并",
    "layers.moe_expert_gate_up": "各层MoE路由专家门控与上投影",
    "layers.moe_expert_up": "各层MoE路由专家上投影",
    "layers.moe_expert_down": "各层MoE路由专家下投影",
    "layers.moe_shared_gate_up": "各层MoE共享专家门控与上投影",
    "layers.moe_shared_up": "各层MoE共享专家上投影",
    "layers.moe_shared_down": "各层MoE共享专家下投影",
    "layers.norms": "各层归一化",
    "final_norm": "最终归一化",
    "lm_head": "词表输出层",
    "sampling": "采样",
}


@dataclass(frozen=True)
class WorkItem:
    name: str
    kind: str
    executed_ops: float
    hbm_payload_bytes: float
    kernel_count: int
    logical_ops: float | None = None
    gemm_shape: tuple[int, int, int] | None = None


def _gemm(
    name: str,
    m: int,
    k: int,
    n: int,
    a_bytes: float, #输入矩阵A每个元素占多少字节
    weight_bytes: float, #权重矩阵W每个元素占多少字节
    output_bytes: float, #输出矩阵每个元素占多少字节
    repeats: int,
    output_elements: int | None = None,
) -> WorkItem:
    ops = 2 * m * k * n * repeats  #计算矩阵运算量
    output_elements = m * n if output_elements is None else output_elements #输出矩阵元素个数
    traffic = (m * k * a_bytes + k * n * weight_bytes + output_elements * output_bytes) * repeats  
    #计算HBM流量=读输入矩阵A+读权重矩阵W+写输出矩阵
    return WorkItem(
        name=name,
        kind="gemm",
        executed_ops=ops, #存运算量
        logical_ops=ops,
        hbm_payload_bytes=traffic, #存访存量
        kernel_count=repeats,
        gemm_shape=(m, n, k),
    )
