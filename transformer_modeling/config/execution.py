"""并行方式与执行效率配置。

本模块聚合两类配置：
- ParallelSpec：并行拓扑（TP/PP/EP/微批/Stage边界）
- ExecutionSpec：执行侧假设（效率系数/重叠系数/融合开关/KV内存/推理模式）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .common import fraction, positive

_fraction = fraction
_positive = positive


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                          ParallelSpec — 并行拓扑                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
@dataclass(frozen=True)
class ParallelSpec:
    """Tensor Parallel 与 Pipeline Parallel 配置。"""

    # ── 基础并行度 ────────────────────────────────────────────────────────
    tensor_parallel: int            # 张量并行度（切分矩阵的 GPU 数）
    kv_head_policy: str             # KV 头分配策略（首版仅支持 shard_or_group_replicate）
    pipeline_parallel: int = 1      # 流水并行 Stage 数量
    expert_parallel: int = 1        # 专家并行度；只对 MoE 算子生效

    # ── 流水调度 ────────────────────────────────────────────────────────
    pipeline_microbatches: int = 1  # 一个固定 Batch 拆分的流水微批数量
    pipeline_stage_boundaries: tuple[int, ...] | None = None  # 可选的累计层边界

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParallelSpec":
        boundaries = data.get("pipeline_stage_boundaries")
        if boundaries is not None and not isinstance(boundaries, list):
            raise ValueError("parallelism.pipeline_stage_boundaries must be an array")
        spec = cls(
            tensor_parallel=int(data.get("tensor_parallel", 1)),
            kv_head_policy=str(
                data.get("kv_head_policy", "shard_or_group_replicate")
            ).lower(),
            pipeline_parallel=int(data.get("pipeline_parallel", 1)),
            pipeline_microbatches=int(data.get("pipeline_microbatches", 1)),
            expert_parallel=int(data.get("expert_parallel", 1)),
            pipeline_stage_boundaries=(
                tuple(int(value) for value in boundaries) if boundaries else None
            ),
        )
        _positive("parallelism.tensor_parallel", spec.tensor_parallel)
        _positive("parallelism.pipeline_parallel", spec.pipeline_parallel)
        _positive("parallelism.pipeline_microbatches", spec.pipeline_microbatches)
        _positive("parallelism.expert_parallel", spec.expert_parallel)
        if spec.kv_head_policy != "shard_or_group_replicate":
            raise ValueError(
                "parallelism.kv_head_policy must be shard_or_group_replicate"
            )
        return spec


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         ExecutionSpec — 执行假设                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
@dataclass(frozen=True)
class ExecutionSpec:
    """融合策略、有效利用率和计算/访存重叠假设。

    效率系数 / 重叠系数 / 融合开关 / KV 内存 / 推理模式。
    """

    # ── 1. 效率系数（经验初值，需按硬件/框架/精度/Shape 校准）─────────────
    prefill_gemm_efficiency: float          # Prefill GEMM 有效利用率（初值 0.65）
    decode_gemm_efficiency: float           # Decode  GEMM 有效利用率（初值 0.20）
    prefill_attention_efficiency: float     # Prefill Attention 有效利用率（初值 0.50）
    decode_attention_efficiency: float      # Decode  Attention 有效利用率（初值 0.15）
    vector_efficiency: float                # 向量/element-wise 算子利用率（初值 0.15）
    hbm_efficiency: float                   # HBM payload 带宽有效利用率（初值 0.75）

    # ── 2. 重叠插值系数 ρ ∈ [0,1]（0=取 max 完全重叠，1=取 sum 完全串行）──
    overlap_rho: float                      # 单算子内 计算 vs HBM 重叠
    prefill_tp_overlap_rho: float           # Prefill 阶段 TP 通信 vs 本地计算 重叠
    decode_tp_overlap_rho: float            # Decode  阶段 TP 通信 vs 本地计算 重叠
    ep_overlap_rho: float                   # MoE 计算 vs All-to-All 重叠

    # ── 3. 融合开关（影响 kernel_count 与 hbm_payload_bytes）──────────────
    flash_attention: bool                   # 是否使用 FlashAttention（不物化 S 矩阵）
    rope_kv_write_fused: bool               # RoPE 与 KV Cache 写入是否融合
    gated_mlp_fused: bool                   # gate 和 up 投影是否融合为一个 kernel
    rmsnorm_linear_fused: bool              # RMSNorm 是否与后续线性层融合

    # ── 4. KV 内存配置 ───────────────────────────────────────────────────
    kv_paged: bool                          # KV Cache 是否分页管理
    kv_page_tokens: int                     # KV Cache 每页的 token 数

    # ── 5. 推理模式 ─────────────────────────────────────────────────────
    prefill_logits_mode: str                # Prefill logits 模式：last_token / all_prompt_tokens

    # ── 6. 可选：GEMM 形状感知效率表 ─────────────────────────────────────
    #   [[rows, efficiency], ...]，按 GEMM 的 M 维在 log2 域插值；
    #   配置后对 gemm 类算子覆盖 Prefill/Decode 常数效率，未配置时回退常数。
    gemm_efficiency_by_rows: tuple[tuple[int, float], ...] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], model_data: dict[str, Any]) -> "ExecutionSpec":
        # 按 JSON 子节点分组读取
        efficiencies = data.get("efficiencies", {})
        overlap = data.get("overlap", {})
        fusion = data.get("fusion", {})
        memory = data.get("memory", {})
        inference = model_data.get("inference", {})

        rows_table = efficiencies.get("gemm_by_rows")
        parsed_table: tuple[tuple[int, float], ...] | None = None
        if rows_table is not None:
            if not isinstance(rows_table, list) or not rows_table:
                raise ValueError(
                    "execution.efficiencies.gemm_by_rows must be a non-empty array"
                )
            points = []
            for entry in rows_table:
                if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                    raise ValueError(
                        "execution.efficiencies.gemm_by_rows entries must be [rows, efficiency]"
                    )
                rows_value, efficiency_value = int(entry[0]), float(entry[1])
                _positive("execution.efficiencies.gemm_by_rows rows", rows_value)
                _fraction("execution.efficiencies.gemm_by_rows efficiency", efficiency_value)
                points.append((rows_value, efficiency_value))
            if [item[0] for item in points] != sorted({item[0] for item in points}):
                raise ValueError(
                    "execution.efficiencies.gemm_by_rows rows must be strictly increasing"
                )
            parsed_table = tuple(points)

        spec = cls(
            # 1. 效率系数
            prefill_gemm_efficiency=float(efficiencies.get("prefill_gemm", 0.65)),
            decode_gemm_efficiency=float(efficiencies.get("decode_gemm", 0.20)),
            prefill_attention_efficiency=float(
                efficiencies.get("prefill_attention", 0.50)
            ),
            decode_attention_efficiency=float(
                efficiencies.get("decode_attention", 0.15)
            ),
            vector_efficiency=float(efficiencies.get("vector", 0.15)),
            hbm_efficiency=float(efficiencies.get("hbm", 0.75)),
            # 2. 重叠系数
            overlap_rho=float(overlap.get("interpolation_rho", 0.10)),
            prefill_tp_overlap_rho=float(
                overlap.get("prefill_tp_interpolation_rho", overlap.get("tp_interpolation_rho", 0.5))
            ),
            decode_tp_overlap_rho=float(
                overlap.get("decode_tp_interpolation_rho", overlap.get("tp_interpolation_rho", 1.0))
            ),
            ep_overlap_rho=float(overlap.get("ep_interpolation_rho", 1.0)),
            # 3. 融合开关
            flash_attention=bool(fusion.get("flash_attention", True)),
            rope_kv_write_fused=bool(fusion.get("rope_kv_write", True)),
            gated_mlp_fused=bool(fusion.get("gated_mlp", True)),
            rmsnorm_linear_fused=bool(fusion.get("rmsnorm_linear", True)),
            # 4. KV 内存
            kv_paged=bool(memory.get("kv_paged", True)),
            kv_page_tokens=int(memory.get("kv_page_tokens", 16)),
            # 5. 推理模式
            prefill_logits_mode=str(inference.get("prefill_logits_mode", "last_token")),
            # 6. 可选形状感知效率表
            gemm_efficiency_by_rows=parsed_table,
        )

        # ── 校验：效率系数必须在 [0,1] ───────────────────────────────────
        for name in (
            "prefill_gemm_efficiency",
            "decode_gemm_efficiency",
            "prefill_attention_efficiency",
            "decode_attention_efficiency",
            "vector_efficiency",
            "hbm_efficiency",
        ):
            _fraction(f"execution.{name}", getattr(spec, name))

        # ── 校验：重叠系数必须在 [0,1] ───────────────────────────────────
        for name in (
            "overlap_rho",
            "prefill_tp_overlap_rho",
            "decode_tp_overlap_rho",
            "ep_overlap_rho",
        ):
            value = getattr(spec, name)
            if not 0 <= value <= 1:
                raise ValueError(f"execution.overlap.{name} must be in [0, 1]")

        # ── 校验：推理模式与 KV 分页 ─────────────────────────────────────
        if spec.prefill_logits_mode not in {"last_token", "all_prompt_tokens"}:
            raise ValueError(
                "model.inference.prefill_logits_mode must be last_token or all_prompt_tokens"
            )
        if spec.kv_paged:
            _positive("execution.memory.kv_page_tokens", spec.kv_page_tokens)
        return spec
