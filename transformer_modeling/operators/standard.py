"""Embedding、Norm、Residual、输出与未建模算子。"""

from __future__ import annotations

from math import ceil

from ..core.work_item import WorkItem
from ..parallel.tensor import (
    require_divisible, tp_all_gather_logits, tp_embedding_all_reduce,
)
from .base import (
    OperatorContext, OperatorEstimate,
    TransformerOperator, blocked_bytes, effective_element_bytes, gemm,
)


class TokenEmbeddingOperator(TransformerOperator):
    type_id, chinese_name, slot = "token_embedding", "词嵌入", "embedding"

    def validate(self, spec, config):
        super().validate(spec, config)
        require_divisible(
            "token_embedding padded_vocab_size",
            config.model.padded_vocab_size, config.parallelism.tensor_parallel,
        )

    def estimate(self, spec, ctx):
        model, tp = ctx.model, ctx.tp
        tied = bool(spec.get("tied_lm_head", True))
        global_params = model.padded_vocab_size * model.hidden_size
        local_params = ceil(global_params / tp)
        ba = effective_element_bytes(ctx.config, model.activation_dtype)
        work = WorkItem(
            "embedding", "vector", ctx.rows * model.hidden_size,
            ctx.rows * model.hidden_size * ba, 1,
        )
        return OperatorEstimate(
            self.type_id, self.chinese_name, 1, global_params, local_params,
            {"embedding_parameters": local_params},
            temporary_bytes=ceil(ctx.rows * model.hidden_size * ba),
            work_items=[work], communication_requests=tp_embedding_all_reduce(ctx),
            assumptions=[f"LM Head权重共享={tied}", "词表并行Embedding输出在TP组内归约。"],
        )


class NormOperator(TransformerOperator):
    slot = "norm"

    def estimate(self, spec, ctx):
        h, rows, count = ctx.model.hidden_size, ctx.rows, ctx.occurrence_count
        ba = effective_element_bytes(ctx.config, ctx.model.activation_dtype)
        ops_per = 5 if self.type_id == "rms_norm" else 8
        # RMSNorm与后续线性层融合时，归一化结果不写回HBM（统计pass只读一次）。
        fused = (
            self.type_id == "rms_norm"
            and ctx.config.execution.rmsnorm_linear_fused
        )
        traffic_factor = 1 if fused else 2
        item = WorkItem(
            f"layers.{self.type_id}", "vector", ops_per * rows * h * count,
            traffic_factor * rows * h * ba * count, count,
        )
        return OperatorEstimate(
            self.type_id, self.chinese_name, count, h * count, h * count,
            {"norm_parameters": h * count},
            temporary_bytes=ceil(rows * h * ba), work_items=[item],
        )


class RMSNormOperator(NormOperator):
    type_id, chinese_name = "rms_norm", "RMSNorm"


class LayerNormOperator(NormOperator):
    type_id, chinese_name = "layer_norm", "LayerNorm"


class StandardResidualOperator(TransformerOperator):
    type_id, chinese_name, slot = "standard_residual", "残差连接", "residual"

    def estimate(self, spec, ctx):
        elements = ctx.rows * ctx.model.hidden_size * ctx.occurrence_count
        ba = effective_element_bytes(ctx.config, ctx.model.activation_dtype)
        item = WorkItem("layers.residual", "vector", elements, 3 * elements * ba,
                        ctx.occurrence_count, elements)
        return OperatorEstimate(
            self.type_id, self.chinese_name, ctx.occurrence_count,
            temporary_bytes=ceil(ctx.rows * ctx.model.hidden_size * ba),
            work_items=[item], confidence="high",
        )


class AttnResOperator(TransformerOperator):
    type_id, chinese_name, slot = "attnres", "AttnRes近似", "residual"

    def validate(self, spec, config):
        super().validate(spec, config)
        blocks = int(spec.get("block_count", 8))
        if blocks <= 0:
            raise ValueError("attnres.block_count must be greater than zero")

    def estimate(self, spec, ctx):
        blocks = int(spec.get("block_count", 8))
        # 深度越深，可见的已完成block越多；工作量用平均可见数量，
        # 临时空间则按最深处的全部block加当前partial block计算。
        visible = (blocks + 1) / 2
        peak_visible = blocks + 1
        h, count = ctx.model.hidden_size, ctx.occurrence_count
        ba = effective_element_bytes(ctx.config, ctx.model.activation_dtype)
        # 每个AttnRes子层包含内部RMSNorm、pseudo-query打分、softmax和加权聚合。
        ops = int((9 * ctx.rows * h * visible + 5 * ctx.rows * visible) * count)
        # Block AttnRes two-phase近似按每子层5.5d的memory I/O计账，
        # 避免把所有历史block在每层重复物化和读写。
        traffic = int(5.5 * ctx.rows * h * ba * count)
        temporary = blocked_bytes(
            ctx.config, ctx.rows * peak_visible * h, ctx.model.activation_dtype
        )
        parameters = 2 * h * count  # learned pseudo-query + AttnRes内部RMSNorm
        return OperatorEstimate(
            self.type_id, self.chinese_name, count,
            global_parameters=parameters,
            local_parameters=parameters,
            parameter_breakdown={"attnres_parameters": parameters},
            persistent_state_bytes=0,
            state_breakdown={},
            temporary_bytes=temporary,
            work_items=[WorkItem("layers.attnres", "vector", ops, traffic, count, ops)],
            assumptions=[
                "Block AttnRes在Attention/FFN前构造输入。",
                "block representation属于单次前向临时空间，不作为跨token持久状态。",
                "工作量按平均可见block数、流量按two-phase 5.5d/子层近似。",
            ], confidence="low",
        )


class MHCOperator(TransformerOperator):
    """mHC：n通道流形约束超连接，B*经Sinkhorn归一化后做通道混合。"""

    type_id, chinese_name, slot = "mhc", "mHC流形约束超连接", "residual"

    def validate(self, spec, config):
        super().validate(spec, config)
        if int(spec.get("channels", 4)) <= 0:
            raise ValueError("mhc.channels must be greater than zero")
        if int(spec.get("sinkhorn_iters", 20)) < 0:
            raise ValueError("mhc.sinkhorn_iters cannot be negative")

    def estimate(self, spec, ctx):
        channels = int(spec.get("channels", 4))
        iters = int(spec.get("sinkhorn_iters", 20))
        h, rows, count = ctx.model.hidden_size, ctx.rows, ctx.occurrence_count
        ba = effective_element_bytes(ctx.config, ctx.model.activation_dtype)
        # B*·X通道混合 + A·X归并 + C广播回写，Sinkhorn迭代只作用于n×n矩阵。
        ops = (
            2 * rows * h * channels * channels
            + 4 * rows * h * channels
            + 4 * channels * channels * iters
        ) * count
        traffic = (2 * channels + 1) * rows * h * ba * count
        parameters = 3 * channels * channels * count
        return OperatorEstimate(
            self.type_id, self.chinese_name, count,
            global_parameters=parameters,
            local_parameters=parameters,
            parameter_breakdown={"mhc_parameters": parameters},
            temporary_bytes=ceil(rows * h * channels * ba),
            work_items=[WorkItem("layers.mhc", "vector", ops, traffic, count, ops)],
            assumptions=[
                f"Sinkhorn {iters}次迭代融入残差混合kernel，不单独启动。",
                "mHC中间激活按重计算处理，不作为持久状态缓存。",
                f"{channels}通道残差流按峰值临时空间计入容量。",
            ], confidence="low",
        )


class LMHeadOperator(TransformerOperator):
    type_id, chinese_name, slot = "lm_head", "LM Head", "output"

    def validate(self, spec, config):
        super().validate(spec, config)
        require_divisible(
            "lm_head padded_vocab_size",
            config.model.padded_vocab_size, config.parallelism.tensor_parallel,
        )

    def estimate(self, spec, ctx):
        model, tp = ctx.model, ctx.tp
        rows = ctx.batch_size if (
            ctx.phase == "decode" or model.prefill_logits_mode == "last_token"
        ) else ctx.rows
        local_vocab = model.padded_vocab_size // tp
        params = model.padded_vocab_size * model.hidden_size
        work = gemm("lm_head", rows, model.hidden_size, local_vocab, ctx, model.logits_bytes)
        return OperatorEstimate(
            self.type_id, self.chinese_name, 1, params, ceil(params / tp),
            {"lm_head_parameters": ceil(params / tp)},
            temporary_bytes=ceil(rows * model.padded_vocab_size * model.logits_bytes),
            work_items=[work], communication_requests=tp_all_gather_logits(ctx, local_vocab, rows),
        )


class SamplingOperator(TransformerOperator):
    type_id, chinese_name, slot = "sampling", "采样", "output"

    def estimate(self, spec, ctx):
        rows, vocab = ctx.batch_size, ctx.model.padded_vocab_size
        ops = 5 * rows * vocab
        item = WorkItem("sampling", "vector", ops, rows * vocab * ctx.model.logits_bytes, 1, ops)
        return OperatorEstimate(self.type_id, self.chinese_name, 1, work_items=[item])


class UnmodeledOperator(TransformerOperator):
    type_id, chinese_name = "unmodeled", "未建模算子"

    def estimate(self, spec, ctx):
        params = int(spec.get("parameter_count", 0)) * ctx.occurrence_count
        state = int(spec.get("state_bytes", 0)) * ctx.occurrence_count
        return OperatorEstimate(
            self.type_id, str(spec.get("name", self.chinese_name)), ctx.occurrence_count,
            params, params, {"unmodeled_parameters": params}, state,
            {"unmodeled_state_bytes": state}, assumptions=[str(spec.get(
                "note", "仅计容量；延迟只对应已建模代理结构，不构成严格上下界。"
            ))], confidence="unmodeled", performance_complete=False,
        )
