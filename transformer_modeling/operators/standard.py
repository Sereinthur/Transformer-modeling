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
    resolve_weight_dtype,
)


class TokenEmbeddingOperator(TransformerOperator):
    type_id, chinese_name, slot = "token_embedding", "词嵌入", "embedding"

    def validate(self, spec, config):
        super().validate(spec, config)
        table_rows = int(spec.get("vocab_size", 0) or config.model.padded_vocab_size)
        embedding_dim = int(spec.get("embedding_dim", 0) or config.model.hidden_size)
        if table_rows <= 0:
            raise ValueError("token_embedding.vocab_size must be greater than zero")
        if embedding_dim != config.model.hidden_size:
            raise ValueError(
                "token_embedding.embedding_dim must equal model.dimensions.hidden_size; "
                "insert an embedding projection before using a different width"
            )
        # Preserve the familiar error for inherited model dimensions while
        # identifying an explicitly overridden embedding table precisely.
        dimension_name = (
            "token_embedding padded_vocab_size"
            if not int(spec.get("vocab_size", 0) or 0)
            else "token_embedding vocab_size"
        )
        require_divisible(dimension_name, table_rows, config.parallelism.tensor_parallel)
        weight_dtype = resolve_weight_dtype(config, spec)
        if bool(spec.get("tied_lm_head", True)) and (
            table_rows != config.model.padded_vocab_size
            or embedding_dim != config.model.hidden_size
            or weight_dtype != config.model.weight_dtype
        ):
            raise ValueError(
                "a tied token_embedding must use the model padded_vocab_size, "
                "hidden_size, and weight dtype"
            )

    def estimate(self, spec, ctx):
        model, tp = ctx.model, ctx.tp
        tied = bool(spec.get("tied_lm_head", True))
        table_rows = int(spec.get("vocab_size", 0) or model.padded_vocab_size)
        embedding_dim = int(spec.get("embedding_dim", 0) or model.hidden_size)
        weight_dtype = resolve_weight_dtype(ctx.config, spec)
        global_params = table_rows * embedding_dim
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
            assumptions=[
                f"LM Head权重共享={tied}",
                f"Embedding表行数={table_rows}，输出宽度={embedding_dim}",
                "词表并行Embedding输出在TP组内归约。",
            ],
            local_parameters_by_dtype={weight_dtype: local_params},
        )


class NormOperator(TransformerOperator):
    slot = "norm"

    def validate(self, spec, config):
        super().validate(spec, config)
        if float(spec.get("eps", 1e-6)) <= 0:
            raise ValueError(f"{self.type_id}.eps must be greater than zero")
        resolve_weight_dtype(config, spec)

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
        wdtype = resolve_weight_dtype(ctx.config, spec)
        return OperatorEstimate(
            self.type_id, self.chinese_name, count, h * count, h * count,
            {"norm_parameters": h * count},
            temporary_bytes=ceil(rows * h * ba), work_items=[item],
            local_parameters_by_dtype={wdtype: h * count},
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
    type_id, chinese_name, slot = "attnres", "AttnRes", "residual"

    def validate(self, spec, config):
        super().validate(spec, config)
        blocks = self._block_count(spec, config.model.layer_count)
        if blocks <= 0:
            raise ValueError("attnres.block_count must be greater than zero")

    @staticmethod
    def _block_count(spec, layer_count: int) -> int:
        block_size = int(spec.get("block_size", 0))
        explicit = spec.get("block_count")
        if block_size < 0:
            raise ValueError("attnres.block_size cannot be negative")
        if block_size:
            derived = ceil(layer_count / block_size)
            if explicit is not None and int(explicit) != derived:
                raise ValueError("attnres.block_count conflicts with layer_count/block_size")
            return derived
        return int(explicit if explicit is not None else 8)

    def estimate(self, spec, ctx):
        blocks = self._block_count(spec, ctx.model.layer_count)
        block_size = int(spec.get("block_size", 0))
        if not block_size:
            block_size = ceil(ctx.model.layer_count / blocks)
        h, layer_count = ctx.model.hidden_size, ctx.occurrence_count
        ba = effective_element_bytes(ctx.config, ctx.model.activation_dtype)
        calls: list[int] = []
        for layer_index in range(ctx.layer_start, ctx.layer_start + layer_count):
            banks_before = ceil(layer_index / block_size)
            if banks_before:
                calls.append(banks_before + 1)  # Attention前：bank + 当前prefix。
            banks_for_mlp = layer_index // block_size + 1
            calls.append(banks_for_mlp + 1)
        if ctx.include_output:
            calls.append(blocks + 1)
        call_count = len(calls)
        source_total = sum(calls)
        peak_visible = max(calls, default=1)
        # 每次聚合包含RMS归一化、pseudo-query打分、softmax和深度加权求和。
        ops = int(9 * ctx.rows * h * source_total + 5 * ctx.rows * source_total)
        traffic = int((2 * source_total + call_count) * ctx.rows * h * ba)
        temporary = blocked_bytes(
            ctx.config, ctx.rows * peak_visible * h, ctx.model.activation_dtype
        )
        # 每层Attention/MLP各一组RMSNorm+projection，最终输出再一组。
        parameters = 4 * h * layer_count + (2 * h if ctx.include_output else 0)
        return OperatorEstimate(
            self.type_id, self.chinese_name, call_count,
            global_parameters=parameters,
            local_parameters=parameters,
            parameter_breakdown={"attnres_parameters": parameters},
            persistent_state_bytes=0,
            state_breakdown={},
            temporary_bytes=temporary,
            work_items=[WorkItem(
                "layers.attnres", "vector", ops, traffic, call_count, ops
            )],
            assumptions=[
                f"按官方block_size={block_size}逐层计算可见bank，最终共有{blocks}个block。",
                "首层Attention前bank为空，不执行聚合；FFN前与最终输出聚合按源码计入。",
                "block representation属于单次前向临时空间，不作为跨token持久状态。",
                "聚合算术和HBM流量仍按等效向量操作估算。",
            ], confidence="medium",
        )


class MHCOperator(TransformerOperator):
    """mHC：n通道流形约束超连接，B*经Sinkhorn归一化后做通道混合。"""

    type_id, chinese_name, slot = "mhc", "mHC", "residual"

    def validate(self, spec, config):
        super().validate(spec, config)
        if int(spec.get("channels", 4)) <= 0:
            raise ValueError("mhc.channels must be greater than zero")
        if int(spec.get("sinkhorn_iters", 20)) < 0:
            raise ValueError("mhc.sinkhorn_iters cannot be negative")
        if float(spec.get("eps", 1e-6)) <= 0:
            raise ValueError("mhc.eps must be greater than zero")

    def estimate(self, spec, ctx):
        channels = int(spec.get("channels", 4))
        iters = int(spec.get("sinkhorn_iters", 20))
        h, rows, count = ctx.model.hidden_size, ctx.rows, ctx.occurrence_count
        ba = effective_element_bytes(ctx.config, ctx.model.activation_dtype)
        mix_width = (2 + channels) * channels
        hc_width = channels * h
        # Each mHC operator card is one hc_pre -> sublayer -> hc_post wrapper.
        # A DeepSeek-V4 block has two such cards: one after Attention and one
        # after MoE.  Keeping the unit cost here prevents a visual split from
        # silently doubling the old aggregate formula.
        layer_parameters = mix_width * hc_width + mix_width + 3
        head_parameters = channels * hc_width + channels + 1 if ctx.include_output else 0
        parameters = layer_parameters * count + head_parameters
        linear_ops = 2 * rows * hc_width * mix_width
        pre_post_ops = 2 * rows * h * (channels + channels * channels + channels)
        sinkhorn_ops = 4 * rows * channels * channels * max(1, iters)
        ops = (linear_ops + pre_post_ops + sinkhorn_ops) * count
        if ctx.include_output:
            ops += 2 * rows * hc_width * channels + 2 * rows * h * channels
        traffic = (3 * channels + channels * channels + 1) * rows * h * ba * count
        occurrences = count + (1 if ctx.include_output else 0)
        return OperatorEstimate(
            self.type_id, self.chinese_name, occurrences,
            global_parameters=parameters,
            local_parameters=parameters,
            parameter_breakdown={"mhc_parameters": parameters},
            temporary_bytes=ceil(rows * h * channels * ba),
            work_items=[WorkItem(
                "layers.mhc", "vector", ops, traffic, occurrences, ops
            )],
            assumptions=[
                "每个Attention和FFN均按hc_pre→子层→hc_post包裹。",
                f"hc_fn按官方[(2+n)n,n*d]矩阵计参；Sinkhorn迭代数={iters}。",
                "mHC中间激活按重计算处理，不作为持久状态缓存。",
                f"{channels}通道hidden state跨层和跨PP Stage持续传递。",
            ], confidence="medium",
            local_parameters_by_dtype={"fp32": parameters},
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
